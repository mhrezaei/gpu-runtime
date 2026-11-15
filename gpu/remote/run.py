"""
Render every job of one model from plan.json, timed per step:

  model.run      wall time of the whole model on N GPUs (meta: gpus, jobs) — the multi-GPU number
  model.warmup   per GPU: first short generation (weights → VRAM + kernel compile)
  render         one row per scene/variant (gpu, exec time, frames, resolution, steps, peak VRAM of that GPU)
  model.unload   per GPU: free VRAM before the next model

Multi-GPU: one ComfyUI per GPU (port 8188+i); a worker per GPU pulls jobs from one shared queue.

  python3 run.py wan22-14b [--only=job_id,job_id] [--skip-warmup]      (GPUs: $OA_GPUS or all)

  python3 run.py --drain                                               (only the rebuild queue, then exit)

  python3 run.py --agent [--fake-render] [--heartbeat-s=30] [--url-wait-s=600] [--exit-when-idle]
                                                                       (agent mode: schema-valid queue/ jobs only, see agent.py)

Outputs: $OA_HOME/out/<model>/<job_id>.mp4 + <job_id>.json; results appended to out/results.jsonl.
Jobs whose result json already exists are skipped, so a stopped run can simply be started again.

Rebuild queue ("render this scene again"): the API drops a full job as queue/<job_id>.json (inputs under
inputs/rebuild/<job_id>/). Every GPU worker takes queued jobs before its next planned job, so a rebuild waits at
most one render; after the pipeline the API runs `run.py --drain`. Claimed → queue/running/, then done/ or failed/.
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import oa
import workflows
from models import MODELS

INPUT_DIR = oa.COMFY_DIR / "input"


def load_plan() -> list[dict]:
    """plan.json (video) + plan_extra.json (stills, RIFE), both written by the API (gpu/plan.ts, gpu/image-plan.ts)."""
    jobs = json.loads((oa.INPUTS / "plan.json").read_text())["jobs"]
    extra = oa.INPUTS / "plan_extra.json"
    return jobs + (json.loads(extra.read_text())["jobs"] if extra.exists() else [])


def stage_input(job: dict, key: str) -> str | None:
    """Copy an input (relative to inputs/ or out/) into ComfyUI/input with a job-unique flat name."""
    rel = job.get(key)
    if not rel:
        return None
    src = (oa.OA_HOME / rel) if rel.startswith(("out/", "inputs/")) else (oa.INPUTS / rel)
    if not src.exists():
        raise FileNotFoundError(f"{key} input missing: {src}")
    name = f"{job['id']}_{key}{src.suffix}"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, INPUT_DIR / name)
    return name


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


_results_lock = threading.Lock()


def save_result(job: dict, video: Path, meta: dict) -> dict:
    out_dir = oa.OUT / job["model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{job['id']}{video.suffix if video.suffix in ('.png', '.flac', '.wav', '.mp3') else '.mp4'}"
    shutil.copyfile(video, dest)
    rec = {"job": job["id"], "model": job["model"], "variant": job["variant"], "project": job["project"], "scene": job["scene"],
           "file": str(dest.relative_to(oa.OA_HOME)), "seconds": None if dest.suffix != ".mp4" else oa.video_seconds(dest), **meta}
    if job.get("video_id"):  # M1: the video by name; `project` (the same id) stays for older readers
        rec["video_id"] = job["video_id"]
    if job.get("model") in AUDIO_KINDS:  # exp-012: what the importer needs for a music track / scene ambience
        rec.update({"audio_kind": job["variant"], "prompt": job.get("prompt"), "seconds": job.get("seconds"), "seed": job.get("seed")})
    if job.get("kind"):  # stills: what the importer needs to place the image
        rec.update({"kind": job["kind"], "ref_key": job.get("ref_key"), "prompt": job.get("prompt"), "refs": job.get("refs", []), "steps": job.get("steps"), "seed": job.get("seed"),
                    # docs/11: review inputs (harmonized expectations) and the candidate index of this frame
                    "qa": job.get("qa"), "candidate": job.get("candidate", 0)})
    (out_dir / f"{job['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    with _results_lock, open(oa.OUT / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ------------------------------------------------------------------ ComfyUI models

def comfy_ready(gpus: list[int], needs: str | None = None) -> dict:
    """ComfyUI up on every GPU; `needs`: a node class that must be loaded (a pack installed after ComfyUI started → restart)."""
    import bootstrap
    if needs and all(oa.comfy_alive(oa.comfy_url(g)) for g in gpus) and needs not in oa.comfy_get("/object_info", oa.comfy_url(gpus[0])):
        with oa.phase("comfy.restart", f"راه‌اندازی دوباره‌ی ComfyUI برای node تازه ({needs})"):
            subprocess.run(["pkill", "-f", "main.py --listen 127.0.0.1 --port 818"], check=False)
            t0 = time.time()
            while any(oa.comfy_alive(oa.comfy_url(g)) for g in gpus) and time.time() - t0 < 60:
                time.sleep(1)
    missing = [g for g in gpus if not oa.comfy_alive(oa.comfy_url(g))]
    if missing:
        with oa.phase("comfy.start", f"بالا آمدن ComfyUI روی کارت‌های {missing}") as m:
            m.update(bootstrap.comfy_start_all(missing))
    oi = oa.comfy_get("/object_info", oa.comfy_url(gpus[0]))
    if needs and needs not in oi:
        raise RuntimeError(f"node {needs} is not available after restart (see logs/comfyui_0.log)")
    return oi


def render_comfy(job: dict, oi: dict, attention: str, gpu: int = 0) -> tuple[Path, dict]:
    build = workflows.BUILDERS[(job["model"], job["variant"])]
    base = oa.comfy_url(gpu)
    staged = {k: stage_input(job, k) for k in ("start", "end", "audio", "video", "audio2", "mask1", "mask2")}
    j = {**job, **{k: v for k, v in staged.items() if v}}
    segments = job.get("segments") or [None]
    parts: list[Path] = []
    seg_meta = []
    for i, seg in enumerate(segments):
        sj = dict(j)
        if seg:  # Wan 10 s scenes: 2 × 81 frames; the second segment starts from the first one's last frame
            sj.update({"frames": seg["frames"], "id": f"{job['id']}_s{i}"})
            if seg.get("from_prev") and parts:
                last = INPUT_DIR / f"{job['id']}_seg{i}_start.png"
                ffmpeg("-sseof", "-0.1", "-i", str(parts[-1]), "-frames:v", "1", "-update", "1", str(last))
                sj["start"] = last.name
        if job["model"] == "wan22-14b":
            use_end = job["variant"] == "flf" and (not seg or seg.get("use_end"))
            api = workflows.wan22(sj, oi, start=sj["start"], end=sj.get("end") if use_end else None, frames=sj["frames"])
        elif job["model"] in ("infinitetalk", "longcat-avatar", "longcat-avatar-15", "skyreels-v3-a2v", "infinitetalk-q"):
            api = build(sj, oi, attention)
        else:
            api = build(sj, oi)
        t0 = time.time()
        r = oa.comfy_run(api, base=base)
        files = [p for p in oa.comfy_output_files(r["outputs"]) if p.suffix in (".mp4", ".webm", ".mov")]
        if not files:
            raise RuntimeError(f"no video output: {json.dumps(r['outputs'])[:500]}")
        best = max(files, key=lambda p: oa.video_seconds(p) or 0)  # LongCat saves several progressive outputs
        parts.append(best)
        seg_meta.append({"exec_ms": int((time.time() - t0) * 1000), "frames": sj.get("frames")})
    if len(parts) == 1:
        return parts[0], {"segments": seg_meta}
    joined = oa.OUT / "_tmp" / f"{job['id']}_joined.mp4"
    joined.parent.mkdir(parents=True, exist_ok=True)
    # drop the first frame of each following segment (it repeats the previous last frame)
    inputs, filt = [], []
    for k, p in enumerate(parts):
        inputs += ["-i", str(p)]
        filt.append(f"[{k}:v]{'trim=start_frame=1,setpts=PTS-STARTPTS' if k else 'null'}[v{k}]")
    ffmpeg(*inputs, "-filter_complex", ";".join(filt) + ";" + "".join(f"[v{k}]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]",
           "-map", "[v]", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(joined))
    return joined, {"segments": seg_meta}


def warmup_job(job: dict, gpu: int) -> dict:
    w = {**job, "id": f"warmup_{job['model']}_g{gpu}", "segments": None, "seconds": 1, "frames": 17}
    if job.get("audio"):
        src = oa.INPUTS / job["audio"] if not job["audio"].startswith(("out/", "inputs/")) else oa.OA_HOME / job["audio"]
        short = oa.INPUTS / f"_warmup_audio_g{gpu}.wav"
        ffmpeg("-i", str(src), "-t", "2", str(short))
        w["audio"] = short.name
    return w


# ------------------------------------------------------------------ rebuild queue

QUEUE = oa.OA_HOME / "queue"


def take_queued() -> tuple[dict, Path] | None:
    """Oldest queued job; claimed by renaming it into queue/running/ (atomic, so two GPU workers never share one)."""
    (QUEUE / "running").mkdir(parents=True, exist_ok=True)
    for f in sorted(QUEUE.glob("*.json"), key=lambda p: p.stat().st_mtime):
        dest = QUEUE / "running" / f.name
        try:
            f.rename(dest)
        except FileNotFoundError:
            continue
        return json.loads(dest.read_text()), dest
    return None


def free_gpu(gpu: int) -> None:
    if oa.comfy_alive(oa.comfy_url(gpu)):
        oa.comfy_post("/free", {"unload_models": True, "free_memory": True}, oa.comfy_url(gpu))


def drain_queue(gpu: int, current: str | None = None) -> int:
    """Render every queued rebuild on this GPU. `current`: the ComfyUI model this worker has loaded (freed and
    reloaded around a rebuild of another model; nothing is freed for a rebuild of the same model)."""
    n, last = 0, current
    while (got := take_queued()) is not None:
        job, claimed = got
        n += 1
        outcome = "failed"
        try:
            if MODELS.get(job["model"], {}).get("engine") != "comfy":
                raise RuntimeError(f"rebuild of {job['model']} is not supported (ComfyUI models only)")
            if job["model"] != last:
                free_gpu(gpu)
                last = job["model"]
            oi = comfy_ready([gpu])
            attention = (oa.OA_HOME / "attention.txt").read_text().strip() if (oa.OA_HOME / "attention.txt").exists() else "sdpa"
            with oa.phase("render", f"{job['scene']} · {job['variant']} · ساخت دوباره", gpu=gpu, model=job["model"], project=job["project"], scene=job["scene"], job=job["id"]) as m:
                path, meta = render_comfy(job, oi, attention, gpu)
                rec = save_result(job, path, {**meta, "gpu": gpu, "width": job["width"], "height": job["height"], "steps": job.get("steps"),
                                              "attention": attention, "seed": job.get("seed"), "rebuild_of": job.get("rebuild_of")})
                m.update({"file": rec["file"], "seconds": rec["seconds"], **meta})
            outcome = "done"
        except Exception as e:
            print(f"rebuild {job.get('id')} failed on gpu {gpu}: {e}", file=sys.stderr)
            job["error"] = f"{type(e).__name__}: {e}"[:2000]
            claimed.write_text(json.dumps(job, ensure_ascii=False))
        (QUEUE / outcome).mkdir(parents=True, exist_ok=True)
        claimed.rename(QUEUE / outcome / claimed.name)
    if n and last != current:
        free_gpu(gpu)  # leave the GPU as the planned work expects it
    return n


def drain_all() -> None:
    """`run.py --drain`: the pipeline is over; render the rebuild queue (one worker per GPU) and exit."""
    pid = f"drain-{int(time.time())}"
    oa.emit({"ev": "start", "id": pid, "phase": "pipeline", "label": "ساخت دوباره‌ی صحنه‌های درخواستی"})
    t0 = time.time()
    threads = [threading.Thread(target=drain_queue, args=(g,), daemon=True) for g in oa.gpu_ids()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    oa.emit({"ev": "end", "id": pid, "phase": "pipeline", "status": "done", "ms": int((time.time() - t0) * 1000)})


def run_parallel(model: str, jobs: list[dict], gpus: list[int], worker_setup, do_job) -> None:
    """One worker thread per GPU pulling from a shared queue; a failed job never stops the others."""
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu: int) -> None:
        try:
            worker_setup(gpu)
        except Exception as e:  # a failed warm-up (a short special-case run) must not skip the real jobs: they report their own errors
            print(f"gpu {gpu} warm-up failed, continuing with the jobs: {e}", file=sys.stderr)
        loaded = model if MODELS[model]["engine"] == "comfy" else None
        while True:
            drain_queue(gpu, loaded)  # requested rebuilds go before the next planned job
            try:
                job = q.get_nowait()
            except queue.Empty:
                if drain_queue(gpu, loaded):
                    continue
                return
            try:
                with oa.phase("render", f"{job['scene']} · {job['variant']}", gpu=gpu, model=model, project=job["project"], scene=job["scene"], job=job["id"]) as m:
                    m.update(do_job(job, gpu))
            except Exception as e:
                print(f"job {job['id']} failed on gpu {gpu}: {e}", file=sys.stderr)

    with oa.phase("model.run", f"{MODELS[model]['label']} روی {len(gpus)} کارت", model=model) as m:
        m.update({"gpus": gpus, "jobs": len(jobs)})
        threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def run_comfy_model(model: str, jobs: list[dict], skip_warmup: bool, gpus: list[int]) -> None:
    oi = comfy_ready(gpus, MODELS[model].get("needs_node"))
    attention = (oa.OA_HOME / "attention.txt").read_text().strip() if (oa.OA_HOME / "attention.txt").exists() else "sdpa"

    def setup(gpu: int) -> None:
        if skip_warmup or MODELS[model].get("warmup") is False:
            return
        with oa.phase("model.warmup", f"بارگذاری در GPU {gpu} + اولین تولید کوتاه ({MODELS[model]['label']})", gpu=gpu, model=model) as m:
            path, meta = render_comfy(warmup_job(jobs[0], gpu), oi, attention, gpu)
            m.update({"seconds": oa.video_seconds(path), **meta})

    def do(job: dict, gpu: int) -> dict:
        path, meta = render_comfy(job, oi, attention, gpu)
        rec = save_result(job, path, {**meta, "gpu": gpu, "width": job["width"], "height": job["height"], "steps": job.get("steps"), "attention": attention,
                                      **({"source": job["video"]} if job.get("video") else {})})
        return {"file": rec["file"], "seconds": rec["seconds"], **meta}

    run_parallel(model, jobs, gpus, setup, do)
    for g in gpus:
        with oa.phase("model.unload", f"آزاد کردن VRAM کارت {g}", gpu=g, model=model):
            oa.comfy_post("/free", {"unload_models": True, "free_memory": True}, oa.comfy_url(g))


# ------------------------------------------------------------------ stills (MODELS[...]["kind"] == "image")

def render_image(job: dict, oi: dict, gpu: int = 0) -> tuple[Path, dict]:
    """refs: paths relative to inputs/ (reference sheets) or out/<model>/<job>.png (this model's own earlier frame)."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for i, rel in enumerate(job.get("refs") or []):
        src = (oa.OA_HOME / rel) if rel.startswith(("out/", "inputs/")) else (oa.INPUTS / rel)
        if not src.exists():
            raise FileNotFoundError(f"reference missing (earlier frame failed?): {rel}")
        name = f"{job['id']}_ref{i}{src.suffix}"
        shutil.copyfile(src, INPUT_DIR / name)
        names.append(name)
    api = workflows.IMAGE_BUILDERS[job["model"]](job, oi, names)
    t0 = time.time()
    r = oa.comfy_run(api, base=oa.comfy_url(gpu))
    files = [p for p in oa.comfy_output_files(r["outputs"]) if p.suffix == ".png"]
    if not files:
        raise RuntimeError(f"no image output: {json.dumps(r['outputs'])[:500]}")
    return files[0], {"exec_ms": int((time.time() - t0) * 1000), "refs_used": len(names)}


AUDIO_KINDS = {"ace-step-15", "minimax-music3", "stable-audio3"}


def run_audio_model(model: str, jobs: list[dict], skip_warmup: bool, gpus: list[int]) -> None:
    """exp-012 music / ambience: short ComfyUI audio graphs, one after another on the first GPU."""
    oi = comfy_ready(gpus)
    gpu = gpus[0]
    with oa.phase("model.run", f"{MODELS[model]['label']} روی کارت {gpu}", model=model) as m:
        m.update({"jobs": len(jobs)})
        for job in jobs:
            try:
                with oa.phase("render", f"{job.get('scene') or job['project']} · {job['variant']}", gpu=gpu, model=model,
                              project=job["project"], scene=job.get("scene"), job=job["id"]) as rm:
                    api = workflows.AUDIO_BUILDERS[model](job, oi)
                    t0 = time.time()
                    r = oa.comfy_run(api, base=oa.comfy_url(gpu))
                    files = [p for p in oa.comfy_output_files(r["outputs"]) if p.suffix in (".flac", ".wav", ".mp3")]
                    if not files:
                        raise RuntimeError(f"no audio output: {json.dumps(r['outputs'])[:500]}")
                    meta = {"exec_ms": int((time.time() - t0) * 1000), "gpu": gpu}
                    rec = save_result(job, files[0], meta)
                    rm.update({"file": rec["file"], **meta})
            except Exception as e:  # noqa: BLE001
                print(f"job {job['id']} failed: {e}", file=sys.stderr)
    with oa.phase("model.unload", f"آزاد کردن VRAM کارت {gpu}", gpu=gpu, model=model):
        free_gpu(gpu)


def run_image_model(model: str, jobs: list[dict], skip_warmup: bool, gpus: list[int]) -> None:
    """One chain per project/variant (frames reference earlier frames of the same chain), chains spread over the GPUs."""
    oi = comfy_ready(gpus)
    # LoRAs trained in an earlier run come with the inputs (GPU queue, docs/11): ComfyUI only sees models/loras
    lora_dir = oa.COMFY_DIR / "models" / "loras"
    lora_dir.mkdir(parents=True, exist_ok=True)
    for f in (oa.INPUTS / "loras").glob("*.safetensors"):
        if not (lora_dir / f.name).exists():
            shutil.copyfile(f, lora_dir / f.name)
    chains: dict[str, list[dict]] = {}
    for j in jobs:
        chains.setdefault(j.get("chain") or j["project"], []).append(j)
    q: queue.Queue = queue.Queue()
    for c in chains.values():
        q.put(c)

    def worker(gpu: int) -> None:
        if not skip_warmup:
            j = jobs[0]
            w = {**j, "id": f"warmup_{model}_g{gpu}", "kind": None, "refs": [], "width": 512, "height": 512, "prompt": "a plain grey studio backdrop"}
            try:
                with oa.phase("model.warmup", f"بارگذاری در GPU {gpu} + اولین تصویر کوچک ({MODELS[model]['label']})", gpu=gpu, model=model):
                    render_image(w, oi, gpu)
            except Exception as e:
                print(f"warmup failed on gpu {gpu}, continuing with the jobs: {e}", file=sys.stderr)
        while True:
            drain_queue(gpu, model)
            try:
                chain = q.get_nowait()
            except queue.Empty:
                return
            for job in chain:
                try:
                    label = f"{job.get('scene') or job.get('ref_key')} · {job['kind']} · {job['variant']}"
                    with oa.phase("render", label, gpu=gpu, model=model, project=job["project"], scene=job.get("scene"), job=job["id"]) as m:
                        path, meta = render_image(job, oi, gpu)
                        rec = save_result(job, path, {**meta, "gpu": gpu, "width": job["width"], "height": job["height"]})
                        m.update({"file": rec["file"], **meta})
                except Exception as e:  # later frames of this chain fail fast on the missing reference
                    print(f"job {job['id']} failed on gpu {gpu}: {e}", file=sys.stderr)

    with oa.phase("model.run", f"{MODELS[model]['label']} روی {len(gpus)} کارت", model=model) as m:
        m.update({"gpus": gpus, "jobs": len(jobs), "chains": len(chains)})
        threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    for g in gpus:
        with oa.phase("model.unload", f"آزاد کردن VRAM کارت {g}", gpu=g, model=model):
            free_gpu(g)


# ------------------------------------------------------------------ repo models (post lip-sync)

def venv_python(key: str) -> Path:
    return oa.REPOS / key / ".venv" / "bin" / "python"


def blackwell() -> bool:
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    return any(float(x) >= 12 for x in r.stdout.split() if x.replace(".", "").isdigit())


def build_tools() -> None:
    """a C/C++ compiler for packages without wheels (LatentSync's deps failed with "command 'c++' failed", 2026-10-21)"""
    if shutil.which("c++") is None:
        oa.sh("export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq build-essential >/dev/null", log="setup.log")


def setup_latentsync() -> None:
    d = oa.REPOS / "latentsync"
    with oa.phase("model.setup", "LatentSync: کد، محیط پایتون و وابستگی‌ها", model="latentsync"):
        build_tools()
        if not (d / ".git").exists():  # the weights may already sit in d/checkpoints (download.py runs first): clone in place
            d.mkdir(parents=True, exist_ok=True)
            oa.sh(f"cd {d} && git init -q && (git remote add origin {MODELS['latentsync']['git']} || true) && git fetch -q --depth 1 origin HEAD && git checkout -q -f FETCH_HEAD",
                  log="latentsync.log")
        if not (d / ".deps_ok").exists():
            oa.sh(f"cd {d} && pip install -q uv && uv venv -q --clear -p 3.10 .venv && uv pip install -q -p .venv/bin/python -r requirements.txt && touch .deps_ok", log="latentsync.log")
        # its pinned torch has no kernels for Blackwell (RTX 50xx, sm_120: "no kernel image is available", 2026-10-21)
        if not (d / ".torch_cu128").exists() and blackwell():
            oa.sh(f"cd {d} && uv pip install -q -p .venv/bin/python torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && touch .torch_cu128",
                  log="latentsync.log")


def setup_musetalk() -> None:
    d = oa.REPOS / "musetalk"
    with oa.phase("model.setup", "MuseTalk: کد، محیط پایتون، mmcv/mmpose و وزن‌ها", model="musetalk"):
        build_tools()
        if not (d / ".git").exists():
            oa.sh(f"git clone -q --depth 1 {MODELS['musetalk']['git']} {d}", log="musetalk.log")
        if not (d / ".deps_ok").exists():  # marker, not the venv: a half-built venv (failed install) must be redone
            oa.sh(
                f"cd {d} && pip install -q uv && uv venv -q --clear -p 3.10 .venv && "
                "uv pip install -q -p .venv/bin/python torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && "
                # chumpy's setup.py imports pip: build it without isolation, with pip/setuptools in the venv
                "uv pip install -q -p .venv/bin/python pip 'setuptools<70' wheel && "
                "uv pip install -q -p .venv/bin/python --no-build-isolation chumpy && "
                "uv pip install -q -p .venv/bin/python -r requirements.txt openmim 'setuptools<70' && "  # mim imports pkg_resources
                ".venv/bin/mim install -q mmengine 'mmcv==2.0.1' 'mmdet==3.1.0' 'mmpose==1.1.0' && touch .deps_ok",
                log="musetalk.log")
        if not (d / "models" / "musetalkV15" / "unet.pth").exists():
            oa.sh(f"cd {d} && PATH=$PWD/.venv/bin:$PATH bash download_weights.sh", log="musetalk.log")


def run_lipsync(job: dict, gpu: int = 0) -> tuple[Path, dict]:
    src_video = oa.OA_HOME / job["video"] if job["video"].startswith(("out/", "inputs/")) else oa.INPUTS / job["video"]
    audio = oa.INPUTS / job["audio"]
    tmp = oa.OUT / "_tmp" / job["id"]
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / "out.mp4"
    if job["model"] == "latentsync":
        d = oa.REPOS / "latentsync"
        cmd = (f"cd {d} && .venv/bin/python -m scripts.inference --unet_config_path configs/unet/stage2_512.yaml "
               f"--inference_ckpt_path checkpoints/latentsync_unet.pt --inference_steps {job.get('steps') or 20} --guidance_scale 1.5 "
               f"--enable_deepcache --video_path {src_video} --audio_path {audio} --video_out_path {out}")
    else:
        d = oa.REPOS / "musetalk"
        cfg = tmp / "task.yaml"
        cfg.write_text(f"task_0:\n  video_path: \"{src_video}\"\n  audio_path: \"{audio}\"\n")
        cmd = (f"cd {d} && .venv/bin/python -m scripts.inference --inference_config {cfg} --result_dir {tmp} "
               f"--unet_model_path models/musetalkV15/unet.pth --unet_config models/musetalkV15/musetalk.json --version v15")
    # the repo venvs have no hf_transfer; with HF_HUB_ENABLE_HF_TRANSFER=1 (env.sh) every HF download fails (2026-10-21)
    oa.sh(cmd, log=f"{job['model']}.log", env={"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    if not out.exists():
        found = sorted(tmp.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise RuntimeError("lip-sync produced no mp4")
        out = found[-1]
    return out, {}


def run_repo_model(model: str, jobs: list[dict], skip_warmup: bool, gpus: list[int]) -> None:
    {"latentsync": setup_latentsync, "musetalk": setup_musetalk}[model]()

    def setup(gpu: int) -> None:
        if skip_warmup:
            return
        with oa.phase("model.warmup", f"اولین اجرای کوتاه روی GPU {gpu} ({MODELS[model]['label']})", gpu=gpu, model=model) as m:
            j = jobs[0]
            src = oa.OA_HOME / j["video"] if j["video"].startswith(("out/", "inputs/")) else oa.INPUTS / j["video"]
            short_v, short_a = oa.INPUTS / f"_warm_v_g{gpu}.mp4", oa.INPUTS / f"_warm_a_g{gpu}.wav"
            ffmpeg("-i", str(src), "-t", "2", "-an", str(short_v))
            ffmpeg("-i", str(oa.INPUTS / j["audio"]), "-t", "2", str(short_a))
            path, _ = run_lipsync({**j, "id": f"warmup_{model}_g{gpu}", "video": str(short_v.relative_to(oa.OA_HOME)), "audio": short_a.name}, gpu)
            m["seconds"] = oa.video_seconds(path)

    def do(job: dict, gpu: int) -> dict:
        path, meta = run_lipsync(job, gpu)
        rec = save_result(job, path, {**meta, "gpu": gpu, "source": job["video"], "steps": job.get("steps")})
        return {"file": rec["file"], "seconds": rec["seconds"]}

    run_parallel(model, jobs, gpus, setup, do)


if __name__ == "__main__":
    oa.ensure_dirs()
    if "--agent" in sys.argv:
        import agent
        sys.exit(agent.main(sys.argv))
    if "--drain" in sys.argv:
        drain_all()
        sys.exit(0)
    model = sys.argv[1]
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    jobs = [j for j in load_plan() if j["model"] == model and (not only or j["id"] in only)]
    if not jobs:
        raise SystemExit(f"no jobs for {model}")
    done = [j for j in jobs if (oa.OUT / model / f"{j['id']}.json").exists()]
    jobs = [j for j in jobs if j not in done]
    if done:
        print(f"{model}: {len(done)} job(s) already rendered, skipped", file=sys.stderr)
    if not jobs:
        drain_queue(oa.gpu_ids()[0])
        sys.exit(0)
    gpus = oa.gpu_ids()
    if MODELS[model]["engine"] == "lora":
        from lora import run_lora_model as runner
    else:
        runner = run_image_model if MODELS[model].get("kind") == "image" else run_audio_model if MODELS[model].get("kind") == "audio" else run_comfy_model if MODELS[model]["engine"] == "comfy" else run_repo_model
    runner(model, jobs, "--skip-warmup" in sys.argv, gpus)
