"""
exp-011 C: character LoRA pilot with ai-toolkit (ostris), one character per GPU in parallel.

Job (plan_extra.json, model "lora-train"): id, variant ("klein4b"), project, scene=None, dataset ("lora/<char>" under inputs/),
trigger ("oa_<char>"), steps. Base: FLUX.2 [klein] base 4B (Apache-2.0, ungated); the LoRA is then applied to the distilled
klein 4B in ComfyUI (workflows.flux2 job["loras"]). Output: ComfyUI/models/loras/<trigger>.safetensors + out/lora-train/<id>.{json,safetensors}.
ai-toolkit gets its own venv (torch 2.13 cu130 per its README), so the ComfyUI environment is untouched.
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import oa

AITK = oa.REPOS / "ai-toolkit"
COMMIT = "04733873cf1da8060fce01316ec45daedbd4a918"  # main, 2026-10-19
PY = AITK / ".venv" / "bin" / "python"


def setup() -> None:
    if (AITK / ".deps_ok").exists():
        return
    with oa.phase("install", "نصب ai-toolkit (venv جدا، torch 2.13 cu130)"):
        oa.REPOS.mkdir(parents=True, exist_ok=True)
        if not (AITK / ".git").exists():
            oa.sh(f"git clone -q https://github.com/ostris/ai-toolkit {AITK}", log="lora_install.log")
        oa.sh(f"git fetch -q origin && git checkout -q {COMMIT} && git submodule update -q --init --recursive", cwd=AITK, log="lora_install.log")
        # the CUDA image has no ensurepip (exp-011 run 1 failed here)
        oa.sh("export DEBIAN_FRONTEND=noninteractive; (apt-get install -y -qq python3-venv python3.12-venv >/dev/null || "
              "(apt-get update -qq && apt-get install -y -qq python3-venv python3.12-venv >/dev/null))", log="lora_install.log")
        oa.sh("rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -q -U pip && "
              ".venv/bin/pip install -q torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && "
              ".venv/bin/pip install -q -r requirements.txt && touch .deps_ok", cwd=AITK, log="lora_install.log")


def config(job: dict, out: Path) -> Path:
    name = job["trigger"]
    cfg = {"job": "extension", "config": {"name": name, "process": [{
        "type": "diffusion_trainer", "training_folder": str(out), "device": "cuda:0",
        "trigger_word": job["trigger"],
        "network": {"type": "lora", "linear": 16, "linear_alpha": 16},
        "save": {"dtype": "float16", "save_every": 10_000, "max_step_saves_to_keep": 1},
        "datasets": [{"folder_path": str(oa.INPUTS / job["dataset"]), "caption_ext": "txt", "caption_dropout_rate": 0.05,
                      "resolution": [512, 768, 1024]}],
        "train": {"batch_size": 1, "steps": int(job.get("steps") or 1500), "gradient_accumulation": 1, "train_unet": True,
                  "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch", "optimizer": "adamw8bit",
                  "lr": 1e-4, "timestep_type": "weighted", "dtype": "bf16", "cache_text_embeddings": True, "disable_sampling": True},
        "model": {"name_or_path": "black-forest-labs/FLUX.2-klein-base-4B", "arch": "flux2_klein_4b", "quantize": False},
    }]}, "meta": {"name": name, "version": "1.0"}}
    p = out / f"{name}.json"
    p.write_text(json.dumps(cfg, indent=1))
    return p


def train(job: dict, gpu: int) -> dict:
    out = oa.OA_HOME / "lora_train" / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(job, out)
    t0 = time.time()
    oa.sh(f"{PY} run.py {cfg}", cwd=AITK, env={"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_ENABLE_HF_TRANSFER": "0"}, log=f"lora_{job['trigger']}.log")
    files = sorted((out / job["trigger"]).glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"no LoRA written under {out}")
    lora = files[-1]
    loras = oa.COMFY_DIR / "models" / "loras"
    loras.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lora, loras / f"{job['trigger']}.safetensors")
    res = oa.OUT / "lora-train"
    res.mkdir(parents=True, exist_ok=True)
    dest = res / f"{job['id']}.safetensors"
    shutil.copyfile(lora, dest)
    images = len(list((oa.INPUTS / job["dataset"]).glob("*.png")))
    rec = {"job": job["id"], "model": "lora-train", "variant": job["variant"], "project": job["project"], "scene": None,
           "file": str(dest.relative_to(oa.OA_HOME)), "trigger": job["trigger"], "steps": job.get("steps"), "images": images,
           "exec_ms": int((time.time() - t0) * 1000), "gpu": gpu, "size_mb": round(dest.stat().st_size / 1e6, 1)}
    if job.get("video_id"):  # M1: the video by name; `project` (the same id) stays for older readers
        rec["video_id"] = job["video_id"]
    (res / f"{job['id']}.json").write_text(json.dumps(rec, indent=1))
    return rec


def run_lora_model(model: str, jobs: list[dict], skip_warmup: bool, gpus: list[int]) -> None:
    setup()
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu: int) -> None:
        if oa.comfy_alive(oa.comfy_url(gpu)):  # the previous model's weights would leave too little VRAM for training
            oa.comfy_post("/free", {"unload_models": True, "free_memory": True}, oa.comfy_url(gpu))
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            try:
                with oa.phase("render", f"آموزش LoRA شخصیت {job['trigger']} ({job.get('steps')} گام)", gpu=gpu, model=model,
                              project=job["project"], scene=None, job=job["id"]) as m:
                    m.update(train(job, gpu))
            except Exception as e:  # noqa: BLE001
                print(f"lora job {job['id']} failed on gpu {gpu}: {e}", flush=True)

    with oa.phase("model.run", f"آموزش LoRA روی {len(gpus)} کارت", model=model) as m:
        m.update({"gpus": gpus, "jobs": len(jobs)})
        threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
