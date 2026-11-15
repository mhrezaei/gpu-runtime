"""
One-time server setup, every step timed (@@ events). Idempotent: re-running skips finished steps.

  python3 bootstrap.py            # base + ComfyUI + node packs, then starts ComfyUI
  python3 bootstrap.py --no-start
  python3 bootstrap.py --image-build STEP   # gpu/image/Dockerfile only: STEP = system|comfy|nodes|sage|aitoolkit|finish

On a server booted from the versioned runtime image (/opt/oneart/IMAGE_VERSION exists, built by gpu/image/Dockerfile)
nothing is installed: main() only verifies the image (selfcheck) and starts ComfyUI. The Dockerfile calls the same
install functions with the pins below, so these constants are the single source of truth for versions.

Versions are pinned to the commits the workflows were validated against locally (gpu/tools/build_local.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import oa

COMFY_COMMIT = "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"
NODES = [
    ("ComfyUI-WanVideoWrapper", "https://github.com/kijai/ComfyUI-WanVideoWrapper.git", "088128b224242e110d3906c6750e9a3a348a659b"),
    ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git", "f20bdd5337ace70abbec147bdb68ea345d340836"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", "4d907bee61e92c2e65af3bd6383a4e4d356126d1"),
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo.git", "220e1d45aa0469f518e982102fca4dca92a61862"),
    ("ComfyUI-MelBandRoFormer", "https://github.com/kijai/ComfyUI-MelBandRoFormer.git", "92c86854e6654f4aacc97484471af95c98ea16d4"),
    # RIFE: added mid-run (2026-10-20); has no requirements.txt, only requirements-no-cupy.txt (RIFE needs no cupy)
    ("ComfyUI-Frame-Interpolation", "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git", "26545cc2dd95bc3d27f056016300673bdeee78f5"),
]
# Not pinned by the validated bootstrap (it installed whatever PyPI had). Set SAGEATTENTION_VERSION (Dockerfile build
# ARG) to freeze a release; empty means the latest at build time, which the image then records in its manifest.
SAGEATTENTION_VERSION = os.environ.get("SAGEATTENTION_VERSION", "").strip()
EXPECTED_BASE_TORCH = "2.14.1"  # pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
EXPECTED_AITK_TORCH = "2.13.0"  # lora.setup(): torch 2.13 cu130 in the ai-toolkit venv
IMAGE_SCRIPTS = ["oa.py", "bootstrap.py", "run.py", "download.py", "models.py", "workflows.py", "pipeline.py", "lora.py", "validate_local.py", "dry_run.py"]
PY = sys.executable
PIP = f"{PY} -m pip install --no-cache-dir -q"


def git_checkout(url: str, dest, commit: str) -> None:
    if not (dest / ".git").exists():
        oa.sh(f"GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none {url} {dest}", log="setup.log")
    oa.sh(f"cd {dest} && git fetch -q origin {commit} 2>/dev/null; git -c advice.detachedHead=false checkout -q {commit}", log="setup.log")


WAV2VEC_OLD = """        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
"""
WAV2VEC_NEW = """        # oneart patch: transformers 5 no longer fills encoder hidden_states for a direct encoder call
        # (InfiniteTalk/LongCat then fail with "'NoneType' object is not subscriptable"); capture them with hooks:
        # [input of layer 0, output of every layer] == what transformers 4 returned.
        _hs, _hooks = [], []
        if output_hidden_states:
            _hooks.append(self.encoder.layers[0].register_forward_pre_hook(lambda m, a: _hs.append(a[0])))
            for _layer in self.encoder.layers:
                _hooks.append(_layer.register_forward_hook(lambda m, a, o: _hs.append(o[0] if isinstance(o, tuple) else o)))
        try:
            encoder_outputs = self.encoder(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        finally:
            for _h in _hooks:
                _h.remove()
        if output_hidden_states and getattr(encoder_outputs, "hidden_states", None) is None:
            encoder_outputs.hidden_states = tuple(_hs)
"""


def patch_wav2vec() -> bool:
    """WanVideoWrapper's wav2vec (InfiniteTalk, LongCat) under transformers 5: see WAV2VEC_NEW. Idempotent; True if it changed the file."""
    f = oa.COMFY_DIR / "custom_nodes" / "ComfyUI-WanVideoWrapper" / "multitalk" / "wav2vec2.py"
    if not f.exists():
        return False
    src = f.read_text()
    if "oneart patch" in src:
        return False
    if WAV2VEC_OLD not in src:
        raise RuntimeError("wav2vec2.py changed upstream: patch_wav2vec needs an update")
    f.write_text(src.replace(WAV2VEC_OLD, WAV2VEC_NEW))
    return True  # changed: a running ComfyUI must restart to load it


def comfy_start(gpu: int = 0, timeout_s: int = 600) -> dict:
    """Start (or reuse) the ComfyUI instance pinned to one GPU: CUDA_VISIBLE_DEVICES=gpu, port 8188+gpu, own temp dir."""
    base = oa.comfy_url(gpu)
    if oa.comfy_alive(base):
        return {"gpu": gpu, "already_running": True}
    port = oa.COMFY_BASE_PORT + gpu
    tmp = oa.OA_HOME / "tmp" / f"gpu{gpu}"
    tmp.mkdir(parents=True, exist_ok=True)
    log = open(oa.LOGS / f"comfyui_{gpu}.log", "a")
    subprocess.Popen([PY, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch", "--preview-method", "none",
                      "--temp-directory", str(tmp)],
                     cwd=oa.COMFY_DIR, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if oa.comfy_alive(base):
            nodes = oa.comfy_get("/object_info", base)
            if gpu == 0:
                (oa.OA_HOME / "object_info.json").write_text(json.dumps(nodes))
            return {"gpu": gpu, "nodes": len(nodes), "s": round(time.time() - t0, 1)}
        time.sleep(2)
    raise TimeoutError(f"ComfyUI on GPU {gpu} did not come up; see logs/comfyui_{gpu}.log")


def comfy_start_all(gpus: list[int]) -> dict:
    """Start one instance per GPU in parallel (restarting running ones if node code changed); returns per-GPU startup info."""
    from concurrent.futures import ThreadPoolExecutor
    flag = oa.OA_HOME / "comfy_restart_needed"
    if flag.exists():
        subprocess.run(["pkill", "-f", "main.py --listen 127.0.0.1 --port 818"], check=False)
        t0 = time.time()
        while any(oa.comfy_alive(oa.comfy_url(g)) for g in gpus) and time.time() - t0 < 60:
            time.sleep(1)
        flag.unlink()
    with ThreadPoolExecutor(len(gpus)) as ex:
        res = list(ex.map(comfy_start, gpus))
    return {"gpus": gpus, "instances": res}


def install_system() -> bool:
    """git, ffmpeg, rsync and friends; True if it installed something."""
    if subprocess.run("command -v ffmpeg && command -v git && command -v rsync", shell=True, capture_output=True).returncode == 0:
        return False
    oa.sh("export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq git ffmpeg rsync curl aria2 libgl1 libglib2.0-0 >/dev/null", log="setup.log")
    return True


def install_comfy() -> None:
    git_checkout("https://github.com/comfyanonymous/ComfyUI.git", oa.COMFY_DIR, COMFY_COMMIT)
    oa.sh(f"cd {oa.COMFY_DIR} && {PIP} -r requirements.txt", log="setup.log")


def install_nodes() -> bool:
    """Clone and pin every node pack, install its requirements, apply our patch; True if node code changed."""
    changed = False
    for name, url, commit in NODES:
        dest = oa.COMFY_DIR / "custom_nodes" / name
        changed |= not (dest / ".git").exists()
        git_checkout(url, dest, commit)
        req = next((dest / r for r in ("requirements.txt", "requirements-no-cupy.txt") if (dest / r).exists()), None)
        if req:
            oa.sh(f"cd {dest} && {PIP} -r {req.name}", log="setup.log")
    return patch_wav2vec() or changed


def install_sage() -> bool:
    pkg = f"sageattention=={SAGEATTENTION_VERSION}" if SAGEATTENTION_VERSION else "sageattention"
    r = subprocess.run(f"{PIP} {pkg} && {PY} -c 'import sageattention'", shell=True, capture_output=True, text=True)
    (oa.OA_HOME / "attention.txt").write_text("sageattn" if r.returncode == 0 else "sdpa")
    return r.returncode == 0


# ---------------------------------------------------------------- versioned runtime image


def _git_head(path: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _dist_version(python: str, dist: str) -> str | None:
    """Installed version of a distribution WITHOUT importing it (importing torch would initialise CUDA)."""
    code = f"import importlib.metadata as m; print(m.version({dist!r}))"
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def image_ready() -> bool:
    return oa.IMAGE_VERSION_FILE.exists()


def image_finish() -> None:
    """Last Dockerfile step: write IMAGE_VERSION and IMAGE_MANIFEST.json from what is really installed."""
    import lora

    manifest = {
        "image_version": os.environ.get("IMAGE_VERSION", "dev"),
        "comfy": _git_head(oa.COMFY_DIR),
        "nodes": {n: _git_head(oa.COMFY_DIR / "custom_nodes" / n) for n, _, _ in NODES},
        "aitoolkit": _git_head(lora.AITK) if lora.AITK.exists() else None,
        "torch": _dist_version(PY, "torch"),
        "aitoolkit_torch": _dist_version(str(lora.PY), "torch") if lora.PY.exists() else None,
        "sageattention": _dist_version(PY, "sageattention"),
    }
    oa.IMAGE_HOME.mkdir(parents=True, exist_ok=True)
    (oa.IMAGE_HOME / "IMAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    oa.IMAGE_VERSION_FILE.write_text(manifest["image_version"] + "\n")


def selfcheck() -> int:
    """Verify the image against the pins above without a GPU. Exit codes: 0 ok, 1 mismatch, 3 not an image."""
    import lora

    if not image_ready():
        print(f"NOT-IN-IMAGE {oa.IMAGE_VERSION_FILE} is missing: this server was not booted from the versioned runtime image")
        return 3
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")

    print(f"image {oa.IMAGE_VERSION_FILE.read_text().strip()} python {sys.version.split()[0]}")
    head = _git_head(oa.COMFY_DIR)
    check("comfyui", head == COMFY_COMMIT, f"{(head or 'missing')[:10]} expected {COMFY_COMMIT[:10]}")
    for name, _, commit in NODES:
        dest = oa.COMFY_DIR / "custom_nodes" / name
        head = _git_head(dest)
        check(f"node {name}", head == commit and any(dest.glob("*.py")), f"{(head or 'missing')[:10]} expected {commit[:10]}")
    wav = oa.COMFY_DIR / "custom_nodes" / "ComfyUI-WanVideoWrapper" / "multitalk" / "wav2vec2.py"
    check("wav2vec patch", wav.exists() and "oneart patch" in wav.read_text(), str(wav))
    torch_v = _dist_version(PY, "torch")
    check("torch", bool(torch_v) and torch_v.startswith(EXPECTED_BASE_TORCH), f"{torch_v} expected {EXPECTED_BASE_TORCH}")
    aitk = oa.IMAGE_HOME / "home" / "repos" / "ai-toolkit"  # where the image built it (OA_HOME/repos/ai-toolkit is a link to it)
    aitk = aitk if aitk.exists() else lora.AITK
    if aitk.exists():  # the runtime-lora image: verify its pins
        aitk_py = aitk / ".venv" / "bin" / "python"
        head = _git_head(aitk)
        check("ai-toolkit", head == lora.COMMIT and (aitk / ".deps_ok").exists(), f"{(head or 'missing')[:10]} expected {lora.COMMIT[:10]}")
        atorch = _dist_version(str(aitk_py), "torch") if aitk_py.exists() else None
        check("ai-toolkit torch", bool(atorch) and atorch.startswith(EXPECTED_AITK_TORCH), f"{atorch} expected {EXPECTED_AITK_TORCH}")
    else:  # the default image leaves it out; a LoRA job installs it on first use (lora.setup())
        print("INFO ai-toolkit: not in this image (installed by the first LoRA job)")
    for tool in ("git", "ffmpeg", "rsync"):
        check(f"tool {tool}", subprocess.run(["bash", "-c", f"command -v {tool}"], capture_output=True).returncode == 0, tool)
    bin_dir = oa.IMAGE_HOME / "bin"
    missing = [n for n in IMAGE_SCRIPTS if not (bin_dir / n).exists()]
    check("scripts", not missing, f"{bin_dir}" + (f" missing {missing}" if missing else ""))
    sage = _dist_version(PY, "sageattention")
    print(f"INFO sageattention: {sage or 'not installed (optional, runs fall back to sdpa)'}")
    failed = [n for n, ok, _ in results if not ok]
    print("selfcheck:", "FAILED " + ", ".join(failed) if failed else "ok")
    return 1 if failed else 0


def image_link_repos() -> None:
    """Make the image's ai-toolkit (built under IMAGE_HOME/home/repos) visible where lora.py expects it (OA_HOME/repos)."""
    src = oa.IMAGE_HOME / "home" / "repos" / "ai-toolkit"
    dst = oa.REPOS / "ai-toolkit"
    if src.exists() and not dst.exists() and src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=True)


def image_build(step: str) -> None:
    """Dockerfile entry: one install step per RUN so layers cache independently (see gpu/image/Dockerfile)."""
    oa.ensure_dirs()
    if step == "system":
        install_system()
    elif step == "comfy":
        install_comfy()
    elif step == "nodes":
        install_nodes()
    elif step == "sage":
        print("sageattention:", "installed" if install_sage() else "unavailable, sdpa fallback")
    elif step == "aitoolkit":
        import lora

        lora.setup()
    elif step == "finish":
        image_finish()
    else:
        raise SystemExit(f"unknown image build step {step!r}")


def main() -> None:
    oa.ensure_dirs()
    start = "--no-start" not in sys.argv

    if image_ready():  # versioned runtime image: nothing to install, only verify
        with oa.phase("setup.image", "ایمیج runtime آماده است؛ فقط بررسی") as m:
            image_link_repos()
            if selfcheck() != 0:
                raise RuntimeError("runtime image selfcheck failed; see the FAIL lines above")
            m["image_version"] = oa.IMAGE_VERSION_FILE.read_text().strip()
    else:
        install_all()
    (oa.COMFY_DIR / "models").mkdir(exist_ok=True)
    if start:
        gpus = oa.gpu_ids()
        with oa.phase("comfy.start", f"بالا آمدن ComfyUI روی {len(gpus)} کارت") as m:
            m.update(comfy_start_all(gpus))


def install_all() -> None:
    with oa.phase("setup.system", "پکیج‌های سیستم (ffmpeg، git، rsync)") as m:
        if not install_system():
            m["skipped"] = True

    with oa.phase("setup.python", "کتابخانه‌های پایتون پایه (huggingface_hub، hf_transfer)") as m:
        oa.sh(f"{PIP} -U 'huggingface_hub[hf_transfer]' hf_transfer", log="setup.log")
        info = subprocess.run([PY, "-c", "import torch,sys;print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"],
                              capture_output=True, text=True).stdout.strip()
        m["python_torch_cuda_gpu"] = info

    with oa.phase("setup.comfy", "نصب ComfyUI (نسخه‌ی ثابت)") as m:
        install_comfy()
        m["commit"] = COMFY_COMMIT[:10]

    with oa.phase("setup.nodes", "نصب node packها (WanVideoWrapper، KJNodes، VHS، LTXVideo)") as m:
        changed = install_nodes()
        m["nodes"] = [n[0] for n in NODES]
        m["changed"] = changed
        if changed:  # node code changed under a running ComfyUI (a later stage on the same server): restart it below
            (oa.OA_HOME / "comfy_restart_needed").write_text("1")

    with oa.phase("setup.sageattention", "SageAttention (اختیاری؛ برای سرعت)") as m:
        m["available"] = install_sage()


if __name__ == "__main__":
    os.environ.setdefault("PIP_ROOT_USER_ACTION", "ignore")
    if "--image-build" in sys.argv:
        image_build(sys.argv[sys.argv.index("--image-build") + 1])
    else:
        main()
