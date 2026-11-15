"""
The whole server run, detached from SSH (nohup), model by model:

  bootstrap → download(model 1) → [prefetch model 2 in background] run(model 1) → model.done → …

  nohup python3 pipeline.py --models=wan22-14b,ltx23,infinitetalk,longcat-avatar,latentsync,musetalk &

The API tails logs/events.jsonl for the timeline and pulls out/<model>/ after each "model.done" event.
A failing model is recorded and skipped; experimental models get a time box.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import oa
from models import DEFAULT_ORDER, MODELS

PY = sys.executable
TIMEBOX_S = {"longcat-avatar": 60 * 60, "musetalk": 60 * 60, "qwen-image-edit-2511": 150 * 60, "flux2-dev": 150 * 60, "flux2-klein-4b": 45 * 60,
             "longcat-avatar-15": 60 * 60, "skyreels-v3-a2v": 180 * 60, "infinitetalk-q": 75 * 60, "minimax-h3": 330 * 60, "lora-train": 120 * 60}


def arg(name: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def downloader(model: str) -> subprocess.Popen | None:
    if not MODELS[model]["files"]:
        return None
    log = open(oa.LOGS / f"download_{model}.log", "a")
    return subprocess.Popen([PY, "download.py", model], stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    oa.ensure_dirs()
    models = (arg("models") or ",".join(DEFAULT_ORDER)).split(",")
    pid = f"pipeline-{int(time.time())}"  # unique: a server can run several stages
    oa.emit({"ev": "start", "id": pid, "phase": "pipeline", "label": f"اجرای کامل: {', '.join(models)}"})
    t0 = time.time()

    if "--skip-bootstrap" not in sys.argv:
        r = subprocess.run([PY, "bootstrap.py"])
        if r.returncode != 0:
            oa.emit({"ev": "end", "id": pid, "phase": "pipeline", "status": "failed", "ms": int((time.time() - t0) * 1000), "error": "bootstrap failed"})
            sys.exit(1)

    pending: dict[str, subprocess.Popen | None] = {models[0]: downloader(models[0])}
    for i, model in enumerate(models):
        dl = pending.pop(model, None)
        if dl is not None:
            dl.wait()  # first model: foreground; later ones were prefetched during the previous render
            if dl.returncode != 0:
                oa.emit({"ev": "mark", "phase": "model.done", "model": model, "status": "failed", "error": "download failed"})
                continue
        if i + 1 < len(models):
            pending[models[i + 1]] = downloader(models[i + 1])
        cmd = [PY, "run.py", model, *(a for a in sys.argv if a.startswith("--only="))]
        try:
            r = subprocess.run(cmd, timeout=TIMEBOX_S.get(model))
            status = "done" if r.returncode == 0 else "failed"
            err = None if r.returncode == 0 else f"run.py exit {r.returncode}"
        except subprocess.TimeoutExpired:
            status, err = "failed", f"time box {TIMEBOX_S[model] // 60} min exceeded"
            oa.comfy_alive() and oa.comfy_post("/interrupt", {})
        oa.emit({"ev": "mark", "phase": "model.done", "model": model, "status": status, "error": err})

    for p in pending.values():
        if p is not None:
            p.wait()
    oa.emit({"ev": "end", "id": pid, "phase": "pipeline", "status": "done", "ms": int((time.time() - t0) * 1000)})
    chain_next()


NEXT = oa.OA_HOME / "next_models"


def chain_next() -> None:
    """A queued next stage (written by our API, orchestrator.queueNextStage) starts right here, in the same process
    (pipeline.pid stays valid): the server must not sit idle when our local API is down (exp-011: 3.7 h idle, ~$3)."""
    if not NEXT.exists():
        return
    models = NEXT.read_text().strip()
    NEXT.unlink()
    if models:
        os.execv(PY, [PY, "pipeline.py", f"--models={models}"])


if __name__ == "__main__":
    main()
