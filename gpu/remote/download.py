"""
Download model weights, one timed phase per model (bytes, seconds, MB/s), skipping files already complete.

  python3 download.py wan22-14b ltx23          # in order, foreground
  nohup python3 download.py ltx23 &            # prefetch while another model renders (events still go to logs/events.jsonl)
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import oa  # noqa: E402
from models import MODELS  # noqa: E402


def dest_dir(key: str, f: dict) -> Path:
    m = MODELS[key]
    base = oa.COMFY_DIR / "models" if m["engine"] == "comfy" else oa.REPOS / key
    return base / f["dest"]


def download_model(key: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    m = MODELS[key]
    api = HfApi()
    with oa.phase("model.download", f"دانلود وزن‌های {m['label']}", model=key) as meta:
        t0 = time.time()
        got = skipped = 0
        files = []
        for f in m["files"]:
            out = dest_dir(key, f) / Path(f["path"]).name
            info = api.get_paths_info(f["repo"], [f["path"]])[0]
            size = getattr(info, "size", None)
            if out.exists() and size and out.stat().st_size == size:
                skipped += size
                files.append({"file": out.name, "bytes": size, "skipped": True})
                continue
            ft = time.time()
            p = hf_hub_download(f["repo"], f["path"], local_dir=oa.MODELS / "_hf")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(p, out)
            n = out.stat().st_size
            got += n
            files.append({"file": out.name, "bytes": n, "s": round(time.time() - ft, 1), "mb_s": round(n / 1e6 / max(time.time() - ft, 0.01), 1)})
        dt = max(time.time() - t0, 0.01)
        meta.update({"bytes": got, "skipped_bytes": skipped, "gb": round(got / 1e9, 2), "mb_s": round(got / 1e6 / dt, 1), "files": files})
    shutil.rmtree(oa.MODELS / "_hf", ignore_errors=True)


if __name__ == "__main__":
    oa.ensure_dirs()
    for k in sys.argv[1:]:
        if k not in MODELS:
            raise SystemExit(f"unknown model {k}; known: {', '.join(MODELS)}")
        download_model(k)
