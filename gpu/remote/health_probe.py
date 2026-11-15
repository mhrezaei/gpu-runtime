"""
Benchmark probe for a freshly rented server (M11 #638): prints ONE JSON line and exits 0, whatever fails.

  python3 health_probe.py <download_url> [download_seconds]

Fields (null when a step could not be measured): gpu_name, gpu_count, vram_mb (smallest card), cuda (driver CUDA version,
"12.8"), disk_free_gb (of /workspace, else /), download_mbps (timed GET of a public ungated file, capped at
download_seconds, default 20), torch_ok (torch.cuda.is_available() and a small matmul ran on the GPU), agent_version.
Nothing here needs a token; the download URL is public.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except Exception:  # noqa: BLE001 — a missing tool is a null field, not a crash
        return None


def gpu_info() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    rows = [ln.split(",") for ln in (out or "").strip().splitlines() if ln.strip()]
    if not rows:
        return {"gpu_name": None, "gpu_count": 0, "vram_mb": None}
    return {
        "gpu_name": rows[0][0].strip(),
        "gpu_count": len(rows),
        "vram_mb": min(int(float(r[1])) for r in rows),
    }


def cuda_version() -> str | None:
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", _run(["nvidia-smi"]) or "")
    return m.group(1) if m else None


def disk_free_gb() -> float:
    path = "/workspace" if os.path.isdir("/workspace") else "/"
    return round(shutil.disk_usage(path).free / 1e9, 1)


def download_mbps(url: str, seconds: float) -> float | None:
    try:
        got = 0
        t0 = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": "oneart-health-probe"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            while time.monotonic() - t0 < seconds:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
        dt = max(time.monotonic() - t0, 0.001)
        return round(got * 8 / dt / 1e6, 1)
    except Exception:  # noqa: BLE001
        return None


def torch_ok() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        a = torch.randn(512, 512, device="cuda")
        return bool(torch.isfinite((a @ a).sum()).item())
    except Exception:  # noqa: BLE001
        return False


def agent_version() -> str | None:
    for p in ("/opt/oneart/AGENT_VERSION", "/workspace/oneart/AGENT_VERSION"):
        try:
            with open(p, encoding="utf-8") as f:
                return f.read().strip() or None
        except OSError:
            continue
    return os.environ.get("ONEART_AGENT_VERSION") or None


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else ""
    seconds = float(argv[2]) if len(argv) > 2 else 20.0
    out = {
        **gpu_info(),
        "cuda": cuda_version(),
        "disk_free_gb": disk_free_gb(),
        "download_mbps": download_mbps(url, seconds) if url else None,
        "torch_ok": torch_ok(),
        "agent_version": agent_version(),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
