"""
Write a one-job plan.json plus zero-byte inputs under $OA_HOME/inputs so dry_run.py has something to plan without a
GPU, a model or a real project (used by the runtime image smoke check, gpu/image/smoke.sh).
Usage: OA_HOME=/tmp/oa python3 smoke_plan.py
"""
import json, sys
from pathlib import Path

here = Path(__file__).resolve().parent
for d in (here.parent / "remote", here):
    sys.path.insert(0, str(d))
import oa  # noqa: E402

JOB = {"id": "smoke-s1", "model": "wan22-14b", "variant": "flf", "project": "smoke", "scene": "smoke-s1", "width": 832, "height": 480,
       "fps": 16, "frames": 81, "steps": 4, "seconds": 5, "seed": 7, "prompt": "an old poet speaks", "negative": "blurry",
       "start": "smoke/start.png", "end": "smoke/end.png"}

oa.INPUTS.joinpath("smoke").mkdir(parents=True, exist_ok=True)
for name in ("start.png", "end.png"):
    (oa.INPUTS / "smoke" / name).write_bytes(b"")
(oa.INPUTS / "plan.json").write_text(json.dumps({"jobs": [JOB]}))
print("wrote", oa.INPUTS / "plan.json")
