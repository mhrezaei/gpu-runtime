"""
Dry run of plan.json on the local CPU ComfyUI: stage real inputs, build every ComfyUI job (each Wan segment too)
and validate it via /prompt without executing. Lip-sync jobs: checks their source/audio paths exist in the plan.
Env: OA_HOME=data/gpu COMFY_DIR=.local-comfy/ComfyUI COMFY_URL=http://127.0.0.1:8190
"""
import json, sys, urllib.request, urllib.error
from pathlib import Path

here = Path(__file__).resolve().parent  # gpu/tools in the repo, /opt/oneart/bin in the runtime image
for d in (here.parent / "remote", here):
    sys.path.insert(0, str(d))
import oa, run, workflows  # noqa: E402

oi = oa.comfy_get("/object_info")
jobs = run.load_plan()
outputs = {f"out/{j['model']}/{j['id']}.mp4" for j in jobs}
bad = 0
for job in jobs:
    if run.MODELS[job["model"]]["engine"] != "comfy":
        v = job["video"]
        ok = (v in outputs) if v.startswith("out/") else (oa.INPUTS / v).exists()
        ok = ok and (oa.INPUTS / job["audio"]).exists()
        print(("OK      " if ok else "MISSING ") + job["id"])
        bad += not ok
        continue
    if job.get("video", "").startswith("out/"):  # produced earlier on the server: check the dependency, stage a stand-in clip
        if job["video"] not in outputs:
            bad += 1
            print(f"MISSING-DEP {job['id']} <- {job['video']}")
            continue
        job = {**job, "video": "ferdowsi/ferdowsi-s4/silent_seedance-1-5-pro_v4.mp4"}
    staged = {k: run.stage_input(job, k) for k in ("start", "end", "audio", "video")}
    j = {**job, **{k: v for k, v in staged.items() if v}}
    segs = job.get("segments") or [None]
    for i, seg in enumerate(segs):
        sj = dict(j)
        if seg:
            sj.update({"frames": seg["frames"], "id": f"{job['id']}_s{i}"})
        if job["model"] == "wan22-14b":
            use_end = job["variant"] == "flf" and (not seg or seg.get("use_end"))
            api = workflows.wan22(sj, oi, start=sj["start"], end=sj.get("end") if use_end else None, frames=sj["frames"])
        elif job["model"] in ("infinitetalk", "longcat-avatar"):
            api = workflows.BUILDERS[(job["model"], job["variant"])](sj, oi, "sdpa")
        else:
            api = workflows.BUILDERS[(job["model"], job["variant"])](sj, oi)
        try:
            oa.comfy_post("/prompt", {"prompt": api})
            print(f"VALID   {sj['id']}")
        except RuntimeError as e:
            bad += 1
            print(f"INVALID {sj['id']}: {str(e)[:400]}")
        oa.comfy_post("/interrupt", {}); oa.comfy_post("/queue", {"clear": True})
print("problems:", bad)
sys.exit(1 if bad else 0)
