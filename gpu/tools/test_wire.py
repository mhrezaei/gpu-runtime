"""
The GPU wire (docs/milestones/M1 T1c.6 AC6): run.py and lora.py copy a job's `video_id` into its result unchanged,
keep `project`, and a job of before M1 (no `video_id`) gives the result it gave before. Stdlib only, no GPU, no ComfyUI.
Usage: python3 -I gpu/tools/test_wire.py
"""
import json, os, shutil, sys, tempfile
from pathlib import Path

home = Path(tempfile.mkdtemp(prefix="oa-wire-"))
os.environ["OA_HOME"] = str(home)
os.environ["COMFY_DIR"] = str(home / "ComfyUI")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote"))
import oa, run, lora  # noqa: E402

failed = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global failed
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    failed += not ok


# run.save_result: a still, with and without video_id
src = home / "src.png"
src.write_bytes(b"png")
job = {"id": "flux2-klein-4b__quality__ferdowsi__s1-start", "model": "flux2-klein-4b", "variant": "quality", "project": "ferdowsi",
       "video_id": "ferdowsi", "scene": "ferdowsi-s1", "kind": "start_frame", "ref_key": "start", "prompt": "p", "refs": [], "seed": 1}
rec = run.save_result(job, src, {"exec_ms": 5})
check("run: video_id copied", rec.get("video_id") == "ferdowsi", rec)
check("run: project kept", rec.get("project") == "ferdowsi", rec)
old = run.save_result({k: v for k, v in job.items() if k != "video_id"} | {"id": "old-job"}, src, {"exec_ms": 5})
check("run: no video_id on a job of before M1", "video_id" not in old, old)
check("run: otherwise the same record", {k: v for k, v in rec.items() if k not in ("video_id", "job", "file")} ==
      {k: v for k, v in old.items() if k not in ("job", "file")}, (rec, old))

# a lip-sync job: `video` (its source clip) is not the video id and passes as before
clip = home / "src.mp4"
clip.write_bytes(b"mp4")
oa.video_seconds = lambda _p: 1.0
talk = {"id": "latentsync__v2v__ferdowsi-s4", "model": "latentsync", "variant": "v2v", "project": "ferdowsi", "video_id": "ferdowsi",
        "scene": "ferdowsi-s4", "video": "ferdowsi/ferdowsi-s4/silent.mp4"}
t = run.save_result(talk, clip, {"source": talk["video"]})
check("run: lip-sync source untouched", t.get("source") == talk["video"] and t.get("video_id") == "ferdowsi", t)

# lora.train: the training itself is replaced by writing a fake weights file
def fake_sh(cmd, cwd=None, env=None, log=None):
    cfg = json.loads(Path(cmd.split()[-1]).read_text())
    out = Path(cfg["config"]["process"][0]["training_folder"]) / cfg["config"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "w.safetensors").write_bytes(b"weights")


lora.oa.sh = fake_sh
(oa.INPUTS / "lora" / "prj_ferdowsi" / "ferdowsi").mkdir(parents=True, exist_ok=True)
lj = {"id": "lora-train__klein4b__ferdowsi", "model": "lora-train", "variant": "klein4b", "project": "ferdowsi", "video_id": "ferdowsi",
      "scene": None, "dataset": "lora/prj_ferdowsi/ferdowsi", "trigger": "oa_ferdowsi_ferd", "steps": 10}
lr = lora.train(lj, 0)
check("lora: video_id copied", lr.get("video_id") == "ferdowsi", lr)
check("lora: project and trigger kept", lr.get("project") == "ferdowsi" and lr.get("trigger") == "oa_ferdowsi_ferd", lr)
saved = json.loads((oa.OUT / "lora-train" / f"{lj['id']}.json").read_text())
check("lora: the record on disk is the same", saved == lr, saved)
lo = lora.train({k: v for k, v in lj.items() if k != "video_id"} | {"id": "lora-train__klein4b__old"}, 0)
check("lora: no video_id on a job of before M1", "video_id" not in lo, lo)

shutil.rmtree(home, ignore_errors=True)
sys.exit(1 if failed else 0)
