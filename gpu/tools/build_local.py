"""
Build every (model, variant) workflow with a sample job and validate it on the local CPU ComfyUI (:8190).
Zero-byte placeholders stand in for model weights (at the dest paths from models.py) and inputs.
Usage: python3 -I gpu/tools/build_local.py <ComfyUI dir>
"""
import json, os, sys, urllib.request, urllib.error
from pathlib import Path

comfy = Path(sys.argv[1]).resolve()
here = Path(__file__).resolve().parent  # gpu/tools in the repo, /opt/oneart/bin in the runtime image
for d in (here.parent / "remote", here):
    sys.path.insert(0, str(d))
import models, workflows  # noqa: E402

API = "http://127.0.0.1:8190"
for m in models.MODELS.values():
    if m["engine"] != "comfy":
        continue
    for f in m["files"]:
        p = comfy / "models" / f["dest"] / Path(f["path"]).name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.exists() or p.write_bytes(b"")
for n in ("t_start.png", "t_end.png", "t_speech.wav", "t_clip.mp4", "t_mask.png"):
    (comfy / "input" / n).exists() or (comfy / "input" / n).write_bytes(b"")

oi = json.loads(urllib.request.urlopen(API + "/object_info").read())
job = {"id": "test", "project": "ferdowsi", "scene": "ferdowsi-s4", "width": 832, "height": 480, "fps": 16, "frames": 81,
       "steps": 4, "seconds": 5, "seed": 7, "prompt": "an old poet speaks", "negative": "blurry",
       "start": "t_start.png", "end": "t_end.png", "audio": "t_speech.wav", "video": "t_clip.mp4"}
VARIANTS = {("skyreels-v3-a2v", "a2v"): [{}, {"audio2": "t_speech.wav", "mask1": "t_mask.png", "mask2": "t_mask.png"}]}
ok = True
cases = [(m, v, b, x) for (m, v), b in workflows.BUILDERS.items() for x in VARIANTS.get((m, v), [{}])]
for model, variant, build, extra in cases:
    name = f"{model}/{variant}" + ("+" + "+".join(extra) if extra else "")
    try:
        api = build({**job, "model": model, "variant": variant, **({"frames": 124, "width": 1344, "height": 768} if model == "minimax-h3" else {}), **extra}, oi)
    except Exception as e:
        ok = False
        print(f"BUILD-FAIL {name}: {type(e).__name__}: {e}")
        continue
    Path(comfy.parent / "api" / f"built_{model}_{variant}.json").write_text(json.dumps(api, indent=1))
    # every file input must be one of ours: a leftover example file (man.png) validates locally but is missing on the server
    stray = [f"{n['class_type']}={v}" for n in api.values() for k, v in n["inputs"].items()
             if k in ("image", "audio", "video") and isinstance(v, str) and not v.startswith("t_")]
    if stray:
        ok = False
        print(f"STRAY-FILE {name}: {stray}")
        continue
    req = urllib.request.Request(API + "/prompt", data=json.dumps({"prompt": api}).encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req).read()
        print(f"VALID      {name} ({len(api)} nodes)")
    except urllib.error.HTTPError as e:
        ok = False
        b = json.loads(e.read())
        print(f"INVALID    {name}: {b.get('error', {}).get('message')} {b.get('error', {}).get('details', '')[:200]}")
        for nid, ne in (b.get("node_errors") or {}).items():
            print("     ", nid, ne.get("class_type"), [(x.get("type"), x.get("details")) for x in ne.get("errors", [])][:3])
# stills: every image model × variant × reference count (0 = reference sheet, 1, max)
for model, build in workflows.IMAGE_BUILDERS.items():
    cap = models.MODELS[model]["max_refs"]
    for variant in (("quality",) if model == "flux2-klein-4b" else ("quality", "fast")):
        for n in sorted({0, 1, cap}):
            name = f"{model}/{variant}/{n}refs"
            refs = ["t_start.png"] * n
            try:
                api = build({**job, "model": model, "variant": variant, "width": 1280, "height": 720, "steps": None}, oi, refs)
            except Exception as e:
                ok = False
                print(f"BUILD-FAIL {name}: {type(e).__name__}: {e}")
                continue
            Path(comfy.parent / "api" / f"built_{model}_{variant}_{n}.json").write_text(json.dumps(api, indent=1))
            req = urllib.request.Request(API + "/prompt", data=json.dumps({"prompt": api}).encode(), headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req).read()
                print(f"VALID      {name} ({len(api)} nodes)")
            except urllib.error.HTTPError as e:
                ok = False
                b = json.loads(e.read())
                print(f"INVALID    {name}: {b.get('error', {}).get('message')} {b.get('error', {}).get('details', '')[:200]}")
                for nid, ne in (b.get("node_errors") or {}).items():
                    print("     ", nid, ne.get("class_type"), [(x.get("type"), x.get("details")) for x in ne.get("errors", [])][:3])
for path, body in (("/interrupt", {}), ("/queue", {"clear": True})):
    urllib.request.urlopen(urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}))
# exp-012 audio graphs
for model, build in workflows.AUDIO_BUILDERS.items():
    name = f"audio/{model}"
    try:
        api = build({**job, "model": model, "variant": "music", "seconds": 30, "bpm": 100, "prompt": "playful orchestral comedy underscore"}, oi)
    except Exception as e:
        ok = False
        print(f"BUILD-FAIL {name}: {type(e).__name__}: {e}")
        continue
    req = urllib.request.Request(API + "/prompt", data=json.dumps({"prompt": api}).encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req).read()
        print(f"VALID      {name} ({len(api)} nodes)")
    except urllib.error.HTTPError as e:
        ok = False
        b = json.loads(e.read())
        print(f"INVALID    {name}: {b.get('error', {}).get('message')}")
        for nid, ne in (b.get("node_errors") or {}).items():
            print("     ", nid, ne.get("class_type"), [(x.get("type"), x.get("details")) for x in ne.get("errors", [])][:3])
sys.exit(0 if ok else 1)
