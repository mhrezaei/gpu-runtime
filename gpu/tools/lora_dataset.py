"""
Character-LoRA pilot (exp-011 C): build an ai-toolkit dataset from the gpt-image stills we already have. Free, local.
  - the selected reference sheet, whole + one crop per panel (sheets are N side-by-side turnaround views)
  - every gpt-image start/end frame of a scene the character appears in (selected and not)
Captions: scenario prompt_en with {char} → trigger word, other {refs} → their appearance/description.
Output: data/gpu/inputs/lora/<char>/NNN.png + NNN.txt; with --project=<projectId> (a new LoRA, M1 T1c.6):
data/gpu/inputs/lora/<projectId>/<char>/ and the trigger oa_<char>_<4 letters of the project id> (OQ-47), as
newLoraTrigger / loraDatasetOf in apps/api/src/models/loras/loras.constants.ts
Images are found through the running API (GET /v1/videos/<video>), never the database;
ONEART_API overrides the API address, ONEART_DATA_DIR the data folder (as for the API).
--dry lists the images and captions it would write and writes nothing.
Usage: python3 -I gpu/tools/lora_dataset.py <video> <character> [panels=4] [--heads] [--dry] [--project=<projectId>]
"""
import json, os, re, subprocess, sys, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
video, char = args[0], args[1]
panels = int(args[2]) if len(args) > 2 else 4
head_crops = "--heads" in sys.argv
dry = "--dry" in sys.argv
project = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--project=")), None)
trigger = f"oa_{char}_{project.removeprefix('prj_')[:4]}" if project else f"oa_{char}"
API = os.environ.get("ONEART_API", "http://127.0.0.1:4281").rstrip("/")
DATA = Path(os.environ.get("ONEART_DATA_DIR") or ROOT / "data")
if not DATA.is_absolute():
    DATA = ROOT / DATA
detail = json.loads(urllib.request.urlopen(f"{API}/v1/videos/{urllib.parse.quote(video)}").read())
sc_file = ROOT / "scenarios" / f"{video}.json"
if sc_file.exists():
    sc = json.loads(sc_file.read_text())
else:  # request-made videos keep their scenario in the database
    sc = {"characters": detail["scenario"]["characters"], "locations": detail["scenario"]["locations"],
          "scenes": [{**x["spec"], "id": x["id"]} for x in detail["scenes"]]}
desc = {c["id"]: c["appearance_en"] for c in sc["characters"]}
desc.update({l["id"]: l.get("appearance_en") or l.get("description_en") or l["id"] for l in sc["locations"]})
desc[char] = trigger

out = DATA / "gpu" / "inputs" / "lora" / (f"{project}/{char}" if project else char)
if not dry:
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        p.unlink()
planned: list[dict] = []
media = DATA / "media"
n = 0


def media_path(asset: dict) -> Path:
    """the file of an asset DTO: its url is /media/<path under the media folder>, each segment URL-encoded"""
    return media / urllib.parse.unquote(asset["url"].removeprefix("/media/"))


def hosted(asset: dict) -> bool:
    """a ready image from a hosted model (gpt-image), not one rendered on our own GPU server"""
    return asset["status"] == "ready" and asset.get("url") and not (asset.get("model") or "").startswith("selfhost/")


def put(src: Path, caption: str, crop: str | None = None) -> None:
    global n
    n += 1
    if dry:
        planned.append({"n": n, "src": str(src.relative_to(media)), "crop": crop, "caption": caption})
        return
    dst = out / f"{n:03d}.png"
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src)] + (["-vf", crop] if crop else []) + [str(dst)]
    subprocess.run(args, check=True)
    (out / f"{n:03d}.txt").write_text(caption)


def fill(text: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: desc.get(m.group(1), m.group(1)), text)


sheets = [a for a in detail["references"] if a["kind"] == "reference_sheet" and a["ref_key"] == char and hosted(a)]
# the selected sheet first, else the newest
sheets.sort(key=lambda a: a["created_at"], reverse=True)
sheets.sort(key=lambda a: a["selected"], reverse=True)
sheet = media_path(sheets[0]) if sheets else None
if sheet:
    base = f"{trigger}, character turnaround on a plain light background, glossy 3D animation"
    put(sheet, f"{base}, several views side by side")
    views = (["front view", "three-quarter view", "side view", "back view", "three-quarter view from the other side"] if panels == 5
             else ["front view", "three-quarter view", "side view", "back view"])[:panels]
    for i, v in enumerate(views):
        put(sheet, f"{base}, {v}, full body", f"crop=iw/{panels}:ih:iw*{i}/{panels}:0")
        if head_crops and v != "back view":  # few images (one sheet only): add head-and-shoulders crops for the face
            put(sheet, f"{trigger}, close-up portrait, head and shoulders, {v}, plain light background",
                f"crop=iw/{panels}:ih*0.38:iw*{i}/{panels}:0,scale=iw*2:-1")

scenes = {s["id"]: s for s in sc["scenes"]}
rows = [(x["id"], a["kind"], a) for x in sorted(detail["scenes"], key=lambda x: x["idx"])
        for a in sorted(x["assets"], key=lambda a: a["kind"]) if a["kind"] in ("start_frame", "end_frame") and hosted(a)]
for scene_id, kind, asset in rows:
    s = scenes.get(scene_id)
    if not s or char not in (s.get("characters") or []):
        continue
    frame = s[kind]
    cap = f"{fill(frame['prompt_en'])} {frame.get('framing_en', '')}".strip()
    put(media_path(asset), cap if trigger in cap else f"{trigger}, {cap}")
result = {"char": char, "trigger": trigger, "images": n, "dir": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)}
print(json.dumps({**result, "dry": True, "planned": planned}, ensure_ascii=False) if dry else json.dumps(result))
