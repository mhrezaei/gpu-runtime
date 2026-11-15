"""
Validate API workflows against a local CPU ComfyUI (no GPU, no model weights).
Model/input files referenced by the workflow get empty placeholders so ComfyUI's
combo validation passes; a 200 from /prompt means graph + inputs are valid.
Usage: python3 -I validate_local.py <comfy_dir> <api.json>...
"""
import json, sys, urllib.request, urllib.error
from pathlib import Path

COMFY = "http://127.0.0.1:8190"
comfy_dir = Path(sys.argv[1])
oi = json.loads(urllib.request.urlopen(COMFY + "/object_info").read())

FOLDER_HINTS = {
    "unet_name": "diffusion_models", "ckpt_name": "checkpoints", "clip_name": "text_encoders", "clip_name1": "text_encoders",
    "vae_name": "vae", "lora_name": "loras", "text_encoder": "text_encoders", "model_name": "diffusion_models",
    "audio_encoder_name": "audio_encoders", "name": "model_patches", "upscale_model_name": "latent_upscale_models",
    "clip_vision": "clip_vision", "image": "../input", "audio": "../input", "video": "../input", "model": "diffusion_models",
}

def placeholders(api):
    made = []
    for nid, n in api.items():
        for k, v in n["inputs"].items():
            if isinstance(v, str) and "." in v and "/" not in v[:1] and len(v) < 200 and any(v.endswith(e) for e in (".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".png", ".jpg", ".wav", ".mp3", ".mp4", ".gguf")):
                folder = FOLDER_HINTS.get(k, "diffusion_models")
                p = (comfy_dir / "models" / folder / v).resolve()
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"")
                    made.append(str(p))
    return made

ok = True
for f in sys.argv[2:]:
    api = json.loads(Path(f).read_text())
    placeholders(api)
    req = urllib.request.Request(COMFY + "/prompt", data=json.dumps({"prompt": api}).encode(), headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req).read())
        print("VALID ", f, r.get("prompt_id"))
        urllib.request.urlopen(urllib.request.Request(COMFY + "/interrupt", data=b"{}", headers={"Content-Type": "application/json"}))
        urllib.request.urlopen(urllib.request.Request(COMFY + "/queue", data=json.dumps({"clear": True}).encode(), headers={"Content-Type": "application/json"}))
    except urllib.error.HTTPError as e:
        ok = False
        body = json.loads(e.read())
        print("INVALID", f, json.dumps(body.get("error"), ensure_ascii=False)[:300])
        for nid, ne in (body.get("node_errors") or {}).items():
            print("   ", nid, ne.get("class_type"), [(x.get("type"), x.get("details"))  for x in ne.get("errors", [])][:4])
sys.exit(0 if ok else 1)
