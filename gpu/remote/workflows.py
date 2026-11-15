"""
Workflow builders: official upstream workflows (workflows/upstream/*.json, UI format) are converted to API format
with the *server's own* /object_info (oa.ui_to_api), then patched with one job's inputs.

Job dict (from plan.json, see apps/api/src/services/gpu/plan.service.ts):
  id, model, variant, project, video_id, scene, width, height, fps, frames, steps, seed, prompt, negative,
  start, end, audio, video   (file names already copied into ComfyUI/input)

Wire contract (M1 T1c.6): `video_id` is the id of the video the job belongs to. `project` holds the same id: it is
what every job and result of before M1 carries, so it stays, and the importer reads `video_id`, else `project`.
`video` is NOT the video id: it is the source clip of a lip-sync / V2V job (a path). run.py and lora.py copy
`video_id` into each result unchanged.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import oa

UPSTREAM = Path(__file__).parent / "workflows" / "upstream"
_cache: dict[str, dict] = {}


def template(name: str, object_info: dict) -> dict:
    if name not in _cache:
        _cache[name] = oa.ui_to_api(json.loads((UPSTREAM / f"{name}.json").read_text()), object_info)
    return copy.deepcopy(_cache[name])


# ------------------------------------------------------------------ graph helpers

def find(api: dict, class_type: str, title: str | None = None, where: dict | None = None) -> list[str]:
    out = []
    for k, v in api.items():
        if v["class_type"] != class_type:
            continue
        if title is not None and v["_meta"]["title"] != title:
            continue
        if where and any(v["inputs"].get(a) != b for a, b in where.items()):
            continue
        out.append(k)
    return out


def one(api: dict, class_type: str, title: str | None = None, where: dict | None = None) -> str:
    ids = find(api, class_type, title, where)
    if len(ids) != 1:
        raise KeyError(f"expected one {class_type} {title or ''} {where or ''}, found {ids}")
    return ids[0]


def rewire(api: dict, old: list, new: list) -> None:
    """Every input that points at `old` ([id, slot]) now points at `new`."""
    for v in api.values():
        for k, x in v["inputs"].items():
            if x == old:
                v["inputs"][k] = new


def bypass(api: dict, node: str, through: str, slot: int = 0) -> None:
    """Remove a pass-through node: consumers of [node, slot] get the node's `through` input instead."""
    rewire(api, [node, slot], api[node]["inputs"][through])
    del api[node]


def consumers(api: dict, node: str) -> list[tuple[str, str]]:
    return [(k, name) for k, v in api.items() for name, x in v["inputs"].items() if isinstance(x, list) and x[:1] == [node]]


def save_prefix(job: dict) -> str:
    return f"oneart/{job['id']}"


# ------------------------------------------------------------------ Wan 2.2 A14B (native)

def wan22(job: dict, oi: dict, *, start: str, end: str | None, frames: int) -> dict:
    """First+last frame (end given) or first frame only. variant steps: 4 (lightx2v LoRAs) or 20 (base)."""
    api = template("video_wan2_2_14B_flf2v", oi)
    high = one(api, "UNETLoader", where={"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"})
    low = one(api, "UNETLoader", where={"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"})
    cond = one(api, "WanFirstLastFrameToVideo")
    k_high = next(k for k, _ in consumers(api, next(m for m, _ in consumers(api, high))) if api[k]["class_type"] == "KSamplerAdvanced")
    k_low = next(k for k, _ in consumers(api, next(m for m, _ in consumers(api, low))) if api[k]["class_type"] == "KSamplerAdvanced")

    steps = int(job.get("steps") or 4)
    if steps <= 8:  # distilled: lightx2v 4-step LoRA on both experts, cfg 1
        for unet, name, tag in ((high, "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "lora_high"),
                                (low, "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "lora_low")):
            api[tag] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [unet, 0], "lora_name": name, "strength_model": 1.0}, "_meta": {"title": tag}}
            for k, inp in consumers(api, unet):
                if k != tag:
                    api[k]["inputs"][inp] = [tag, 0]
        cfg = 1.0
    else:
        cfg = 3.5
    split = steps // 2
    seed = int(job.get("seed") or 42)
    api[k_high]["inputs"].update({"steps": steps, "cfg": cfg, "start_at_step": 0, "end_at_step": split, "noise_seed": seed})
    api[k_low]["inputs"].update({"steps": steps, "cfg": cfg, "start_at_step": split, "end_at_step": 10000, "noise_seed": seed})

    pos = api[cond]["inputs"]["positive"][0]
    neg = api[cond]["inputs"]["negative"][0]
    api[pos]["inputs"]["text"] = job["prompt"]
    api[neg]["inputs"]["text"] = job["negative"]
    start_node = api[cond]["inputs"]["start_image"][0]
    end_node = api[cond]["inputs"]["end_image"][0]
    api[start_node]["inputs"]["image"] = start
    if end:
        api[end_node]["inputs"]["image"] = end
    else:  # first frame only: swap to WanImageToVideo
        del api[end_node]
        i = api[cond]["inputs"]
        api[cond] = {"class_type": "WanImageToVideo", "_meta": {"title": "WanImageToVideo"}, "inputs": {
            "positive": i["positive"], "negative": i["negative"], "vae": i["vae"], "batch_size": 1, "start_image": i["start_image"]}}
    api[cond]["inputs"].update({"width": job["width"], "height": job["height"], "length": frames})
    api[one(api, "CreateVideo")]["inputs"]["fps"] = job.get("fps", 16)
    api[one(api, "SaveVideo")]["inputs"].update({"filename_prefix": save_prefix(job), "format": "mp4", "format.codec": "h264"})
    return api


# ------------------------------------------------------------------ LTX-2.3 (native)

def _primitive(api: dict, cls: str, title: str) -> str:
    ids = [k for k, v in api.items() if v["class_type"] == cls and v["_meta"]["title"].lower() == title.lower()]
    if len(ids) != 1:
        raise KeyError(f"{cls} '{title}': {ids}")
    return ids[0]


def ltx23_flf(job: dict, oi: dict) -> dict:
    api = template("video_ltx2_3_flf2v", oi)
    api[one(api, "LoadImage", title="Load First Frame")]["inputs"]["image"] = job["start"]
    api[one(api, "LoadImage", title="Load Last Frame")]["inputs"]["image"] = job["end"]
    texts = find(api, "CLIPTextEncode")
    neg_text = next(k for k in texts if api[k]["inputs"]["text"].startswith("blurry"))
    pos_text = next(k for k in texts if k != neg_text)
    api[pos_text]["inputs"]["text"] = job["prompt"]
    api[neg_text]["inputs"]["text"] = job["negative"]
    api[_primitive(api, "PrimitiveInt", "Width")]["inputs"]["value"] = job["width"]
    api[_primitive(api, "PrimitiveInt", "height")]["inputs"]["value"] = job["height"]
    api[_primitive(api, "PrimitiveInt", "Duration")]["inputs"]["value"] = int(job["seconds"])
    api[_primitive(api, "PrimitiveInt", "Frame Rate(int)")]["inputs"]["value"] = job.get("fps", 24)
    api[one(api, "RandomNoise")]["inputs"]["noise_seed"] = int(job.get("seed") or 42)
    api[one(api, "SaveVideo")]["inputs"].update({"filename_prefix": save_prefix(job), "format": "mp4", "format.codec": "h264"})
    return api


def ltx23_ia2v(job: dict, oi: dict) -> dict:
    """Image + our Persian audio → talking video. Uses the distilled checkpoint (drops the dev+distill-LoRA path)."""
    api = template("video_ltx2_3_ia2v", oi)
    for k in find(api, "CheckpointLoaderSimple") + find(api, "LTXAVTextEncoderLoader") + find(api, "LTXVAudioVAELoader"):
        api[k]["inputs"]["ckpt_name"] = "ltx-2.3-22b-distilled-fp8.safetensors"
    for k in find(api, "LoraLoaderModelOnly"):
        if "distilled" in api[k]["inputs"].get("lora_name", ""):
            bypass(api, k, "model")
    api[one(api, "LoadImage")]["inputs"]["image"] = job["start"]
    api[one(api, "LoadAudio")]["inputs"]["audio"] = job["audio"]
    api[_primitive(api, "PrimitiveStringMultiline", "Prompt")]["inputs"]["value"] = job["prompt"]
    for k in find(api, "PrimitiveBoolean"):
        t = api[k]["_meta"]["title"].lower()
        if "prompt enhan" in t or "text to video" in t:
            api[k]["inputs"]["value"] = False
    neg = next(k for k in find(api, "CLIPTextEncode") if isinstance(api[k]["inputs"].get("text"), str))
    api[neg]["inputs"]["text"] = job["negative"]
    api[_primitive(api, "PrimitiveInt", "Width")]["inputs"]["value"] = job["width"]
    api[_primitive(api, "PrimitiveInt", "Height")]["inputs"]["value"] = job["height"]
    api[_primitive(api, "PrimitiveInt", "Frame Rate")]["inputs"]["value"] = job.get("fps", 24)
    api[_primitive(api, "PrimitiveFloat", "Duration")]["inputs"]["value"] = float(job["seconds"])
    for k in find(api, "RandomNoise"):
        api[k]["inputs"]["noise_seed"] = int(job.get("seed") or 42)
    api[one(api, "SaveVideo")]["inputs"].update({"filename_prefix": save_prefix(job), "format": "mp4", "format.codec": "h264"})
    return oa.prune(api)


# ------------------------------------------------------------------ Kijai WanVideoWrapper: InfiniteTalk / LongCat

def _kijai_common(api: dict, job: dict, *, model: str, lora: str, attention: str) -> None:
    api[one(api, "WanVideoModelLoader")]["inputs"].update({"model": model, "attention_mode": attention, "quantization": "disabled"})
    for k in find(api, "WanVideoLoraSelect"):
        api[k]["inputs"]["lora"] = lora
    api[one(api, "WanVideoVAELoader")]["inputs"]["model_name"] = "Wan2_1_VAE_bf16.safetensors"
    te = one(api, "WanVideoTextEncodeCached")
    api[te]["inputs"].update({"model_name": "umt5-xxl-enc-bf16.safetensors", "positive_prompt": job["prompt"], "negative_prompt": job["negative"]})
    # our TTS is already clean speech: skip the vocal separator
    for k in find(api, "MelBandRoFormerSampler"):
        rewire(api, [k, 0], api[k]["inputs"]["audio"])
        del api[k]
    for k in find(api, "LoadAudio"):
        api[k]["inputs"]["audio"] = job["audio"]
    w = one(api, "INTConstant", title="Width")
    h = one(api, "INTConstant", title="Height")
    api[w]["inputs"]["value"] = job["width"]
    api[h]["inputs"]["value"] = job["height"]
    for k in find(api, "VHS_VideoCombine"):
        api[k]["inputs"].update({"filename_prefix": save_prefix(job), "save_output": True})
    for k in find(api, "WanVideoSampler") + find(api, "WanVideoSamplerv2"):
        if "seed" in api[k]["inputs"]:
            api[k]["inputs"]["seed"] = int(job.get("seed") or 42)


def infinitetalk_i2v(job: dict, oi: dict, attention: str = "sdpa") -> dict:
    api = template("kj_infinitetalk_i2v", oi)
    _kijai_common(api, job, model="Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                  lora="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", attention=attention)
    api[one(api, "MultiTalkModelLoader")]["inputs"]["model"] = "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
    api[one(api, "CLIPVisionLoader")]["inputs"]["clip_name"] = "clip_vision_h.safetensors"
    api[one(api, "LoadImage")]["inputs"]["image"] = job["start"]
    if job.get("steps"):
        api[one(api, "WanVideoSampler")]["inputs"]["steps"] = int(job["steps"])
    return oa.prune(api)


def infinitetalk_v2v(job: dict, oi: dict, attention: str = "sdpa") -> dict:
    api = template("kj_infinitetalk_v2v", oi)
    _kijai_common(api, job, model="Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                  lora="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", attention=attention)
    api[one(api, "MultiTalkModelLoader")]["inputs"]["model"] = "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
    api[one(api, "CLIPVisionLoader")]["inputs"]["clip_name"] = "clip_vision_h.safetensors"
    api[one(api, "VHS_LoadVideo")]["inputs"].update({"video": job["video"], "force_rate": 25, "frame_load_cap": 0})
    return oa.prune(api)


def longcat_avatar(job: dict, oi: dict, attention: str = "sdpa") -> dict:
    api = template("kj_longcat_avatar", oi)
    _kijai_common(api, job, model="LongCat-Avatar-15_bf16.safetensors",
                  lora="LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors", attention=attention)
    api[one(api, "LoadImage")]["inputs"]["image"] = job["start"]
    return oa.prune(api)


# ------------------------------------------------------------------ stills: storyboard frames + reference sheets

def _node(api: dict, nid: str, cls: str, **inputs) -> list:
    api[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": nid}}
    return [nid, 0]


def _save_image(api: dict, job: dict, image: list) -> None:
    _node(api, "save", "SaveImage", images=image, filename_prefix=save_prefix(job))


def qwen_edit_2511(job: dict, oi: dict, refs: list[str]) -> dict:
    """
    Qwen-Image-Edit-2511, settings of the official template (image_qwen_image_edit_2511.json): AuraFlow shift 3.1,
    CFGNorm, index_timestep_zero reference method, euler/simple; quality = 40 steps cfg 4, fast = Lightning LoRA 4 steps cfg 1.
    Output is an empty latent of the exact frame size (the template edits image 1 in place instead).
    """
    api: dict = {}
    fast = job["variant"] == "fast"
    model = _node(api, "unet", "UNETLoader", unet_name="qwen_image_edit_2511_fp8mixed.safetensors", weight_dtype="default")
    clip = _node(api, "clip", "CLIPLoader", clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image", device="default")
    vae = _node(api, "vae", "VAELoader", vae_name="qwen_image_vae.safetensors")
    model = _node(api, "shift", "ModelSamplingAuraFlow", model=model, shift=3.1)
    model = _node(api, "cfgnorm", "CFGNorm", model=model, strength=1.0)
    if fast:
        model = _node(api, "lora", "LoraLoaderModelOnly", model=model, lora_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", strength_model=1.0)
    images = {f"image{i + 1}": _node(api, f"ref{i}", "LoadImage", image=r) for i, r in enumerate(refs[:3])}
    pos = _node(api, "pos", "TextEncodeQwenImageEditPlus", clip=clip, vae=vae, prompt=job["prompt"], **images)
    neg = _node(api, "neg", "TextEncodeQwenImageEditPlus", clip=clip, vae=vae, prompt=job.get("negative") or "", **images)
    pos = _node(api, "pos_ref", "FluxKontextMultiReferenceLatentMethod", conditioning=pos, reference_latents_method="index_timestep_zero")
    neg = _node(api, "neg_ref", "FluxKontextMultiReferenceLatentMethod", conditioning=neg, reference_latents_method="index_timestep_zero")
    latent = _node(api, "latent", "EmptySD3LatentImage", width=job["width"], height=job["height"], batch_size=1)
    samples = _node(api, "sampler", "KSampler", model=model, positive=pos, negative=neg, latent_image=latent, seed=int(job.get("seed") or 42),
                    steps=int(job.get("steps") or (4 if fast else 40)), cfg=1.0 if fast else 4.0, sampler_name="euler", scheduler="simple", denoise=1.0)
    _save_image(api, job, _node(api, "decode", "VAEDecode", samples=samples, vae=vae))
    return api


def flux2(job: dict, oi: dict, refs: list[str], *, klein: bool) -> dict:
    """
    FLUX.2 [dev] (image_flux2_fp8.json: FluxGuidance 4, BasicGuider, 20 steps; fast = Turbo LoRA 8 steps) and
    FLUX.2 [klein] 4B distilled (image_flux2_klein_image_edit_4b_distilled.json: CFGGuider cfg 1 with a zeroed negative, 4 steps).
    Every reference: scaled to ~1 MP, VAE-encoded, chained with ReferenceLatent (in our order: reference 1 first).
    """
    api: dict = {}
    if klein:
        model = _node(api, "unet", "UNETLoader", unet_name="flux-2-klein-4b-fp8.safetensors", weight_dtype="default")
        clip = _node(api, "clip", "CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2", device="default")
        for i, name in enumerate(job.get("loras") or []):  # exp-011 C: our character LoRAs (trained on klein base 4B, gpu/remote/lora.py)
            model = _node(api, f"char_lora{i}", "LoraLoaderModelOnly", model=model, lora_name=name, strength_model=float(job.get("lora_strength", 1.0)))
    else:
        model = _node(api, "unet", "UNETLoader", unet_name="flux2_dev_fp8mixed.safetensors", weight_dtype="default")
        clip = _node(api, "clip", "CLIPLoader", clip_name="mistral_3_small_flux2_fp8.safetensors", type="flux2", device="default")
        if job["variant"] == "fast":
            model = _node(api, "lora", "LoraLoaderModelOnly", model=model, lora_name="Flux2TurboComfyv2.safetensors", strength_model=1.0)
    vae = _node(api, "vae", "VAELoader", vae_name="flux2-vae.safetensors")
    pos = _node(api, "pos", "CLIPTextEncode", clip=clip, text=job["prompt"])
    neg = _node(api, "neg", "ConditioningZeroOut", conditioning=pos) if klein else None
    if not klein:
        pos = _node(api, "guidance", "FluxGuidance", conditioning=pos, guidance=4.0)
    for i, r in enumerate(refs):
        img = _node(api, f"ref{i}", "LoadImage", image=r)
        img = _node(api, f"ref{i}_scale", "ImageScaleToTotalPixels", image=img, upscale_method="area", megapixels=1.0, resolution_steps=1)
        lat = _node(api, f"ref{i}_latent", "VAEEncode", pixels=img, vae=vae)
        pos = _node(api, f"pos_ref{i}", "ReferenceLatent", conditioning=pos, latent=lat)
        if neg:
            neg = _node(api, f"neg_ref{i}", "ReferenceLatent", conditioning=neg, latent=lat)
    guider = (_node(api, "guider", "CFGGuider", model=model, positive=pos, negative=neg, cfg=1.0) if klein
              else _node(api, "guider", "BasicGuider", model=model, conditioning=pos))
    steps = int(job.get("steps") or (4 if klein else 8 if job["variant"] == "fast" else 20))
    samples = _node(api, "sampler", "SamplerCustomAdvanced",
                    noise=_node(api, "noise", "RandomNoise", noise_seed=int(job.get("seed") or 42)),
                    guider=guider,
                    sampler=_node(api, "ksampler", "KSamplerSelect", sampler_name="euler"),
                    sigmas=_node(api, "sigmas", "Flux2Scheduler", steps=steps, width=job["width"], height=job["height"]),
                    latent_image=_node(api, "latent", "EmptyFlux2LatentImage", width=job["width"], height=job["height"], batch_size=1))
    _save_image(api, job, _node(api, "decode", "VAEDecode", samples=samples, vae=vae))
    return api


# ------------------------------------------------------------------ exp-011: LongCat 1.5, SkyReels-V3 A2V, InfiniteTalk quality

def longcat_avatar15(job: dict, oi: dict, attention: str = "sdpa") -> dict:
    """
    LongCat-Video-Avatar 1.5 on the v1.0 Kijai example graph, changed per the 1.5 model card / LongCat/nodes.py:
    Whisper-large-v3 audio (LongCatAvatarWhisperEmbeds instead of wav2vec2 — exp-010 failed with 46080 vs 32000),
    25 fps (trained at 25), DMD distill LoRA with 8 steps, audio CFG 4 (card: 3–5), and the 1.5 overlap path
    (each extension re-encodes the previous segment's decoded frames: prev_images + vae).
    """
    api = template("kj_longcat_avatar", oi)
    _kijai_common(api, job, model="LongCat-Avatar-15_bf16.safetensors",
                  lora="LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors", attention=attention)
    w2v = one(api, "MultiTalkWav2VecEmbeds")
    i = api[w2v]["inputs"]
    for k in find(api, "DownloadAndLoadWav2VecModel") + find(api, "Wav2VecModelLoader"):
        del api[k]
    api["whisper"] = {"class_type": "WhisperModelLoader", "_meta": {"title": "whisper"},
                      "inputs": {"model": "whisper_large_v3_encoder_fp16.safetensors", "base_precision": "fp16", "load_device": "main_device"}}
    api[w2v] = {"class_type": "LongCatAvatarWhisperEmbeds", "_meta": {"title": "LongCat Avatar Whisper Embeds (v1.5)"},
                "inputs": {"whisper_model": ["whisper", 0], "audio_1": i["audio_1"], "normalize_loudness": True, "num_frames": 1000,
                           "fps": 25.0, "audio_scale": 1.0, "audio_cfg_scale": float(job.get("audio_cfg", 4.0)), "multi_audio_type": "para"}}
    for k in find(api, "LoadImage"):  # the example portrait (man.png) — exp-011 run 1 failed on it
        api[k]["inputs"]["image"] = job["start"]
    for k in find(api, "WanVideoSchedulerv2"):
        api[k]["inputs"]["steps"] = int(job.get("steps") or 8)
    for k in find(api, "VHS_VideoCombine"):
        api[k]["inputs"]["frame_rate"] = 25
    # v1.5 overlap: decoded frames of the segment the extension continues from
    for k in find(api, "WanVideoLongCatAvatarExtendEmbeds"):
        prev = api[k]["inputs"].get("prev_latents")
        # the decode of that segment: straight from the sampler, or after ReplaceVideoLatentFrames put the overlap back
        srcs = [prev] + [[r, 0] for r in find(api, "ReplaceVideoLatentFrames") if api[r]["inputs"].get("destination") == prev]
        dec = next((d for d in find(api, "WanVideoDecode") if api[d]["inputs"].get("samples") in srcs), None)
        if dec and api[k]["inputs"].get("overlap") != 1:  # not the first segment (prev_latents = the reference encode)
            api[k]["inputs"]["prev_images"] = [dec, 0]
            api[k]["inputs"]["vae"] = api[dec]["inputs"]["vae"]
    return oa.prune(api)


def skyreels_v3_a2v(job: dict, oi: dict, attention: str = "sdpa") -> dict:
    """SkyReels-V3 A2V (Kijai example): pass 1 renders a reference clip from the portrait, pass 2 drives it with our audio."""
    api = template("kj_skyreels_v3_a2v", oi)
    api[one(api, "WanVideoModelLoader")]["inputs"].update({"model": "Wan21-SkyReelsV3-A2V_fp8_scaled_mixed.safetensors", "attention_mode": attention,
                                                            "quantization": "disabled"})
    api[one(api, "WanVideoVAELoader")]["inputs"]["model_name"] = "Wan2_1_VAE_bf16.safetensors"
    api[one(api, "CLIPVisionLoader")]["inputs"]["clip_name"] = "clip_vision_h.safetensors"
    for k in find(api, "MelBandRoFormerSampler"):  # clean TTS already: no vocal separation
        rewire(api, [k, 0], api[k]["inputs"]["audio"])
        del api[k]
    api[one(api, "LoadAudio")]["inputs"]["audio"] = job["audio"]
    api[one(api, "LoadImage")]["inputs"]["image"] = job["start"]
    if job.get("audio2"):  # one speaker among several characters: route speech/silence with per-character masks (MultiTalk)
        emb = one(api, "MultiTalkWav2VecEmbeds")
        a2 = _node(api, "audio2", "LoadAudio", audio=job["audio2"])
        m1 = _node(api, "mask1", "LoadImageMask", image=job["mask1"], channel="red")
        m2 = _node(api, "mask2", "LoadImageMask", image=job["mask2"], channel="red")
        masks = _node(api, "masks", "MaskBatchMulti", inputcount=2, mask_1=m1, mask_2=m2)
        api[emb]["inputs"].update({"audio_2": a2, "ref_target_masks": masks})
    api[one(api, "INTConstant", title="Width")]["inputs"]["value"] = job["width"]
    api[one(api, "INTConstant", title="Height")]["inputs"]["value"] = job["height"]
    api[one(api, "INTConstant", title="Max frames")]["inputs"]["value"] = int(round(float(job["seconds"]) * 25)) + 10
    for k in find(api, "WanVideoTextEncodeCached"):
        api[k]["inputs"]["negative_prompt"] = job["negative"]
        if api[k]["inputs"]["positive_prompt"] != "a person is talking":  # the reference-clip prompt
            api[k]["inputs"]["positive_prompt"] = job["prompt"]
    for k in find(api, "WanVideoSamplerv2"):
        api[k]["inputs"]["seed"] = int(job.get("seed") or 42)
    for k in find(api, "VHS_VideoCombine"):
        api[k]["inputs"].update({"filename_prefix": save_prefix(job), "save_output": True})
    return oa.prune(api)


def infinitetalk_quality(job: dict, oi: dict, attention: str = "sdpa") -> dict:
    """InfiniteTalk i2v with the README quality settings: no lightx2v distill LoRA, 40 steps, text CFG 5, audio CFG 4."""
    api = infinitetalk_i2v({**job, "steps": None}, oi, attention)
    loader = one(api, "WanVideoModelLoader")
    api[loader]["inputs"].pop("lora", None)
    api = oa.prune(api)
    api[one(api, "WanVideoSampler")]["inputs"].update({"steps": int(job.get("steps") or 40), "cfg": 5.0, "shift": 7.0, "scheduler": "unipc"})
    api[one(api, "MultiTalkWav2VecEmbeds")]["inputs"]["audio_cfg_scale"] = float(job.get("audio_cfg", 4.0))
    return api


# ------------------------------------------------------------------ RIFE frame interpolation (ComfyUI-Frame-Interpolation)

def rife(job: dict, oi: dict) -> dict:
    """16 fps clip → ×3 → 48 fps; our 24 fps normalization then keeps every other frame (even cadence, no repeats)."""
    api: dict = {}
    frames = _node(api, "load", "VHS_LoadVideo", video=job["video"], force_rate=0, custom_width=0, custom_height=0, frame_load_cap=0, skip_first_frames=0, select_every_nth=1, format="None")
    frames = _node(api, "rife", "RIFE VFI", frames=frames, ckpt_name=job.get("ckpt", "rife47.pth"), clear_cache_after_n_frames=10,
                   multiplier=int(job.get("multiplier", 3)), fast_mode=False, ensemble=True, scale_factor=1.0, dtype="float16",
                   torch_compile=False, batch_size=4)
    _node(api, "save", "VHS_VideoCombine", images=frames, frame_rate=int(job.get("fps", 48)), loop_count=0, filename_prefix=save_prefix(job),
          format="video/h264-mp4", pingpong=False, save_output=True)
    return api


# ------------------------------------------------------------------ MiniMax H3 (native ComfyUI nodes, comfy_extras/nodes_minimax_h3.py)

H3_PROMPT = ("<Picture 1> is the first frame. {prompt} The character speaks the Persian line of the soundtrack; "
             "mouth and lips follow the given speech exactly. No music, no captions.")


def minimax_h3(job: dict, oi: dict) -> dict:
    """
    fl2va with our start frame, and our Persian TTS anchored as the soundtrack from frame 0 (MiniMaxH3AddGuide audio):
    the audio is fixed in the latent like a keyframe, so H3 animates the mouth to *our* speech instead of writing its own.
    variant "turbo": 8-step distill LoRA; "full": 20 steps. Sampler/scheduler as the official i2v template (res_multistep).
    """
    api: dict = {}
    model = _node(api, "unet", "UNETLoader", unet_name="minimax_h3_fl2va_pruned_int8_convrot.safetensors", weight_dtype="default")
    if job["variant"] == "turbo":
        model = _node(api, "lora", "LoraLoaderModelOnly", model=model, lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", strength_model=1.0)
    clip = _node(api, "clip", "CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default")
    vae = _node(api, "vae", "VAELoader", vae_name="minimax_h3_video_vae_int8_convrot.safetensors")
    avae = _node(api, "avae", "VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    image = _node(api, "image", "LoadImage", image=job["start"])
    audio = _node(api, "audio", "LoadAudio", audio=job["audio"])
    i2v = _node(api, "i2v", "MiniMaxH3ImageToVideo", clip=clip, vae=vae, prompt=H3_PROMPT.format(prompt=job["prompt"]), width=job["width"], height=job["height"],
                length=int(job["frames"]), first_frame=image)
    pos = _node(api, "guide", "MiniMaxH3AddGuide", positive=[i2v[0], 0], latent=[i2v[0], 1], audio_vae=avae, audio=audio, frame_idx=0)
    guider = _node(api, "guider", "BasicGuider", model=model, conditioning=pos)
    steps = int(job.get("steps") or (8 if job["variant"] == "turbo" else 20))
    sigmas = _node(api, "sched", "BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1.0)
    sampler = _node(api, "sampler", "KSamplerSelect", sampler_name="res_multistep")
    noise = _node(api, "noise", "RandomNoise", noise_seed=int(job.get("seed") or 42))
    out = _node(api, "sample", "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=[i2v[0], 1])
    frames = _node(api, "decode", "VAEDecode", samples=out, vae=vae)
    sound = _node(api, "decode_audio", "VAEDecodeAudio", samples=out, vae=avae)
    # raw clip keeps H3's own decoded soundtrack (shows whether the anchored speech survived); the final mux uses our TTS anyway
    _node(api, "save", "VHS_VideoCombine", images=frames, audio=sound, frame_rate=24, loop_count=0, filename_prefix=save_prefix(job),
          format="video/h264-mp4", pingpong=False, save_output=True)
    return api


# ------------------------------------------------------------------ exp-012: music + scene ambience (native ComfyUI audio models)
# job: id, model, variant ("music" | "sfx"), project, scene, seconds, seed, prompt (English), bpm (music)

def _save_audio(api: dict, job: dict, audio: list) -> None:
    _node(api, "save", "SaveAudio", audio=audio, filename_prefix=save_prefix(job))


def ace_step15(job: dict, oi: dict) -> dict:
    """ACE-Step 1.5 XL turbo (audio_ace_step1_5_xl_turbo.json): 8 steps, cfg 1, AuraFlow shift 3; instrumental (empty lyrics)."""
    api: dict = {}
    model = _node(api, "unet", "UNETLoader", unet_name="acestep_v1.5_xl_turbo_bf16.safetensors", weight_dtype="default")
    model = _node(api, "shift", "ModelSamplingAuraFlow", model=model, shift=3.0)
    clip = _node(api, "clip", "DualCLIPLoader", clip_name1="qwen_0.6b_ace15.safetensors", clip_name2="qwen_4b_ace15.safetensors", type="ace", device="default")
    vae = _node(api, "vae", "VAELoader", vae_name="ace_1.5_vae.safetensors")
    seconds = float(job["seconds"])
    pos = _node(api, "pos", "TextEncodeAceStepAudio1.5", clip=clip, tags=job["prompt"], lyrics="[Instrumental]", seed=int(job.get("seed") or 42),
                bpm=int(job.get("bpm") or 100), duration=seconds, timesignature="4", language="en", keyscale=job.get("keyscale", "C major"),
                generate_audio_codes=True, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0)
    neg = _node(api, "neg", "ConditioningZeroOut", conditioning=pos)
    lat = _node(api, "latent", "EmptyAceStep1.5LatentAudio", seconds=seconds, batch_size=1)
    out = _node(api, "sampler", "KSampler", model=model, seed=int(job.get("seed") or 42), steps=8, cfg=1.0, sampler_name="euler", scheduler="simple",
                positive=pos, negative=neg, latent_image=lat, denoise=1.0)
    _save_audio(api, job, _node(api, "decode", "VAEDecodeAudio", samples=out, vae=vae))
    return api


def minimax_music3(job: dict, oi: dict) -> dict:
    """MiniMax Music 3 (audio_minimax_music_3.json): 30 steps, cfg 1.7; instrumental."""
    api: dict = {}
    model = _node(api, "unet", "UNETLoader", unet_name="minimax_music3_dit_fp16.safetensors", weight_dtype="default")
    clip = _node(api, "clip", "CLIPLoader", clip_name="minimax_music3_text_encoder_pruned_int8_convrot.safetensors", type="minimax", device="default")
    vae = _node(api, "vae", "VAELoader", vae_name="minimax_music3_dav.safetensors")
    seconds = float(job["seconds"])
    pos = _node(api, "pos", "MiniMaxMusic3TextEncode", clip=clip, caption=job["prompt"], lyrics="[Instrumental]", seed=int(job.get("seed") or 42),
                max_duration=seconds, cfg_scale=1.7, top_k=50)
    neg = _node(api, "neg", "ConditioningZeroOut", conditioning=pos)
    lat = _node(api, "latent", "EmptyMiniMaxMusic3LatentAudio", seconds=seconds, batch_size=1)
    out = _node(api, "sampler", "KSampler", model=model, seed=int(job.get("seed") or 42), steps=30, cfg=1.7, sampler_name="euler", scheduler="simple",
                positive=pos, negative=neg, latent_image=lat, denoise=1.0)
    _save_audio(api, job, _node(api, "decode", "VAEDecodeAudio", samples=out, vae=vae))
    return api


def stable_audio3(job: dict, oi: dict) -> dict:
    """Stable Audio 3 medium (audio_stable_audio_3_medium.json without the Qwen prompt rewriter): 8 steps lcm; music or SFX/ambience."""
    api: dict = {}
    ckpt = _node(api, "ckpt", "CheckpointLoaderSimple", ckpt_name="stable_audio_3_medium.safetensors")
    clip = _node(api, "clip", "CLIPLoader", clip_name="t5gemma_b_b_ul2.safetensors", type="stable_audio", device="default")
    pos = _node(api, "pos", "CLIPTextEncode", clip=clip, text=job["prompt"])
    neg = _node(api, "neg", "CLIPTextEncode", clip=clip, text=job.get("negative") or "")
    lat = _node(api, "latent", "EmptyLatentAudio", seconds=float(job["seconds"]), batch_size=1)
    out = _node(api, "sampler", "KSampler", model=[ckpt[0], 0], seed=int(job.get("seed") or 42), steps=8, cfg=1.0, sampler_name="lcm", scheduler="simple",
                positive=pos, negative=neg, latent_image=lat, denoise=1.0)
    _save_audio(api, job, _node(api, "decode", "VAEDecodeAudio", samples=out, vae=[ckpt[0], 2]))
    return api


AUDIO_BUILDERS = {"ace-step-15": ace_step15, "minimax-music3": minimax_music3, "stable-audio3": stable_audio3}


IMAGE_BUILDERS = {
    "qwen-image-edit-2511": qwen_edit_2511,
    "flux2-dev": lambda j, oi, refs: flux2(j, oi, refs, klein=False),
    "flux2-klein-4b": lambda j, oi, refs: flux2(j, oi, refs, klein=True),
}


BUILDERS = {
    ("wan22-14b", "flf"): lambda j, oi: wan22(j, oi, start=j["start"], end=j["end"], frames=j["frames"]),
    ("wan22-14b", "i2v"): lambda j, oi: wan22(j, oi, start=j["start"], end=None, frames=j["frames"]),
    ("ltx23", "flf"): ltx23_flf,
    ("ltx23", "ia2v"): ltx23_ia2v,
    ("infinitetalk", "i2v"): infinitetalk_i2v,
    ("infinitetalk", "v2v"): infinitetalk_v2v,
    ("longcat-avatar", "i2v"): longcat_avatar,
    ("rife", "x3"): rife,
}
# Kijai-wrapper builders take the attention mode (run.py render_comfy passes it for these models)
BUILDERS[("longcat-avatar-15", "i2v")] = longcat_avatar15
BUILDERS[("skyreels-v3-a2v", "a2v")] = skyreels_v3_a2v
BUILDERS[("infinitetalk-q", "i2v")] = infinitetalk_quality
BUILDERS[("minimax-h3", "turbo")] = minimax_h3
BUILDERS[("minimax-h3", "full")] = minimax_h3
