"""
Model registry for the self-hosted tests (docs/07-self-hosted-test-plan.md).

Every file URL and size was checked against the Hugging Face API on 2026-10-20 (no repo is gated).
`dest` is relative to ComfyUI/models for engine "comfy", or to the repo checkout for engine "repo".
"""
from __future__ import annotations

HF = "https://huggingface.co"


def f(repo: str, path: str, dest: str, gb: float) -> dict:
    return {"url": f"{HF}/{repo}/resolve/main/{path}", "repo": repo, "path": path, "dest": dest, "gb": gb}


WAN22_RP = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN21_RP = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

MODELS: dict[str, dict] = {
    # ---------------------------------------------------------------- silent B-roll, first/last frame
    "wan22-14b": {
        "label": "Wan 2.2 I2V-A14B (fp8, ComfyUI native) + lightx2v 4-step",
        "license": "Apache-2.0",
        "engine": "comfy",
        "files": [
            f(WAN22_RP, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models/", 14.29),
            f(WAN22_RP, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models/", 14.29),
            f(WAN22_RP, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "loras/", 1.23),
            f(WAN22_RP, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "loras/", 1.23),
            f(WAN22_RP, "split_files/vae/wan_2.1_vae.safetensors", "vae/", 0.25),
            f(WAN21_RP, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders/", 6.74),
        ],
    },
    "ltx23": {
        "label": "LTX-2.3 22B distilled (fp8, ComfyUI native) — FLF and image+audio→video",
        "warmup": False,  # the 1 s warm-up hits a shape mismatch (2026-10-20 run) that skipped every LTX job
        "license": "LTX-2 Community (free under $10M revenue)",
        "engine": "comfy",
        "files": [
            f("Lightricks/LTX-2.3-fp8", "ltx-2.3-22b-distilled-fp8.safetensors", "checkpoints/", 29.53),
            f("Comfy-Org/ltx-2", "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", "text_encoders/", 9.45),
            f("Comfy-Org/ltx-2", "split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors", "loras/", 0.63),
            f("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "latent_upscale_models/", 1.00),
        ],
    },
    "wan22-5b": {
        "label": "Wan 2.2 TI2V-5B (optional baseline)",
        "license": "Apache-2.0",
        "engine": "comfy",
        "optional": True,
        "files": [
            f(WAN22_RP, "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models/", 10.0),
            f(WAN22_RP, "split_files/vae/wan2.2_vae.safetensors", "vae/", 1.41),
            f(WAN21_RP, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders/", 6.74),
        ],
    },
    # ---------------------------------------------------------------- talking head (portrait + our audio)
    "infinitetalk": {
        "label": "InfiniteTalk single (Wan 2.1 I2V 14B 480p fp8, Kijai WanVideoWrapper)",
        "license": "Apache-2.0",
        "engine": "comfy",
        "files": [
            f("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models/", 16.64),
            f("Kijai/WanVideo_comfy_fp8_scaled", "InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models/", 2.71),
            f("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras/", 0.74),
            f("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae/", 0.25),
            f("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "text_encoders/", 11.36),
            f(WAN21_RP, "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision/", 1.26),
            # wav2vec2 (TencentGameMate/chinese-wav2vec2-base, ~0.4GB) is auto-downloaded by DownloadAndLoadWav2VecModel
        ],
    },
    "longcat-avatar": {
        "label": "LongCat-Video-Avatar 1.5 (Kijai WanVideoWrapper) — experimental",
        "license": "MIT",
        "engine": "comfy",
        "experimental": True,
        "files": [
            f("Kijai/WanVideo_comfy", "LongCat/LongCat-Avatar-15_bf16.safetensors", "diffusion_models/", 31.71),
            f("Kijai/WanVideo_comfy", "LongCat/LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors", "loras/", 1.26),
            f("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae/", 0.25),
            f("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "text_encoders/", 11.36),
        ],
    },
    # ---------------------------------------------------------------- post lip-sync on an existing clip
    "latentsync": {
        "label": "LatentSync 1.6 (official repo)",
        "license": "OpenRAIL++ (weights) / Apache-2.0 (code)",
        "engine": "repo",
        "git": "https://github.com/bytedance/LatentSync.git",
        "files": [
            f("ByteDance/LatentSync-1.6", "latentsync_unet.pt", "checkpoints/", 5.07),
            f("ByteDance/LatentSync-1.6", "whisper/tiny.pt", "checkpoints/whisper/", 0.08),
        ],
    },
    "musetalk": {
        "label": "MuseTalk 1.5 (official repo) — experimental (mmcv/mmpose install)",
        "license": "MIT",
        "engine": "repo",
        "experimental": True,
        "git": "https://github.com/TMElyralab/MuseTalk.git",
        # MuseTalk's own download_weights.sh fetches unet + sd-vae + whisper + dwpose + syncnet + face-parse
        "files": [],
    },
    # ---------------------------------------------------------------- stills (storyboard frames + reference sheets)
    # "kind": "image" → run.py renders per-project chains (a start frame uses the previous frame of the same model).
    # Sizes checked against the HF API on 2026-10-20 (none gated). Settings follow the official ComfyUI templates
    # (workflows/upstream/image_*.json); graphs are built in workflows.py for N references and an exact output size.
    "qwen-image-edit-2511": {
        "label": "Qwen-Image-Edit-2511 (20B fp8mixed, ComfyUI native) — up to 3 references",
        "license": "Apache-2.0",
        "engine": "comfy",
        "kind": "image",
        "max_refs": 3,
        "files": [
            f("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", "diffusion_models/", 20.53),
            f("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "text_encoders/", 9.38),
            f("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae/", 0.25),
            f("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "loras/", 0.85),
        ],
    },
    "flux2-dev": {
        "label": "FLUX.2 [dev] (32B fp8mixed, ComfyUI native) — multi-reference; weights > 32 GB VRAM (partial offload)",
        "license": "FLUX [dev] Non-Commercial (a paid product needs a BFL license)",
        "engine": "comfy",
        "kind": "image",
        "max_refs": 4,
        "files": [
            f("Comfy-Org/flux2-dev", "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors", "diffusion_models/", 35.46),
            f("Comfy-Org/flux2-dev", "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors", "text_encoders/", 18.03),
            f("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", "vae/", 0.34),
            f("Comfy-Org/flux2-dev", "split_files/loras/Flux2TurboComfyv2.safetensors", "loras/", 2.76),
        ],
    },
    "flux2-klein-4b": {
        "label": "FLUX.2 [klein] 4B distilled (fp8, ComfyUI native) — 4 steps",
        "license": "Apache-2.0",
        "engine": "comfy",
        "kind": "image",
        "max_refs": 4,
        "files": [
            f("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors", "diffusion_models/", 4.07),
            f("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders/", 8.04),
            f("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", "vae/", 0.34),
        ],
    },
}

# ---------------------------------------------------------------- exp-011: next-generation lip-sync (docs/research/11)
KJ = "Kijai/WanVideo_comfy"
MODELS["longcat-avatar-15"] = {
    "label": "LongCat-Video-Avatar 1.5 (Whisper-large-v3 audio, DMD 8-step LoRA, 25 fps; Kijai WanVideoWrapper)",
    "license": "MIT",
    "engine": "comfy",
    "warmup": False,
    "files": [
        f(KJ, "LongCat/LongCat-Avatar-15_bf16.safetensors", "diffusion_models/", 31.71),
        f(KJ, "LongCat/LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors", "loras/", 1.26),
        f(KJ, "HuMo/whisper_large_v3_encoder_fp16.safetensors", "audio_encoders/", 1.69),
        f(KJ, "Wan2_1_VAE_bf16.safetensors", "vae/", 0.25),
        f(KJ, "umt5-xxl-enc-bf16.safetensors", "text_encoders/", 11.36),
    ],
}
MODELS["skyreels-v3-a2v"] = {
    "label": "SkyReels-V3 A2V 19B talking avatar (fp8 scaled, Kijai WanVideoWrapper; 2 passes: reference video → audio-driven)",
    "license": "Skywork Community License (commercial use allowed)",
    "engine": "comfy",
    "warmup": False,
    "files": [
        f("Kijai/WanVideo_comfy_fp8_scaled", "SkyReelsV3/Wan21-SkyReelsV3-A2V_fp8_scaled_mixed.safetensors", "diffusion_models/", 19.13),
        f(KJ, "Wan2_1_VAE_bf16.safetensors", "vae/", 0.25),
        f(KJ, "umt5-xxl-enc-bf16.safetensors", "text_encoders/", 11.36),
        f(WAN21_RP, "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision/", 1.26),
    ],
}
# InfiniteTalk with the README quality settings (no distill LoRA, 40 steps, audio CFG) — a separate key so the
# 31 fast InfiniteTalk jobs of plan.json are not re-run on a new server
MODELS["infinitetalk-q"] = {**MODELS["infinitetalk"], "label": "InfiniteTalk single, quality settings (40 steps, no distill LoRA, audio CFG 4)", "warmup": False}

# MiniMax H3 fl2va (native ComfyUI): start frame + our TTS anchored as soundtrack (workflows.minimax_h3).
# License excludes US/EU/UK/KR territory — owner's decision 2026-10-21: test only, server outside those (UA).
H3 = "Comfy-Org/MiniMax-H3"
MODELS["minimax-h3"] = {
    "label": "MiniMax H3 fl2va 20B pruned int8 + our audio as guide (native ComfyUI)",
    "license": "MiniMax H3 license (territory excludes US/EU/UK/KR; must display \"MiniMax H3\")",
    "engine": "comfy",
    "needs_node": "MiniMaxH3AddGuide",
    "warmup": False,
    "files": [
        f(H3, "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models/", 20.97),
        f(H3, "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "loras/", 1.96),
        f(H3, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders/", 15.69),
        f(H3, "vae/minimax_h3_video_vae_int8_convrot.safetensors", "vae/", 2.81),
        f(H3, "vae/minimax_h3_audio_vae_fp32.safetensors", "vae/", 0.61),
    ],
}

# exp-011 C: character LoRA pilot (gpu/remote/lora.py); ai-toolkit fetches the klein base 4B + Qwen3-4B from HF itself
MODELS["lora-train"] = {
    "label": "Character LoRA training, ai-toolkit on FLUX.2 [klein] base 4B (rank 16)",
    "license": "Apache-2.0 (base model); training data = our gpt-image stills (owner's decision)",
    "engine": "lora",
    "warmup": False,
    "files": [],
}

# exp-012: background music + scene ambience (native ComfyUI audio models; workflows.AUDIO_BUILDERS)
ACE = "Comfy-Org/ace_step_1.5_ComfyUI_files"
MODELS["ace-step-15"] = {
    "label": "ACE-Step 1.5 XL turbo — instrumental music (8 steps)", "license": "MIT (ACE-Step) / Apache-2.0 (Comfy repack)",
    "engine": "comfy", "kind": "audio", "warmup": False,
    "files": [
        f(ACE, "split_files/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors", "diffusion_models/", 9.29),
        f(ACE, "split_files/vae/ace_1.5_vae.safetensors", "vae/", 0.32),
        f(ACE, "split_files/text_encoders/qwen_0.6b_ace15.safetensors", "text_encoders/", 1.11),
        f(ACE, "split_files/text_encoders/qwen_4b_ace15.safetensors", "text_encoders/", 7.8),
    ],
}
MM3 = "Comfy-Org/MiniMax-Music-3"
MODELS["minimax-music3"] = {
    "label": "MiniMax Music 3 — instrumental music (30 steps)", "license": "Apache-2.0 per the Comfy repack (upstream not public — unverified)",
    "engine": "comfy", "kind": "audio", "warmup": False,
    "files": [
        f(MM3, "diffusion_models/minimax_music3_dit_fp16.safetensors", "diffusion_models/", 4.58),
        f(MM3, "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors", "text_encoders/", 8.57),
        f(MM3, "vae/minimax_music3_dav.safetensors", "vae/", 0.21),
    ],
}
SA3 = "Comfy-Org/stable-audio-3"
MODELS["stable-audio3"] = {
    "label": "Stable Audio 3 medium — music and scene ambience/SFX (8 steps)", "license": "Stability AI Community (free under $1M revenue)",
    "engine": "comfy", "kind": "audio", "warmup": False,
    "files": [
        f(SA3, "checkpoints/stable_audio_3_medium.safetensors", "checkpoints/", 8.59),
        f(SA3, "text_encoders/t5gemma_b_b_ul2.safetensors", "text_encoders/", 1.11),
    ],
}

# ---------------------------------------------------------------- post: smoother motion for 16 fps outputs
MODELS["rife"] = {
    "label": "RIFE 4.7 frame interpolation ×3 (ComfyUI-Frame-Interpolation) — 16→48 fps",
    "license": "MIT",
    "engine": "comfy",
    "needs_node": "RIFE VFI",
    "warmup": False,  # a warm-up on a missing source clip would stop the worker; RIFE loads in a second anyway
    "files": [],  # rife47.pth (~20 MB) is fetched by the node pack on first use
}

IMAGE_ORDER = ["flux2-klein-4b", "qwen-image-edit-2511", "flux2-dev"]

# Order on the server: renders first, lip-sync last (it consumes Wan/LTX outputs).
DEFAULT_ORDER = ["wan22-14b", "ltx23", "infinitetalk", "longcat-avatar", "latentsync", "musetalk"]


def total_gb(keys: list[str]) -> float:
    seen, gb = set(), 0.0
    for k in keys:
        for x in MODELS[k]["files"]:
            if x["url"] not in seen:
                seen.add(x["url"])
                gb += x["gb"]
    return round(gb, 1)


if __name__ == "__main__":
    for k, m in MODELS.items():
        print(f"{k:<16} {sum(x['gb'] for x in m['files']):6.1f} GB  {m['label']}")
    print("total (default order, shared files once):", total_gb(DEFAULT_ORDER), "GB")
