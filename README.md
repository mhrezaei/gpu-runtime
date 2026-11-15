# gpu-runtime

Container images for GPU render servers: ComfyUI with a pinned set of custom nodes, ffmpeg, and a PyTorch CUDA base.
Model weights are not included; they are downloaded at boot.

## Images

| Tag | Contents |
|---|---|
| `ghcr.io/mhrezaei/gpu-runtime:<version>` | Video runtime (default) |
| `ghcr.io/mhrezaei/gpu-runtime:<version>-runtime-lora` | Video runtime plus LoRA training tools |

```bash
docker pull ghcr.io/mhrezaei/gpu-runtime:<version>
```

Base: `pytorch/pytorch` (CUDA 13, cuDNN 9). Pull by digest for reproducible servers.

## Build

GitHub Actions (`.github/workflows/build.yml`) builds both targets of `gpu/image/Dockerfile` when `VERSION` or `gpu/` changes, runs a secret scan, a no-weights check and the smoke checks, then pushes. A tag that already exists is never overwritten.
