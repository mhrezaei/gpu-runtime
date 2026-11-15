# gpu-runtime

Container images for GPU render servers: ComfyUI with a pinned set of custom nodes, ffmpeg, and a PyTorch CUDA base.
Model weights are not included; they are downloaded at boot.

## Images

| Tag | Contents |
|---|---|
| `ghcr.io/mhrezaei/gpu-runtime:<version>` | Video runtime (default) |
| `ghcr.io/mhrezaei/gpu-runtime:<version>-lora` | Video runtime plus LoRA training tools |

```bash
docker pull ghcr.io/mhrezaei/gpu-runtime:<version>
```

Base: `pytorch/pytorch` (CUDA 13, cuDNN 9). Pull by digest for reproducible servers.
