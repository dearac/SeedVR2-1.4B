# SeedVR2-1.4B support

This fork includes a single-GPU still-image inference path for [`lvladikov/SeedVR2-1.4B`](https://huggingface.co/lvladikov/SeedVR2-1.4B).

The checkpoint is a six-layer distillation of SeedVR2-7B. It keeps the 7B model width, head count, conditioning layout, and VAE, but reduces the NaDiT transformer from 36 blocks to 6. The model is intended primarily for 2x-4x still-image restoration and upscaling.

## Install

Use the normal SeedVR environment, then install the updated requirements:

```bash
pip install -r requirements.txt
```

The official repository still requires a CUDA-capable PyTorch environment. Apex and FlashAttention requirements depend on the rest of your SeedVR installation; the 1.4B checkpoint support itself only adds `safetensors` and `huggingface_hub`.

## Download the checkpoints

```bash
python scripts/download_seedvr2_1_4b.py
```

This downloads:

- `ckpts/SeedVR2-1.4B/seedvr2_distill_6L_1.4B_sharp_fp16.safetensors`
- `ckpts/SeedVR2-1.4B/ema_vae_fp16.safetensors`

## Text embeddings

The upstream SeedVR2 PyTorch inference implementation still passes precomputed positive and negative text-conditioning tensors into NaDiT, even though restoration is prompt-free and does not require a live text encoder.

Place the same `pos_emb.pt` and `neg_emb.pt` files used by the upstream SeedVR2-7B inference script in the repository root, or pass their locations using `--pos-emb` and `--neg-emb`.

## Run one image

Launch through `torchrun`, because the upstream initialization creates a one-process NCCL group even for a single GPU:

```bash
torchrun --standalone --nproc-per-node=1 projects/inference_seedvr2_1_4b.py \
  --input ./test_images/input.png \
  --output-dir ./results_1_4b \
  --height 2048 \
  --width 2048
```

## Run a folder

```bash
torchrun --standalone --nproc-per-node=1 projects/inference_seedvr2_1_4b.py \
  --input ./test_images \
  --output-dir ./results_1_4b \
  --height 2048 \
  --width 2048
```

Supported inputs are PNG, JPEG, BMP, TIFF, and WebP.

## Useful options

```text
--dtype float16|bfloat16
--seed 666
--color-fix
--transformer PATH
--vae PATH
--pos-emb PATH
--neg-emb PATH
```

`float16` is the default because the published checkpoints are FP16. `bfloat16` can be tested on supported NVIDIA GPUs, but it does not reduce checkpoint size.

## Memory notes

The 1.4B checkpoint substantially reduces transformer weight memory, but output-resolution activations and VAE decoding still scale with image area. Start with 2x or 4x output dimensions. Very large outputs may still exhaust consumer GPUs because spatial tiling is not yet implemented in this inference path.

The script moves the transformer and VAE between CPU and GPU for the major encode, diffusion, and decode stages. This reduces simultaneous VRAM residency at the cost of transfer time.

## Current scope

This implementation intentionally supports still images first. The published model is described and evaluated primarily as an image upscaler. Video chunking, temporal overlap, and tiled spatial inference should be added separately rather than silently inheriting the high-memory 7B video path.
