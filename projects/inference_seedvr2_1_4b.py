# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Single-GPU still-image inference for lvladikov/SeedVR2-1.4B.

The distilled model keeps the SeedVR2-7B width and attention layout, but uses
six NaDiT blocks. Its transformer and VAE are distributed as safetensors.
"""

import argparse
import datetime
import gc
import os
from pathlib import Path
from typing import Dict, Iterable

import mediapy
import torch
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torchvision.io import read_image
from torchvision.transforms import Compose, Lambda, Normalize

from common.config import load_config
from common.distributed import get_device, init_torch
from common.seed import set_seed
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
from projects.video_diffusion_sr.infer import VideoDiffusionInfer

try:
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
except ImportError:
    wavelet_reconstruction = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _load_safetensors_with_torch_load_compat(path, *args, **kwargs):
    """Load safetensors while preserving the interface expected by SeedVR2."""
    path = os.fspath(path)
    if path.lower().endswith(".safetensors"):
        return load_file(path, device="cpu")
    return _ORIGINAL_TORCH_LOAD(path, *args, **kwargs)


_ORIGINAL_TORCH_LOAD = torch.load


def configure_runner(
    transformer_path: str,
    vae_path: str,
    dtype: torch.dtype,
) -> VideoDiffusionInfer:
    config = load_config("./configs_1_4b/main.yaml")
    OmegaConf.set_readonly(config, False)
    config.vae.checkpoint = vae_path
    config.vae.dtype = str(dtype).removeprefix("torch.")

    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    runner = VideoDiffusionInfer(config)

    # The upstream loader uses torch.load for both .pth files. Temporarily
    # route safetensors paths through safetensors.torch.load_file instead.
    torch.load = _load_safetensors_with_torch_load_compat
    try:
        runner.configure_dit_model(device="cpu", checkpoint=transformer_path)
        runner.configure_vae_model()
    finally:
        torch.load = _ORIGINAL_TORCH_LOAD

    runner.dit.to(device=get_device(), dtype=dtype).eval()
    runner.vae.to(device="cpu", dtype=dtype).eval()

    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

    return runner


def iter_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for path in sorted(input_path.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_text_embeddings(pos_path: str, neg_path: str) -> Dict[str, list]:
    if not os.path.isfile(pos_path) or not os.path.isfile(neg_path):
        raise FileNotFoundError(
            "SeedVR2's upstream inference path requires precomputed text "
            "embeddings. Supply --pos-emb and --neg-emb, or place "
            "pos_emb.pt and neg_emb.pt in the repository root."
        )

    return {
        "texts_pos": [_ORIGINAL_TORCH_LOAD(pos_path, map_location="cpu")],
        "texts_neg": [_ORIGINAL_TORCH_LOAD(neg_path, map_location="cpu")],
    }


def restore_image(
    runner: VideoDiffusionInfer,
    image_path: Path,
    output_path: Path,
    text_embeddings: Dict[str, list],
    height: int,
    width: int,
    seed: int,
    dtype: torch.dtype,
    use_color_fix: bool,
) -> None:
    set_seed(seed, same_across_ranks=True)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    transform = Compose(
        [
            NaResize(
                resolution=(height * width) ** 0.5,
                mode="area",
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ]
    )

    source = read_image(str(image_path)).unsqueeze(0).float().div_(255.0)
    transformed = transform(source.to(get_device()))

    runner.dit.to("cpu")
    runner.vae.to(get_device())
    cond_latent = runner.vae_encode([transformed])[0]
    runner.vae.to("cpu")
    runner.dit.to(get_device())

    noise = torch.randn_like(cond_latent)
    aug_noise = torch.randn_like(cond_latent)
    timestep = torch.tensor([0.0], device=get_device())
    shape = torch.tensor(cond_latent.shape[1:], device=get_device())[None]
    timestep = runner.timestep_transform(timestep, shape)
    latent_blur = runner.schedule.forward(cond_latent, aug_noise, timestep)
    condition = runner.get_condition(noise, task="sr", latent_blur=latent_blur)

    embeds = {
        key: [tensor.to(get_device()) for tensor in values]
        for key, values in text_embeddings.items()
    }

    autocast_enabled = get_device().type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        samples = runner.inference(
            noises=[noise.to(get_device())],
            conditions=[condition.to(get_device())],
            dit_offload=True,
            **embeds,
        )

    sample = samples[0]
    sample = rearrange(sample[:, None], "c t h w -> t c h w") if sample.ndim == 3 else rearrange(sample, "c t h w -> t c h w")
    original = rearrange(transformed[:, None], "c t h w -> t c h w") if transformed.ndim == 3 else rearrange(transformed, "c t h w -> t c h w")

    if use_color_fix and wavelet_reconstruction is not None:
        sample = wavelet_reconstruction(sample.cpu(), original[: sample.size(0)].cpu())
    else:
        sample = sample.cpu()

    image = rearrange(sample, "t c h w -> t h w c")
    image = image.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
    image = image.to(torch.uint8).numpy().squeeze(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_image(str(output_path), image)

    del source, transformed, cond_latent, noise, aug_noise, condition, samples, sample
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--output-dir", default="./results_1_4b")
    parser.add_argument(
        "--transformer",
        default="./ckpts/SeedVR2-1.4B/seedvr2_distill_6L_1.4B_sharp_fp16.safetensors",
    )
    parser.add_argument(
        "--vae",
        default="./ckpts/SeedVR2-1.4B/ema_vae_fp16.safetensors",
    )
    parser.add_argument("--pos-emb", default="./pos_emb.pt")
    parser.add_argument("--neg-emb", default="./neg_emb.pt")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--color-fix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    for checkpoint in (args.transformer, args.vae):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}. Run "
                "scripts/download_seedvr2_1_4b.py first or pass an explicit path."
            )

    images = list(iter_images(Path(args.input)))
    if not images:
        raise ValueError(f"No supported images found in {args.input}")

    text_embeddings = load_text_embeddings(args.pos_emb, args.neg_emb)
    runner = configure_runner(args.transformer, args.vae, dtype)
    output_dir = Path(args.output_dir)

    for index, image_path in enumerate(images):
        output_path = output_dir / image_path.name
        print(f"[{index + 1}/{len(images)}] Restoring {image_path} -> {output_path}")
        restore_image(
            runner=runner,
            image_path=image_path,
            output_path=output_path,
            text_embeddings=text_embeddings,
            height=args.height,
            width=args.width,
            seed=args.seed + index,
            dtype=dtype,
            use_color_fix=args.color_fix,
        )


if __name__ == "__main__":
    main()
