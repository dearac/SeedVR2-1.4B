# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Inference entry point for lvladikov/SeedVR2-1.4B.

This reuses the upstream SeedVR2 pipeline while selecting the distilled six-layer
NaDiT configuration and safetensors checkpoints.
"""

import argparse
import os

from huggingface_hub import snapshot_download

from projects.inference_seedvr2_3b import generation_loop
from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.distributed import init_torch
from common.distributed.advanced import init_sequence_parallel
from omegaconf import OmegaConf
import datetime

MODEL_REPO_ID = "lvladikov/SeedVR2-1.4B"
MODEL_DIR = "./ckpts/SeedVR2-1.4B"
DIT_FILENAME = "seedvr2_distill_6L_1.4B_sharp_fp16.safetensors"


def ensure_checkpoints(model_dir: str, download: bool) -> None:
    required = [
        os.path.join(model_dir, DIT_FILENAME),
        os.path.join(model_dir, "ema_vae_fp16.safetensors"),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if not missing:
        return
    if not download:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "SeedVR2-1.4B checkpoints are missing:\n"
            f"{formatted}\nRun again with --download-model or place the files manually."
        )
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=model_dir,
        allow_patterns=["*.safetensors", "*.md", "*.txt", "*.json"],
    )


def configure_runner(sp_size: int, model_dir: str) -> VideoDiffusionInfer:
    config = load_config("./configs_1_4b/main.yaml")
    OmegaConf.set_readonly(config, False)
    config.vae.checkpoint = os.path.join(model_dir, "ema_vae_fp16.safetensors")

    runner = VideoDiffusionInfer(config)
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    if sp_size > 1:
        init_sequence_parallel(sp_size)

    runner.configure_dit_model(
        device="cuda",
        checkpoint=os.path.join(model_dir, DIT_FILENAME),
    )
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SeedVR2-1.4B restoration")
    parser.add_argument("--video_path", type=str, default="./test_videos")
    parser.add_argument("--output_dir", type=str, default="./results_1_4b")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--res_h", type=int, default=2048)
    parser.add_argument("--res_w", type=int, default=2048)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--out_fps", type=float, default=None)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()

    ensure_checkpoints(args.model_dir, args.download_model)
    runner = configure_runner(args.sp_size, args.model_dir)
    generation_loop(
        runner=runner,
        video_path=args.video_path,
        output_dir=args.output_dir,
        seed=args.seed,
        res_h=args.res_h,
        res_w=args.res_w,
        sp_size=args.sp_size,
        out_fps=args.out_fps,
        cfg_scale=1.0,
        cfg_rescale=0.0,
        sample_steps=1,
    )


if __name__ == "__main__":
    main()
