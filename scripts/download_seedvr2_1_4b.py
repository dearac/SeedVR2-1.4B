"""Download the lvladikov/SeedVR2-1.4B runtime checkpoints."""

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "lvladikov/SeedVR2-1.4B"
FILES = [
    "seedvr2_distill_6L_1.4B_sharp_fp16.safetensors",
    "ema_vae_fp16.safetensors",
]


def main() -> None:
    output_dir = Path("ckpts/SeedVR2-1.4B")
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(output_dir),
        allow_patterns=FILES,
    )
    print(f"Downloaded SeedVR2-1.4B checkpoints to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
