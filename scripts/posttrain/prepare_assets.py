"""Download pinned scorer sources/weights and write a machine-local asset map.

Run in an internet-enabled preparation environment. Dependency overlays are
supplied separately; this command never installs into a running training runtime.
"""

import argparse
import json
import subprocess
from pathlib import Path

from src.posttrain.config import load_config
from src.posttrain.records import atomic_json


def source_checkout(repository, revision, destination):
    if not destination.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", repository, str(destination)], check=True
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", revision],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"existing source differs from pin: {destination}")


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/posttrain.toml"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-type", choices=["npu", "cuda", "cpu"], default="npu")
    parser.add_argument(
        "--overlays",
        type=Path,
        help="JSON mapping learner/hpsv3/vqascore to ordered dependency directories",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    overlays = json.loads(args.overlays.read_text()) if args.overlays else {}
    root = args.artifact_root.resolve()
    assets = {
        "checkpoint": str(args.checkpoint.resolve()),
        "text_encoder": str(args.text_encoder.resolve()),
        "vae": str(args.vae.resolve()),
        "device_type": args.device_type,
        "learner": {"python_paths": overlays.get("learner", [])},
    }
    for role, spec in config["scorers"].items():
        source = root / "sources" / f"{role}-{spec['source_revision']}"
        source.parent.mkdir(parents=True, exist_ok=True)
        source_checkout(spec["source_repository"], spec["source_revision"], source)
        weights = root / "weights" / f"{role}-{spec['weights_revision']}"
        patterns = (
            [spec["weights_file"]]
            if role == "hpsv3"
            else ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]
        )
        snapshot_download(
            spec["weights_repository"],
            revision=spec["weights_revision"],
            local_dir=weights,
            allow_patterns=patterns,
        )
        assets[role] = {
            "source": str(source),
            "weights": str(
                weights / spec["weights_file"] if role == "hpsv3" else weights
            ),
            "python_paths": overlays.get(role, []),
        }
        if role == "hpsv3":
            backbone = root / "weights" / f"hpsv3-backbone-{spec['backbone_revision']}"
            snapshot_download(
                spec["backbone_repository"],
                revision=spec["backbone_revision"],
                local_dir=backbone,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.jinja",
                    "*.model",
                ],
            )
            assets[role]["backbone"] = str(backbone)
    atomic_json(args.output, assets)


if __name__ == "__main__":
    main()
