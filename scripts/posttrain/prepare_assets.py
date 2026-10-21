"""Prepare all reward-model assets; runtime startup only loads these paths."""

import argparse
import fnmatch
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from src.posttrain.artifacts import verify_model
from src.posttrain.records import atomic_json
from src.posttrain.schema import require
from src.posttrain.scoring.config import load_server_config


def source_checkout(repository, destination):
    if not destination.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", repository, str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", "HEAD"],
            check=True,
        )


def model_manifest(model, destination, *, download, source=None):
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(model.repository, files_metadata=True)
    files = {}
    for entry in info.siblings:
        if any(fnmatch.fnmatch(entry.rfilename, pattern) for pattern in model.files):
            files[entry.rfilename] = {
                "algorithm": "sha256" if entry.lfs else "git_blob_sha1",
                "digest": entry.lfs.sha256 if entry.lfs else entry.blob_id,
            }
    if download:
        snapshot_download(
            model.repository,
            local_dir=destination,
            allow_patterns=model.files,
        )
    if source is not None:
        destination.mkdir(parents=True, exist_ok=True)
        for relative in files:
            target = source / relative if source.is_dir() else source
            member = destination / relative
            member.parent.mkdir(parents=True, exist_ok=True)
            require(target.is_file(), f"missing prepared model member: {target}")
            if not member.exists():
                member.symlink_to(target.resolve())
    manifest = {"repository": model.repository, "files": files}
    manifest_path = destination.parent / (destination.name + "-manifest.json")
    atomic_json(manifest_path, manifest)
    verify_model(destination, manifest_path)
    return str(manifest_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="scoring server TOML"
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--assets", type=Path, help="existing service paths for verify-only preparation"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-type", choices=["npu", "cuda", "cpu"], default="npu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_server_config(args.config)
    if args.validate_only:
        print(json.dumps(asdict(config), indent=2))
        return
    if args.output is None or (args.artifact_root is None and args.assets is None):
        parser.error("--output plus --artifact-root or --assets required")
    assets = (
        json.loads(args.assets.read_text())
        if args.assets
        else {"device_type": args.device_type}
    )
    for role in ("rubric", "hpsv3", "vqascore"):
        spec = getattr(config, role)
        existing = dict(assets.get(role, {}))
        root = args.artifact_root.resolve() / role if args.artifact_root else None
        prepared = dict(existing)
        if role != "rubric" and not existing:
            prepared["source"] = str(root / "source")
            source_checkout(spec.source.repository, Path(prepared["source"]))
        for field in ("weights", "backbone") if role == "hpsv3" else ("weights",):
            model = getattr(spec, field)
            source = Path(existing[field]) if existing else None
            destination = root / field if root else source
            model_manifest(
                model,
                destination,
                download=not existing,
                source=source if root and existing else None,
            )
            prepared[field] = str(destination)
        if role == "hpsv3" and Path(prepared["weights"]).is_dir():
            prepared["weights"] = str(Path(prepared["weights"]) / "HPSv3.safetensors")
        assets[role] = prepared
    atomic_json(args.output, assets)


if __name__ == "__main__":
    main()
