"""Prepare one pinned scorer and checksum manifest in an isolated asset tree."""

import argparse
import fnmatch
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from src.posttrain.artifacts import verify_model, verify_source
from src.posttrain.records import atomic_json, file_hash
from src.posttrain.schema import require
from src.posttrain.scoring.config import load_service_config


def source_checkout(pin, destination):
    if not destination.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", pin.repository, str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", pin.revision],
            check=True,
        )
    verify_source(str(destination), pin)


def model_manifest(pin, destination, *, download, source=None):
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(
        pin.repository, revision=pin.revision, files_metadata=True
    )
    files = {}
    for entry in info.siblings:
        if any(fnmatch.fnmatch(entry.rfilename, pattern) for pattern in pin.files):
            files[entry.rfilename] = {
                "algorithm": "sha256" if entry.lfs else "git_blob_sha1",
                "digest": entry.lfs.sha256 if entry.lfs else entry.blob_id,
            }
    if download:
        snapshot_download(
            pin.repository,
            revision=pin.revision,
            local_dir=destination,
            allow_patterns=pin.files,
        )
    if source is not None:
        destination.mkdir(parents=True, exist_ok=True)
        for relative in files:
            target = source / relative if source.is_dir() else source
            member = destination / relative
            member.parent.mkdir(parents=True, exist_ok=True)
            require(target.is_file(), f"missing pinned model member: {target}")
            if not member.exists():
                member.symlink_to(target.resolve())
    manifest = {"repository": pin.repository, "revision": pin.revision, "files": files}
    manifest_path = destination.parent / (destination.name + "-manifest.json")
    atomic_json(manifest_path, manifest)
    verify_model(destination, manifest_path, pin)
    return str(manifest_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="complete scorer service TOML"
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--assets", type=Path, help="existing service paths for verify-only preparation"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--runtime-files",
        type=Path,
        help="local JSON map from native runtime labels to version files",
    )
    parser.add_argument("--device-type", choices=["npu", "cuda", "cpu"], default="npu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_service_config(args.config)
    if args.validate_only:
        print(json.dumps(asdict(config.capabilities), indent=2))
        return
    if args.output is None or (args.artifact_root is None and args.assets is None):
        parser.error("--output plus --artifact-root or --assets required")
    if args.assets:
        assets = json.loads(args.assets.read_text())
    else:
        root = args.artifact_root.resolve()
        source = root / "source"
        source.parent.mkdir(parents=True, exist_ok=True)
        source_checkout(config.semantics.source, source)
        assets = {
            "source": str(source),
            "weights": str(root / "weights"),
            "device_type": args.device_type,
        }
        if config.semantics.backend == "hpsv3":
            assets["backbone"] = str(root / "backbone")
    verify_source(assets["source"], config.semantics.source)
    if args.runtime_files:
        assets["runtime_files"] = json.loads(args.runtime_files.read_text())
    require(
        set(assets.get("runtime_files", {})) == set(config.semantics.runtime_files),
        "provide the scorer's native runtime file map",
    )
    for key, expected in config.semantics.runtime_files.items():
        require(
            file_hash(assets["runtime_files"][key]) == expected,
            f"unqualified native runtime: {key}",
        )
    source = Path(assets["weights"]) if args.assets and args.artifact_root else None
    if source is not None:
        assets["weights"] = str(args.artifact_root.resolve() / "weights")
    assets["weights_manifest"] = model_manifest(
        config.semantics.weights,
        Path(assets["weights"]),
        download=not args.assets,
        source=source,
    )
    if config.semantics.backend == "hpsv3":
        source = (
            Path(assets["backbone"]) if args.assets and args.artifact_root else None
        )
        if source is not None:
            assets["backbone"] = str(args.artifact_root.resolve() / "backbone")
        assets["backbone_manifest"] = model_manifest(
            config.options.backbone,
            Path(assets["backbone"]),
            download=not args.assets,
            source=source,
        )
        if Path(assets["weights"]).is_dir():
            assets["weights"] = str(Path(assets["weights"]) / "HPSv3.safetensors")
    atomic_json(args.output, assets)


if __name__ == "__main__":
    main()
