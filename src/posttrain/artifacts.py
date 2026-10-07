"""Content inventories and pinned model/source verification."""

import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

from .records import file_hash
from .schema import require


def package_versions():
    names = {
        re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower()
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    return {name: importlib.metadata.version(name) for name in sorted(names)}


def tree_identity(path):
    path = Path(path)
    if path.is_file():
        return file_hash(path)
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and not any(
            x.startswith(".") or x == "__pycache__" for x in p.relative_to(path).parts
        )
        and p.suffix != ".pyc"
    )
    require(bool(files), f"empty or missing artifact {path}")
    return {str(p.relative_to(path)): file_hash(p) for p in files}


def verify_source(path, pin):
    revision = subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()
    require(revision == pin.revision, "scorer source revision mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", path, "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    require(not dirty, "scorer source checkout has modifications")
    return revision


def verify_model(path, manifest_path, pin):
    """Manifest records checksums obtained from the pinned upstream file metadata."""
    manifest = json.loads(Path(manifest_path).read_text())
    require(
        manifest["repository"] == pin.repository
        and manifest["revision"] == pin.revision,
        "weight manifest revision mismatch",
    )
    root = Path(path)
    actual = {}
    for relative, expected in manifest["files"].items():
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "unsafe model member",
        )
        member = root / relative if root.is_dir() else root
        require(
            root.is_dir() or len(manifest["files"]) == 1,
            "file model manifest has multiple members",
        )
        if expected["algorithm"] == "sha256":
            value = file_hash(member)
        elif expected["algorithm"] == "git_blob_sha1":
            data = member.read_bytes()
            value = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        else:
            raise ValueError("unknown model checksum algorithm")
        require(value == expected["digest"], f"pinned model bytes differ: {relative}")
        actual[relative] = value
    require(bool(actual), "empty model manifest")
    if root.is_dir():
        visible = {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file()
            and not any(x.startswith(".") for x in p.relative_to(root).parts)
        }
        require(visible == actual.keys(), "model inventory has missing or extra files")
    return manifest
