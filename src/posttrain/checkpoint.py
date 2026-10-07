"""Durable complete-round checkpoints with per-rank RNG and strict identity."""

import json
import os
import random
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .distributed import barrier, world_size
from .records import atomic_json, file_hash, identity
from .schema import require


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def save_tensor_file(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "accelerator": None
        if device.type == "cpu"
        else getattr(torch, device.type).get_rng_state(device).cpu(),
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type != "cpu":
        getattr(torch, device.type).set_rng_state(state["accelerator"], device)


def publish(root, step, contract, state, rank_state, rank, device):
    """Only manifest-bearing directories are recovery points; staging is never reused."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    token = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(token, src=0, device=device)
    staging = root / f".incomplete-{step:06d}-{token[0]}"
    if rank == 0:
        staging.mkdir()
    barrier()
    save_tensor_file(
        staging / f"rank-{rank:03d}.pt",
        cpu_state({**rank_state, "rng": rng_state(device)}),
    )
    if rank == 0:
        save_tensor_file(staging / "state.pt", cpu_state(state))
    barrier()
    destination = root / f"step-{step:06d}"
    if rank == 0:
        files = {p.name: file_hash(p) for p in staging.iterdir()}
        require(
            set(files)
            == {"state.pt", *(f"rank-{r:03d}.pt" for r in range(world_size()))},
            "checkpoint inventory incomplete",
        )
        atomic_json(
            staging / "manifest.json",
            {
                "schema": 1,
                "step": step,
                "world_size": world_size(),
                "contract": contract,
                "contract_sha256": identity(contract),
                "files": files,
            },
        )
        require(not destination.exists(), "refusing to overwrite committed checkpoint")
        staging.rename(destination)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    barrier()
    return destination


def inspect(path, expected_contract=None):
    path = Path(path)
    require(
        not path.name.startswith(".incomplete-"),
        "unpublished checkpoint staging directory",
    )
    record = json.loads((path / "manifest.json").read_text())
    require(
        record["schema"] == 1
        and record["contract_sha256"] == identity(record["contract"]),
        "invalid checkpoint contract",
    )
    if expected_contract is not None:
        require(
            record["contract"] == expected_contract,
            "exact resume contract differs; initialize an explicit new run",
        )
    required = {"state.pt", *(f"rank-{r:03d}.pt" for r in range(record["world_size"]))}
    require(set(record["files"]) == required, "incomplete checkpoint files")
    for name, digest in record["files"].items():
        require(file_hash(path / name) == digest, "checkpoint file digest differs")
    return record


def load(path, contract, rank):
    manifest = inspect(path, contract)
    require(manifest["world_size"] == world_size(), "resume learner count differs")
    state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
    if "step" in state:
        require(
            state["step"] == manifest["step"], "checkpoint manifest/state step differs"
        )
    return (
        state,
        torch.load(
            Path(path) / f"rank-{rank:03d}.pt", map_location="cpu", weights_only=False
        ),
    )


def assert_exact(expected, actual):
    if isinstance(expected, torch.Tensor):
        require(
            isinstance(actual, torch.Tensor)
            and expected.dtype == actual.dtype
            and expected.shape == actual.shape
            and torch.equal(expected.cpu(), actual.detach().cpu()),
            "restored tensor differs",
        )
    elif isinstance(expected, np.ndarray):
        require(np.array_equal(expected, actual), "restored array differs")
    elif isinstance(expected, dict):
        require(expected.keys() == actual.keys(), "restored keys differ")
        for k in expected:
            assert_exact(expected[k], actual[k])
    elif isinstance(expected, (list, tuple)):
        require(
            type(expected) is type(actual) and len(expected) == len(actual),
            "restored sequence differs",
        )
        for a, b in zip(expected, actual, strict=True):
            assert_exact(a, b)
    else:
        require(expected == actual, "restored scalar differs")
