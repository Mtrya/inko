"""Explicit cold-checkpoint initialization; optimizer/cursor/RNG are reset."""

from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import inspect
from .records import file_hash
from .rollout import tensor_digest
from .schema import require


def boundary_manifest(config, assets):
    key = config.stages.initial_checkpoint
    if not key:
        return None
    path = Path(assets["training_checkpoints"][key])
    require(
        file_hash(path / "manifest.json") == config.stages.initial_sha256,
        "initial cold checkpoint manifest bytes differ",
    )
    manifest = inspect(path)
    source = manifest["contract"]["config"]
    require(
        source["student"] == asdict(config.student),
        "initial student tuning/capacity differs",
    )
    for role in ("initialization", "teacher"):
        if role == "teacher" and config.joint.dmd_coefficient == 0:
            continue
        current = asdict(getattr(config, role))
        require(
            all(
                current[name] == source[role][name]
                for name in ("weights", "sha256", "architecture_sha256")
            ),
            "initial base/teacher identity differs",
        )
    if config.joint.dmd_coefficient > 0:
        require(
            all(
                asdict(config.fake)[name] == source["fake"][name]
                for name in ("strategy", "rank", "multiplier")
            ),
            "initial fake capacity differs",
        )
    require(manifest["step"] > 0, "joint entry requires a committed cold update")
    return {
        "manifest_sha256": config.stages.initial_sha256,
        "source_contract_sha256": manifest["contract_sha256"],
        "generator_steps": manifest["step"],
        "fake_steps": manifest["step"] * source["fake"]["updates"],
    }


def load_boundary(config, assets, roles, expected):
    path = Path(assets["training_checkpoints"][config.stages.initial_checkpoint])
    require(
        boundary_manifest(config, assets) == expected, "initial cold boundary changed"
    )
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    require(
        state["stage"] == "cold"
        and state["step"]
        == state["stage_generator_steps"]
        == state["global_generator_steps"]
        == expected["generator_steps"]
        and state["fake_steps"] == expected["fake_steps"],
        "initial checkpoint is not the declared committed cold boundary",
    )
    require(
        state["architecture"] == roles.student.get_config(),
        "initial architecture differs",
    )
    roles.student.load_state_dict(state["student"])
    if roles.fake is not None:
        frozen = {
            n: tensor_digest(p)
            for n, p in roles.fake.named_parameters()
            if not p.requires_grad
        }
        roles.fake.load_state_dict(state["fake"])
        require(
            frozen
            == {
                n: tensor_digest(p)
                for n, p in roles.fake.named_parameters()
                if not p.requires_grad
            },
            "initial fake changed its frozen teacher base",
        )
    roles.reset_ema()
    return expected["generator_steps"], expected["fake_steps"]
