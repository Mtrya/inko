"""Export an actual committed live/EMA student, merging only weight LoRA."""

import argparse
import json
import shutil
import uuid
from pathlib import Path

import torch

from src.models.inko import Inko
from src.posttrain.adapters import install_adapters, merge_student
from src.posttrain.checkpoint import assert_exact, inspect, save_tensor_file
from src.posttrain.records import atomic_json
from src.posttrain.schema import decode, require
from src.posttrain.training_config import TrainingConfig


def export(checkpoint, weights, output):
    checkpoint, output = Path(checkpoint), Path(output)
    manifest = inspect(checkpoint)
    config = decode(TrainingConfig, manifest["contract"]["config"])
    require(weights in {"live", "ema"}, "unknown student weight view")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    model = Inko(**state["architecture"]).float().eval()
    if config.student.mode == "lora":
        install_adapters(
            model,
            rank=config.student.rank,
            scale=config.student.alpha,
            targets=config.student.targets,
        )
    model.load_state_dict(state["student"])
    if weights != "live":
        shadow = state["ema"]
        parameters = dict(model.named_parameters())
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        require(shadow.keys() == expected, "EMA parameter inventory differs")
        with torch.no_grad():
            for name, value in shadow.items():
                parameters[name].copy_(value)
    merged = merge_student(model)
    require(not output.exists(), "export output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".incomplete-{output.name}-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        return _publish_export(output, staging, weights, state, merged)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _publish_export(output, staging, weights, state, merged):
    atomic_json(staging / "transformer_config.json", state["architecture"])
    save_tensor_file(staging / "weights.pt", merged.state_dict())
    reloaded = Inko.from_single_file(str(staging / "weights.pt"))
    assert_exact(merged.state_dict(), reloaded.state_dict())
    record = {
        "weights": weights,
        "step": state["step"],
        "committed_rounds": state["committed_rounds"],
    }
    atomic_json(staging / "export.json", record)
    staging.rename(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("live", "ema"), default="ema")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        record = inspect(args.checkpoint)
        decode(TrainingConfig, record["contract"]["config"])
        print(
            json.dumps(
                {
                    "valid": True,
                    "step": record["step"],
                    "committed_rounds": record["committed_rounds"],
                }
            )
        )
        return
    if args.output is None:
        parser.error("--output required")
    print(json.dumps(export(args.checkpoint, args.weights, args.output), indent=2))


if __name__ == "__main__":
    main()
