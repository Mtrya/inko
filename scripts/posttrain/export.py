"""Export an actual committed live/EMA student, merging only weight LoRA."""

import argparse
import json
from pathlib import Path

import torch

from src.models.inko import Inko
from src.posttrain.adapters import install_adapters, merge_student
from src.posttrain.checkpoint import assert_exact, inspect, save_tensor_file
from src.posttrain.records import atomic_json, file_hash
from src.posttrain.schema import decode, require
from src.posttrain.training_config import TrainingConfig


def export(checkpoint, weights, output):
    checkpoint, output = Path(checkpoint), Path(output)
    require(weights in {"live", "ema"}, "select live or EMA weights")
    manifest = inspect(checkpoint)
    config = decode(TrainingConfig, manifest["contract"]["config"])
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    model = Inko(**state["architecture"]).float().eval()
    if config.student.mode == "lora":
        install_adapters(
            model,
            kind="weight",
            rank=config.student.rank,
            scale=config.student.alpha,
            targets=config.student.targets,
        )
    model.load_state_dict(state["student"])
    if weights == "ema":
        require(state["ema"] is not None, "checkpoint has no student EMA")
        parameters = dict(model.named_parameters())
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        require(state["ema"].keys() == expected, "EMA parameter inventory differs")
        with torch.no_grad():
            for name, value in state["ema"].items():
                parameters[name].copy_(value)
    merged = merge_student(model)
    require(not output.exists(), "export output already exists")
    output.mkdir(parents=True)
    atomic_json(output / "transformer_config.json", state["architecture"])
    save_tensor_file(output / "weights.pt", merged.state_dict())
    reloaded = Inko.from_single_file(str(output / "weights.pt"))
    assert_exact(merged.state_dict(), reloaded.state_dict())
    record = {
        "schema": 1,
        "weights": weights,
        "tuning": config.student.mode,
        "step": state["step"],
        "base": manifest["contract"]["config"]["initialization"],
        "source_checkpoint": file_hash(checkpoint / "manifest.json"),
        "contract_sha256": manifest["contract_sha256"],
        "merged_sha256": file_hash(output / "weights.pt"),
        "architecture_sha256": file_hash(output / "transformer_config.json"),
        "student_targets": state["ownership"]["student"],
        "reload_exact": True,
    }
    atomic_json(output / "export.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=["live", "ema"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        record = inspect(args.checkpoint)
        config = record["contract"]["config"]
        require(
            args.weights != "ema" or config["training"]["ema_enabled"], "EMA disabled"
        )
        print(json.dumps({"valid": True, "step": record["step"]}))
        return
    if args.output is None:
        parser.error("--output required")
    print(json.dumps(export(args.checkpoint, args.weights, args.output), indent=2))


if __name__ == "__main__":
    main()
