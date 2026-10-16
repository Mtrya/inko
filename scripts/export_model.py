"""Export pretrained EMA or merged posttraining weights for InkoPipeline.

Run ``python -m scripts.export_model --help`` from the repository root.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.models import Inko


def export_model(checkpoint: str, output: Path, pipeline_config: Path) -> None:
    model = Inko.from_single_file(checkpoint).eval()
    config = json.loads(pipeline_config.read_text())
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        output / "model.safetensors",
    )
    (output / "transformer_config.json").write_text(
        json.dumps(model.get_config(), indent=2) + "\n"
    )
    (output / "pipeline_config.json").write_text(json.dumps(config, indent=2) + "\n")
    reloaded = Inko.from_single_file(str(output / "model.safetensors"))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            reloaded.state_dict()[name], value, rtol=0, atol=0, msg=name
        )
    print(
        f"Exported and verified {sum(p.numel() for p in model.parameters()):,} parameters in {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Weights with adjacent transformer_config.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("configs/inference.json")
    )
    args = parser.parse_args()
    export_model(args.checkpoint, args.output, args.pipeline_config)


if __name__ == "__main__":
    main()
