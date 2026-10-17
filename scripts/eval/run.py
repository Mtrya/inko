"""Prepare benchmark catalogs, generate images, and run official scorers."""

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Generate missing images from a catalog"
    )
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--assets", type=Path, required=True)
    generate.add_argument("--prompts", type=Path, required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output", type=Path, required=True)
    for name in ("prepare", "score"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--assets", type=Path, required=True)
        command.add_argument(
            "--benchmark",
            choices=("geneval", "dpg", "oneig", "tiif", "qwen_image"),
            required=True,
        )
        command.add_argument("--output", type=Path, required=True)
        if name == "score":
            command.add_argument("--images", type=Path, required=True)
            command.add_argument("--model", required=True)
    args = parser.parse_args()

    if args.command != "generate":
        from src.evaluation.benchmarks.run import prepare, score

        with args.config.open("rb") as stream:
            settings = tomllib.load(stream)["benchmarks"][args.benchmark]
        with args.assets.open("rb") as stream:
            assets = tomllib.load(stream)["benchmarks"][args.benchmark]
        if args.command == "prepare":
            result = prepare(args.benchmark, settings, assets, args.output)
        else:
            result = score(
                args.benchmark, settings, assets, args.model, args.images, args.output
            )
        print(json.dumps(result, indent=2))
        return

    from src.evaluation.generation.config import ExternalModel, load_config
    from src.evaluation.records import load_catalog

    with args.assets.open("rb") as stream:
        assets = tomllib.load(stream)
    config = load_config(args.config)
    model = next((m for m in config.models if m.name == args.model), None)
    if model is None:
        raise ValueError(f"unknown model {args.model}")
    if isinstance(model, ExternalModel):
        entry = assets["models"][model.asset]
        python = str(Path(entry["python"]).absolute())
        if python != str(Path(sys.executable).absolute()):
            env = {**os.environ, **entry.get("env", {})}
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            subprocess.run(
                [python, "-m", "scripts.eval.run", *sys.argv[1:]], env=env, check=True
            )
            return
    from src.evaluation.generation.run import run

    print(
        json.dumps(
            run(
                config,
                assets,
                load_catalog(args.prompts),
                args.model,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
