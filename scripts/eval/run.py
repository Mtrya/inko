"""Standalone evaluation generation from prepared prompt catalogs."""

import argparse
import json
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
    args = parser.parse_args()

    from src.evaluation.generation.config import load_config
    from src.evaluation.generation.run import run
    from src.evaluation.records import load_catalog

    with args.assets.open("rb") as stream:
        assets = tomllib.load(stream)
    print(
        json.dumps(
            run(
                load_config(args.config),
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
