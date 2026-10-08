"""Independent deterministic protocol fixture; never a production reward model."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageStat

from src.posttrain.records import atomic_json, file_hash, identity
from src.posttrain.schema import decode, digest, require
from src.posttrain.scoring.config import Limits
from src.posttrain.scoring.contracts import Capabilities, Channel
from src.posttrain.scoring.service import ScoringEngine, make_server


@dataclass(frozen=True)
class Fixture:
    source_sha256: str
    input_schema: str
    structured_fields: tuple[str, ...]
    outputs: tuple[Channel, ...]
    rubric: str
    limits: Limits


def load_fixture(path):
    config = decode(Fixture, json.loads(Path(path).read_text()))
    digest(config.source_sha256)
    require(
        file_hash(__file__) == config.source_sha256, "protocol fixture source differs"
    )
    require(
        config.structured_fields == ("rubric",) and bool(config.rubric),
        "fixture rubric contract differs",
    )
    require(
        len(config.outputs) == 1
        and config.outputs[0].direction == "lower"
        and config.outputs[0].bounds == (0.0, 1.0),
        "fixture output contract differs",
    )
    return config


class Backend:
    def __init__(self, config):
        self.config = config

    def score(self, inputs):
        values = []
        for item in inputs:
            require(
                item.structured == {"rubric": self.config.rubric},
                "fixture structured input differs",
            )
            with Image.open(item.image_path) as image:
                cost = ImageStat.Stat(image.convert("RGB")).mean[0] / 255.0
            values.append({self.config.outputs[0].name: cost})
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    config = load_fixture(args.config)
    fingerprint = identity(asdict(config))
    capabilities = Capabilities(
        1,
        fingerprint,
        config.input_schema,
        config.structured_fields,
        config.outputs,
        config.limits.max_batch_size,
        ("shared_file", "png_base64"),
        True,
        True,
    )
    engine = ScoringEngine(
        Backend(config),
        capabilities,
        config.limits,
        args.output / "cache",
        telemetry=lambda: {
            "rubric_diagnostic": json.dumps(
                {
                    "rubric": config.rubric,
                    "kind": "protocol-fixture",
                    "quality_claim": False,
                }
            ),
        },
    )
    server = make_server(("127.0.0.1", args.port), engine)
    atomic_json(
        args.output / "ready.json",
        {
            "pid": os.getpid(),
            "endpoint": f"http://127.0.0.1:{server.server_port}",
            "fingerprint": fingerprint,
            "capabilities": asdict(capabilities),
            "source_sha256": config.source_sha256,
        },
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
