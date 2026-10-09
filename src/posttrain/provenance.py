"""The boundary and prompt contract shared by training, resume and export."""

import json
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def training_contract(config, assets, prompts):
    from .initialization import boundary_manifest

    return json.loads(
        json.dumps(
            {
                "config": asdict(config),
                "boundary": boundary_manifest(config, assets),
                "prompts": prompts,
            }
        )
    )
