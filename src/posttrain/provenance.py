"""The training contract shared by training, resume and export."""

import json
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def training_contract(config, prompts):
    return json.loads(json.dumps({"config": asdict(config), "prompts": prompts}))
