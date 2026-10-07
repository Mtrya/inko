"""Prompt manifests independent of training datasets and trainer state."""

import json
import re
import unicodedata
from pathlib import Path


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_prompts(path: str | Path, shapes: dict) -> list[dict]:
    records = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    seen = set()
    for record in records:
        name = record["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("prompt IDs must be unique safe filename components")
        seen.add(name)
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError("prompt text must be nonempty")
        if type(record["seed"]) is not int or not 0 <= record["seed"] < 2**63:
            raise ValueError("each prompt requires an explicit nonnegative seed")
        if record["aspect"] not in shapes:
            raise ValueError(f"no explicit shape for aspect {record['aspect']}")
        record.setdefault("scene_id", name)
    if not records:
        raise ValueError("empty prompt suite")
    return records
