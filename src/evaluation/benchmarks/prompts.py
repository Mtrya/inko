"""Prompt manifests independent of training datasets and trainer state."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


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


def load_evaluation_prompts(config, suite):
    settings = config.evaluation
    monitor = suite == "monitor"
    path = settings.monitor_suite if monitor else settings.suite
    rows = load_prompts(REPO / path, config.sampling.shapes)
    if monitor:
        return rows
    if settings.prompt_limit:
        rows = rows[: settings.prompt_limit]
    if settings.draws_per_prompt == 1:
        return rows
    return [
        {
            **row,
            "id": f"{row['id']}_seed{draw}",
            "seed": int(
                hashlib.sha256(
                    f"eval-noise:{row['scene_id']}:{draw}".encode()
                ).hexdigest()[:15],
                16,
            ),
        }
        for row in rows
        for draw in range(settings.draws_per_prompt)
    ]


def load_rubric_prompts(path, shapes, count, seed):
    import random

    from src.posttrain.prompts import load_rollout_prompts

    rows = load_rollout_prompts(path, shapes)
    rng = random.Random(seed)
    rows = rng.sample(rows, min(count, len(rows)))
    return [
        {
            **row,
            "seed": int(
                hashlib.sha256(f"rubric-eval:{seed}:{row['id']}".encode()).hexdigest()[
                    :15
                ],
                16,
            ),
        }
        for row in rows
    ]
