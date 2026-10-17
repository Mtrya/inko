"""Prompt manifests independent of training datasets and trainer state."""

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


from src.evaluation.generation.prompts import load_prompts


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
