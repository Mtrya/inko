"""Prepared rollout rows with rubric, aspect and optional style reference."""

import json
from pathlib import Path

from .schema import name, require
from .scoring.contracts import rubric_criteria


def load_rollout_prompts(path, shapes):
    path = Path(path).resolve()
    rows, seen = [], set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            name(row["id"])
            require(row["id"] not in seen, "duplicate rollout prompt ID")
            seen.add(row["id"])
            require(
                isinstance(row["text"], str) and bool(row["text"].strip()),
                "rollout prompt must be nonempty",
            )
            require(row["aspect"] in shapes, "no shape for rollout aspect")
            row["rubric"] = rubric_criteria(row["rubric"])
            reference = row.get("reference")
            if reference is not None:
                reference = (path.parent / reference).resolve()
                require(reference.is_file(), f"missing style reference: {reference}")
                reference = str(reference)
            rows.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "aspect": row["aspect"],
                    "rubric": row["rubric"],
                    "reference": reference,
                    "scene_id": row["scene_id"],
                }
            )
    require(bool(rows), "empty rollout pool")
    return rows
