"""Prompt validation, explicit noise seeds and resolution planning."""

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


# Module-level cache: bucket_id -> (h_lat, w_lat), keyed by dataset path
_BUCKET_SHAPE_CACHE: Dict[str, Dict[int, Tuple[int, int]]] = {}


def load_prompt_suite(path: str) -> List[Dict[str, Any]]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    ids = [p["id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    for prompt in prompts:
        resolved_prompt_seed(prompt)
        if "scene_id" in prompt:
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", prompt["scene_id"]):
                raise ValueError("scene_id must be a safe filename component")
            if prompt.get("lang") not in ("zh", "en") or prompt.get("variant") not in (
                "short",
                "long",
            ):
                raise ValueError(
                    "scene prompts require lang=zh/en and variant=short/long"
                )
    scenes = {}
    for prompt in prompts:
        if "scene_id" in prompt:
            scenes.setdefault(prompt["scene_id"], []).append(prompt)
    for scene, records in scenes.items():
        variants = {(p["lang"], p["variant"]) for p in records}
        if len(records) != 4 or len(variants) != 4:
            raise ValueError(
                f"scene {scene} requires exactly four language/length variants"
            )
        if len({resolved_prompt_seed(p) for p in records}) != 1:
            raise ValueError(f"scene {scene} variants must share a seed")
        if len({_aspect(p) for p in records}) != 1:
            raise ValueError(f"scene {scene} variants must share an aspect ratio")
    return prompts


def resolved_prompt_seed(prompt: Dict[str, Any]) -> int:
    """Require explicit noise seeds so prompt edits cannot change the noise."""
    seed = prompt.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("prompt seed must be an integer in [0, 2**63)")
    return seed


def bucket_shapes_from_dataset(
    dataset_path: str, scan_rows: int = 2000
) -> Dict[int, Tuple[int, int]]:
    """Map resolution_bucket_id -> (h_lat, w_lat) from actual eval data."""
    if dataset_path in _BUCKET_SHAPE_CACHE:
        return _BUCKET_SHAPE_CACHE[dataset_path]

    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    scan = dataset.select(range(min(scan_rows, len(dataset))))
    shapes: Dict[int, Tuple[int, int]] = {}
    for item in scan:
        bid = int(item["resolution_bucket_id"])
        if bid not in shapes:
            z = item["latents"]
            shapes[bid] = (int(z.shape[-2]), int(z.shape[-1]))
    _BUCKET_SHAPE_CACHE[dataset_path] = shapes
    return shapes


def _aspect(prompt: Dict[str, Any]) -> float:
    """Prompt aspect as w/h (e.g. '3:4' -> 0.75)."""
    spec = prompt.get("aspect", "1:1")
    w, h = spec.split(":")
    return float(w) / float(h)


def assign_bucket(prompt: Dict[str, Any], shapes: Dict[int, Tuple[int, int]]) -> int:
    pa = _aspect(prompt)
    best, best_dist = None, float("inf")
    for bid, (h_lat, w_lat) in shapes.items():
        dist = abs(math.log((w_lat / h_lat) / pa))
        if dist < best_dist:
            best, best_dist = bid, dist
    return best


def load_prompt_plan(
    prompts_path: str, eval_dataset_path: str
) -> Tuple[List[Dict[str, Any]], Dict[int, Tuple[int, int]]]:
    """Load the fixed suite and attach each prompt's nearest resolution bucket.

    Returns the prompt records (each carrying ``_bucket``) and the bucket
    shapes read from the eval dataset, so a caller can either sample the whole
    suite or shard it across processes and still sample each prompt exactly as
    training does.
    """
    prompts = load_prompt_suite(prompts_path)
    shapes = bucket_shapes_from_dataset(eval_dataset_path)
    for prompt in prompts:
        prompt["_bucket"] = assign_bucket(prompt, shapes)
    return prompts, shapes
