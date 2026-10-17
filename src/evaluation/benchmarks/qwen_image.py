"""Qwen-Image-Bench's Q-Judger and hierarchical native aggregation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.records import append_jsonl, read_jsonl

from .common import load_source, prompt


def prepare(assets, settings):
    rows = read_jsonl(assets["data"])
    metadata = json.loads(
        (Path(assets["source"]) / "metadata/bench_metadata.json").read_text()
    )
    dims = {row["ID"]: row["dims_en"] for row in metadata}
    result = []
    for row in rows:
        if row["dims_en"] != dims[row["ID"]]:
            raise ValueError(f"Qwen metadata differs for ID {row['ID']}")
        result.append(
            prompt(
                "qwen_image",
                "released",
                row["ID"],
                row["prompt_en"],
                {"ID": row["ID"], "dims_en": row["dims_en"]},
                [settings["seed"] + row["ID"]],
                "en",
            )
        )
    return result


def layout(groups, root):
    path = root / "input.jsonl"
    path.write_text("")
    append_jsonl(
        path,
        [
            {
                "ID": g["prompt"]["metadata"]["ID"],
                "prompt": g["prompt"]["text"],
                "image_path": g["samples"][0]["image_path"],
            }
            for g in groups
        ],
    )


def upstream(assets):
    sys.path.insert(0, assets["source"])
    return load_source(Path(assets["source"]) / "judge.py")


def evaluate(groups, assets, settings, root, emit):
    import pandas as pd

    native = upstream(assets)
    from backends.ms_swift_backend import MsSwiftJudge

    judge = MsSwiftJudge(
        assets["model"], max_batch_size=settings["batch_size"], max_new_tokens=4096
    )
    args = SimpleNamespace(batch_size=settings["batch_size"])
    emit.loaded()
    for start in range(0, len(groups), settings["batch_size"]):
        batch = groups[start : start + settings["batch_size"]]
        inputs = pd.DataFrame(
            [
                {
                    "ID": group["prompt"]["metadata"]["ID"],
                    "prompt": group["prompt"]["text"],
                    "image_path": group["samples"][0]["image_path"],
                }
                for group in batch
            ]
        )
        meta = pd.DataFrame([group["prompt"]["metadata"] for group in batch])
        results, failures, _, image_failures = native._run_batch_inference(
            judge, args, inputs, meta, "Q-Judger"
        )
        if failures or image_failures:
            raise ValueError(
                f"Q-Judger failed to evaluate IDs {inputs['ID'].tolist()}: {results}"
            )
        for group, result in zip(batch, results, strict=True):
            emit(group, "judge", result)


def summarize(rows, assets):
    return upstream(assets).compute_bench_scores(
        [json.loads(row["native"]["judge_model_output"]) for row in rows]
    )
