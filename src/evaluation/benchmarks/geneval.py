"""Original GenEval: released metadata, Mask2Former and CLIP color checks."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.records import read_jsonl

from .common import link_image, load_source, mean, prompt, write_json


def prepare(assets, settings):
    rows = read_jsonl(Path(assets["source"]) / "prompts/evaluation_metadata.jsonl")
    return [
        prompt(
            "geneval",
            "original",
            i,
            row["prompt"],
            {"native": row},
            [settings["seed"] + i * 4 + j for j in range(4)],
        )
        for i, row in enumerate(rows)
    ]


def layout(groups, root):
    for group in groups:
        row = group["prompt"]
        directory = root / f"{int(row['prompt_id']):05d}"
        write_json(directory / "metadata.jsonl", row["metadata"]["native"])
        for sample in group["samples"]:
            link_image(
                sample["image_path"],
                directory / "samples" / f"{sample['draw']:04d}.png",
            )


def evaluate(groups, assets, settings, root, emit):
    upstream = load_source(Path(assets["source"]) / "evaluation/evaluate_images.py")
    upstream.args = SimpleNamespace(
        model_config=assets["model_config"], model_path=assets["detector"], options={}
    )
    (
        upstream.object_detector,
        (upstream.clip_model, upstream.transform, upstream.tokenizer),
        upstream.classnames,
    ) = upstream.load_models(upstream.args)
    upstream.THRESHOLD = 0.3
    upstream.COUNTING_THRESHOLD = 0.9
    upstream.MAX_OBJECTS = 16
    upstream.NMS_THRESHOLD = 1.0
    upstream.POSITION_THRESHOLD = 0.1
    emit.loaded()
    for group in groups:
        native = group["prompt"]["metadata"]["native"]
        results = [
            upstream.evaluate_image(s["image_path"], native) for s in group["samples"]
        ]
        emit(group, "detector", results)


def summarize(rows, assets):
    categories = defaultdict(list)
    for row in rows:
        for result in row["native"]:
            categories[result["tag"]].append(float(result["correct"]))
    scores = {key: mean(values) for key, values in categories.items()}
    return {"categories": scores, "overall": mean(scores.values())}
