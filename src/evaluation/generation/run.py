"""Generate saved images from a prepared catalog, resuming completed outputs."""

import fcntl
import json
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from src.evaluation.records import (
    append_jsonl,
    path_component,
    prompt_key,
    read_jsonl,
    sample_key,
)
from src.posttrain.records import atomic_json
from src.utils import prompt_contract

from .config import ExternalModel
from .workers import ResidentWorkers, run_pipeline


def run(config, assets, catalog, model, output):
    arm = next((a for a in config.models if a.name == model), None)
    if arm is None:
        raise ValueError(f"unknown model {model}")
    output = Path(output).resolve()
    directory = output / arm.name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, assets, catalog, arm, output)


def _run(config, assets, catalog, arm, output):
    external = isinstance(arm, ExternalModel)
    shapes = arm.shapes if external else config.sampling.shapes
    batch_size = arm.batch_size if external else config.sampling.batch_size
    sampling = asdict(config.sampling)
    sampling.pop("batch_size")
    recipe = (
        {
            "model": asdict(arm),
            "weights": str(Path(assets["models"][arm.asset]["path"]).resolve()),
        }
        if external
        else {
            "model": asdict(arm),
            "sampling": sampling,
            "weights": str(Path(assets["checkpoints"][arm.checkpoint.asset]).resolve()),
            "text_encoder": str(Path(assets["text_encoder"]).resolve()),
            "vae": str(Path(assets["vae"]).resolve()),
            "conditioning": {
                "template": prompt_contract.PROMPT_TEMPLATE,
                "system": prompt_contract.SYSTEM_PROMPT,
                "drop_idx": prompt_contract.DROP_IDX,
                "max_sequence_length": prompt_contract.MAX_SEQUENCE_LENGTH,
            },
        }
    )
    if external:
        recipe["model"].pop("batch_size")
    recipe = json.loads(json.dumps(recipe))
    root = output / arm.name
    settings = root / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != recipe:
        raise ValueError("generation settings changed; use a new output directory")
    atomic_json(settings, recipe)
    catalogs = {}
    for row in catalog:
        catalogs.setdefault((row["benchmark"], row["split"]), []).append(row)
    tasks, journals, records = [], {}, {}
    recovered = 0
    for (benchmark, split), prompts in catalogs.items():
        directory = root / path_component(benchmark) / path_component(split)
        directory.mkdir(parents=True, exist_ok=True)
        catalog_path = directory / "prompts.jsonl"
        ordered = sorted(prompts, key=prompt_key)
        if catalog_path.exists():
            if read_jsonl(catalog_path) != ordered:
                raise ValueError("prompt catalog changed; use a new output directory")
        else:
            temporary = catalog_path.with_suffix(".tmp")
            temporary.write_text("")
            append_jsonl(temporary, ordered)
            temporary.replace(catalog_path)
        journal = directory / "samples.jsonl"
        saved = read_jsonl(journal, repair_tail=True)
        completed = {sample_key(r): r for r in saved}
        if len(completed) != len(saved):
            raise ValueError("duplicate saved image record")
        pending, retained = {}, []
        for prompt in prompts:
            aspect = prompt["metadata"]["aspect"]
            height, width = shapes[aspect]
            for draw, seed in enumerate(prompt["seeds"]):
                path = (
                    directory
                    / "images"
                    / path_component(prompt["prompt_id"])
                    / path_component(prompt["variant"])
                    / f"{draw}.png"
                )
                record = {
                    "model": arm.name,
                    "benchmark": benchmark,
                    "split": split,
                    "prompt_id": prompt["prompt_id"],
                    "variant": prompt["variant"],
                    "draw": draw,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "image_path": str(path.relative_to(output)),
                }
                key = sample_key(record)
                sample_id = json.dumps(key, ensure_ascii=False)
                record["sample_id"] = sample_id
                journals[sample_id] = journal
                if key in completed and any(
                    completed[key][k] != v for k, v in record.items()
                ):
                    raise ValueError("saved image record differs from its request")
                valid = False
                if path.exists():
                    try:
                        with Image.open(path) as image:
                            valid = (
                                image.size == (width, height) and image.format == "PNG"
                            )
                            image.verify()
                    except (OSError, ValueError, SyntaxError):
                        valid = False
                if valid:
                    if key not in completed:
                        recovered += 1
                    retained.append(completed.get(key, record))
                    records[sample_id] = retained[-1]
                    continue
                pending.setdefault(aspect, []).append(
                    (
                        {"text": prompt["text"], "seed": seed, "aspect": aspect},
                        record,
                        str(path),
                    )
                )
        # Repair the journal once, retaining complete PNGs even if their append
        # was interrupted. Missing/corrupt PNGs are regenerated below.
        if saved != retained:
            temporary = journal.with_suffix(".tmp")
            temporary.write_text("")
            append_jsonl(temporary, retained)
            temporary.replace(journal)
        for group in pending.values():
            for offset in range(0, len(group), batch_size):
                chunk = group[offset : offset + batch_size]
                tasks.append(
                    {
                        "task_id": str(len(tasks)),
                        "arm": arm,
                        "prompts": [p for p, _, _ in chunk],
                        "records": [r for _, r, _ in chunk],
                        "sample_ids": [r["sample_id"] for _, r, _ in chunk],
                        "image_paths": [p for _, _, p in chunk],
                    }
                )

    def generated(batch):
        for row in batch:
            records[row["sample_id"]] = row
            append_jsonl(journals[row["sample_id"]], [row])

    started = time.perf_counter()
    startup, telemetry = [], []
    if tasks:
        worker_assets = {**assets, "external_model": arm} if external else assets
        with ResidentWorkers(
            config.sampling, config.execution, worker_assets
        ) as workers:
            startup = workers.startup
            telemetry = run_pipeline(tasks, workers, config.execution, generated)
    summary = {
        "samples": len(records),
        "generated": sum(len(t["records"]) for t in tasks),
        "recovered_images": recovered,
        "wall_seconds": time.perf_counter() - started,
        "execution": asdict(config.execution),
        "batch_size": batch_size,
        "device_type": assets["device_type"],
        "startup": startup,
        "workers": telemetry,
    }
    atomic_json(root / "execution" / f"{time.time_ns()}.json", summary)
    return summary
