"""CPU comparison tables from saved images and native benchmark judgments."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from src.evaluation.benchmarks.common import write_json
from src.evaluation.benchmarks.run import adapter, dimensions, result_key
from src.evaluation.records import path_component, prompt_key, read_jsonl, sample_key


def scores_match(published, native):
    """Aggregates recomputed from judgments may differ from the published
    summary by float summation-order noise; compare within tolerance."""
    if isinstance(published, dict) and isinstance(native, dict):
        return published.keys() == native.keys() and all(
            scores_match(published[key], native[key]) for key in published
        )
    if isinstance(published, list) and isinstance(native, list):
        return len(published) == len(native) and all(
            scores_match(a, b) for a, b in zip(published, native)
        )
    if isinstance(published, (int, float)) and isinstance(native, (int, float)):
        return math.isclose(published, native, rel_tol=1e-9, abs_tol=1e-12)
    return published == native


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key != "count":
                result.update(flatten(child, f"{prefix}/{key}".strip("/")))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten(child, f"{prefix}/{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result


def paired_difference(name, before, after, assets, repeats, seed):
    """Resample source prompts, keeping all draws, variants and dimensions together."""
    indexed = [{result_key(r): r for r in rows} for rows in (before, after)]
    common = sorted(indexed[0].keys() & indexed[1].keys())
    clusters = defaultdict(list)
    for key in common:
        clusters[key[:3]].append(key)
    if not clusters:
        return {"source_prompts": 0, "results": 0, "delta": {}}
    summarize = adapter(name).summarize

    def difference(keys):
        values = [
            flatten(summarize([index[k] for k in keys], assets)) for index in indexed
        ]
        return {
            k: values[1][k] - values[0][k] for k in values[0].keys() & values[1].keys()
        }

    delta = difference(common)
    draws = defaultdict(list)
    rng = random.Random(seed)
    sources = list(clusters.values())
    for _ in range(repeats):
        keys = [
            key for cluster in rng.choices(sources, k=len(sources)) for key in cluster
        ]
        for metric, value in difference(keys).items():
            draws[metric].append(value)
    intervals = {}
    for metric, values in draws.items():
        values.sort()
        intervals[metric] = [values[int((len(values) - 1) * q)] for q in (0.025, 0.975)]
    return {
        "source_prompts": len(clusters),
        "results": len(common),
        "delta": delta,
        "prompt_bootstrap_95_percent": intervals,
    }


def _json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _resources(directory):
    paths = sorted((directory / "execution").glob("*.json"))
    generation = [_json(p) for p in paths]
    workers = [w for r in generation for w in r["workers"]]
    return {
        "generation_wall_seconds": sum(r["wall_seconds"] for r in generation),
        "generated_images": sum(r["generated"] for r in generation),
        "generation_device_seconds": sum(
            r["wall_seconds"] * len(r["startup"]) for r in generation
        ),
        "peak_device_bytes": max(
            (w["peak_memory_bytes"] for w in workers), default=None
        ),
        "startup": next((r["startup"] for r in generation if r["startup"]), []),
        "generation_executions": [str(p) for p in paths],
    }


def build(catalog, images, scores, models, pending, settings, output):
    images, scores, output = (
        Path(images).resolve(),
        Path(scores).resolve(),
        Path(output).resolve(),
    )
    if set(models) & set(pending) or len(set(models)) != len(models):
        raise ValueError(
            "requested models must be unique and separate from pending models"
        )
    catalogs = defaultdict(list)
    if len({prompt_key(p) for p in catalog}) != len(catalog):
        raise ValueError("duplicate report catalog prompt")
    for row in catalog:
        catalogs[row["benchmark"]].append(row)
    report = {"benchmarks": {}, "models": {}, "pending": pending, "paired": {}}
    for model in models:
        root = images / model
        report["models"][model] = {
            "inference": _json(root / "settings.json"),
            **_resources(root),
        }
    for name, prompts in catalogs.items():
        expected = {
            (*prompt_key(p), dimension)
            for p in prompts
            for dimension in dimensions(name, p)
        }
        expected_images = {
            (*prompt_key(p), draw): seed
            for p in prompts
            for draw, seed in enumerate(p["seeds"])
        }
        prompt_index = {prompt_key(p): p for p in prompts}
        entries, records, judge = {}, {}, None
        for model in models:
            root = scores / name / model
            request = _json(root / "request.json")
            all_rows = read_jsonl(root / "scores.jsonl")
            rows = [r for r in all_rows if prompt_key(r["prompt"]) in prompt_index]
            keys = [result_key(r) for r in rows]
            if len(set(keys)) != len(keys) or set(keys) - expected:
                raise ValueError(f"unexpected or duplicate results: {name}/{model}")
            if rows and (
                not request or request["model"] != model or request["benchmark"] != name
            ):
                raise ValueError(f"score request identity differs: {root}")
            if request:
                identity = {k: request[k] for k in ("settings", "assets")}
                if judge is not None and identity != judge:
                    raise ValueError(
                        f"different judge recipes for {name}; use separate reports"
                    )
                judge = identity
            saved = []
            for path in (images / model / path_component(name)).glob("*/samples.jsonl"):
                saved.extend(read_jsonl(path))
            samples = {sample_key(r): r for r in saved}
            if len(samples) != len(saved):
                raise ValueError(f"duplicate image records: {name}/{model}")
            available = set()
            for key, seed in expected_images.items():
                sample = samples.get((model, *key))
                if (
                    sample
                    and sample["seed"] == seed
                    and (images / sample["image_path"]).is_file()
                ):
                    available.add(key)
            for row in rows:
                p = row["prompt"]
                if p != prompt_index[prompt_key(p)]:
                    raise ValueError(
                        f"scored prompt differs from report catalog: {name}/{model}"
                    )
                for sample in row["samples"]:
                    key = (*prompt_key(sample), sample["draw"])
                    if sample["model"] != model or sample[
                        "seed"
                    ] != expected_images.get(key):
                        raise ValueError(
                            f"scored image identity differs: {name}/{model}"
                        )
            native = adapter(name).summarize(rows, request["assets"]) if rows else {}
            published = _json(root / "summary.json")
            if (
                published
                and len(rows) == len(all_rows)
                and set(keys) == expected
                and not scores_match(published["scores"], native)
            ):
                raise ValueError(f"published aggregate differs from judgments: {root}")
            scored = {(*prompt_key(s), s["draw"]) for r in rows for s in r["samples"]}
            missing = sorted(expected - set(keys))
            entries[model] = {
                "status": "complete"
                if not missing and len(available) == len(expected_images)
                else "incomplete",
                "expected_images": len(expected_images),
                "available_images": len(available),
                "evaluated_images": len(scored),
                "expected_results": len(expected),
                "evaluated_results": len(rows),
                "missing_results": missing,
                "missing_images": sorted(expected_images.keys() - available),
                "scores": native,
                "judgments": str(root / "scores.jsonl"),
                "judgment_seconds": sum(r["seconds"] for r in rows),
                "scoring_executions": read_jsonl(root / "execution.jsonl"),
            }
            records[model] = rows
        report["benchmarks"][name] = {"judge": judge, "models": entries}
        reference, comparison = settings["reference"], settings["comparison"]
        if records.get(reference) and records.get(comparison):
            recipes, guidance = [], {}
            for model in (reference, comparison):
                recipe = report["models"][model]["inference"]
                if recipe is None:
                    raise ValueError("paired comparison needs saved inference settings")
                recipe = {**recipe, "model": dict(recipe["model"])}
                recipe["model"].pop("name")
                recipe["model"].pop("checkpoint")
                guidance[model] = recipe["model"].pop("guidance")
                recipe.pop("weights")
                recipes.append(recipe)
            if recipes[0] != recipes[1]:
                raise ValueError("paired Inko inference recipes differ beyond CFG")
            report["paired"][name] = {
                "reference": reference,
                "comparison": comparison,
                "guidance": guidance,
                **paired_difference(
                    name,
                    records[reference],
                    records[comparison],
                    judge["assets"],
                    settings["bootstrap_samples"],
                    settings["seed"],
                ),
            }
    write_json(output / "report.json", report)
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report):
    lines = [
        "# Model evaluation",
        "",
        "Scores use each benchmark's native aggregation. Coverage refers to the supplied catalogs.",
        "",
    ]
    if report["pending"]:
        lines += ["Pending: " + ", ".join(report["pending"]) + ".", ""]
    lines += [
        "| Model | Steps | CFG | Precision | Generation wall (s) | Device-hours |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for model, entry in report["models"].items():
        recipe = entry["inference"] or {}
        arm = recipe.get("model", {})
        parameters = arm.get("parameters", {})
        steps = arm.get("steps", parameters.get("num_inference_steps", "—"))
        cfg = arm.get(
            "guidance",
            parameters.get("guidance_scale", parameters.get("true_cfg_scale", "—")),
        )
        precision = arm.get(
            "precision", recipe.get("sampling", {}).get("precision", "—")
        )
        lines.append(
            f"| {model} | {steps} | {cfg} | {precision} | {entry['generation_wall_seconds']:.2f} | {entry['generation_device_seconds'] / 3600:.4f} |"
        )
    lines.append("")
    for name, benchmark in report["benchmarks"].items():
        entries = benchmark["models"]
        lines += [
            f"## {name}",
            "",
            "| Model | Status | Available / expected images | Evaluated images | Results / expected |",
            "|---|---|---:|---:|---:|",
        ]
        for model, entry in entries.items():
            lines.append(
                f"| {model} | {entry['status']} | {entry['available_images']} / {entry['expected_images']} | {entry['evaluated_images']} | {entry['evaluated_results']} / {entry['expected_results']} |"
            )
        values = {m: flatten(e["scores"]) for m, e in entries.items()}
        metrics = sorted({key for v in values.values() for key in v})
        lines += [
            "",
            "| Native metric | " + " | ".join(entries) + " |",
            "|---|" + "---:|" * len(entries),
        ]
        for metric in metrics:
            cells = [
                f"{values[m][metric]:.6f}" if metric in values[m] else "—"
                for m in entries
            ]
            lines.append(
                "| " + metric.replace("|", "\\|") + " | " + " | ".join(cells) + " |"
            )
        pair = report["paired"].get(name)
        if pair:
            lines += [
                "",
                f"### {pair['comparison']} minus {pair['reference']}",
                "",
                f"Paired on {pair['source_prompts']} source prompts ({pair['results']} results). Intervals resample source prompts with all draws and description variants together.",
                f"Guidance: {pair['guidance']}. Differences include the selected CFG for each entry.",
                "",
                "| Metric | Difference | Prompt bootstrap 95% |",
                "|---|---:|---|",
            ]
            for metric, value in sorted(pair["delta"].items()):
                interval = pair["prompt_bootstrap_95_percent"].get(metric)
                cell = (
                    "—"
                    if interval is None
                    else f"[{interval[0]:.6f}, {interval[1]:.6f}]"
                )
                lines.append(f"| {metric} | {value:.6f} | {cell} |")
        lines += [
            "",
            "Judge recipe, missing keys, raw judgment paths, inference settings and measured resource costs are in `report.json`.",
            "",
        ]
    return "\n".join(lines)
