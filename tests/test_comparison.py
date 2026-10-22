"""Coverage, native aggregation, paired joins and adding another model."""

import copy
import json

import pytest
from PIL import Image

from src.evaluation.benchmarks.common import prompt, write_json
from src.evaluation.records import append_jsonl, path_component
from src.evaluation.report import build, paired_difference


def publish(root, model, row, correct):
    images = root / "images" / model
    directory = images / path_component("geneval") / path_component("original")
    directory.mkdir(parents=True, exist_ok=True)
    samples = []
    for draw, seed in enumerate(row["seeds"]):
        path = directory / f"{draw}.png"
        Image.new("RGB", (3, 2), "red").save(path)
        samples.append(
            {
                "model": model,
                **{k: row[k] for k in ("benchmark", "split", "prompt_id", "variant")},
                "draw": draw,
                "seed": seed,
                "width": 3,
                "height": 2,
                "image_path": str(path.relative_to(root / "images")),
            }
        )
    append_jsonl(directory / "samples.jsonl", samples)
    write_json(
        images / "settings.json",
        {
            "model": {
                "name": model,
                "checkpoint": {"asset": model, "weights": "ema"},
                "steps": 7,
                "guidance": 1.0,
            },
            "weights": f"/{model}.pt",
            "sampling": {"precision": "float32"},
        },
    )
    score = root / "scores/geneval" / model
    write_json(
        score / "request.json",
        {
            "model": model,
            "benchmark": "geneval",
            "settings": {},
            "assets": {"detector": "official-detector"},
        },
    )
    append_jsonl(
        score / "scores.jsonl",
        [
            {
                "prompt": row,
                "samples": samples,
                "dimension": "detector",
                "native": [{"tag": "counting", "correct": v} for v in correct],
                "seconds": 0.1,
            }
        ],
    )


def report(root, catalog, models, pending=()):
    return build(
        catalog,
        root / "images",
        root / "scores",
        models,
        list(pending),
        {"reference": "base", "comparison": "ema", "bootstrap_samples": 20, "seed": 8},
        root / "report",
    )


def test_pending_then_append_preserves_previous_results(tmp_path):
    row = prompt("geneval", "original", "2", "two cats", {}, [11, 13])
    publish(tmp_path, "base", row, [True, False])
    score = tmp_path / "scores/geneval/base/scores.jsonl"
    original = score.read_bytes(), score.stat().st_mtime_ns
    before = report(tmp_path, [row], ["base"], ["ema"])
    assert before["pending"] == ["ema"]
    assert before["benchmarks"]["geneval"]["models"]["base"]["scores"]["overall"] == 0.5
    publish(tmp_path, "ema", row, [True, True])
    after = report(tmp_path, [row], ["base", "ema"])
    assert (
        after["benchmarks"]["geneval"]["models"]["base"]
        == before["benchmarks"]["geneval"]["models"]["base"]
    )
    assert after["paired"]["geneval"]["delta"]["overall"] == 0.5
    assert after["paired"]["geneval"]["prompt_bootstrap_95_percent"]["overall"] == [
        0.5,
        0.5,
    ]
    assert (score.read_bytes(), score.stat().st_mtime_ns) == original


def test_missing_requested_outputs_are_incomplete(tmp_path):
    row = prompt("geneval", "original", "2", "two cats", {}, [11, 13])
    publish(tmp_path, "base", row, [True, False])
    report_data = report(tmp_path, [row], ["base", "missing"], ["ema"])
    missing = report_data["benchmarks"]["geneval"]["models"]["missing"]
    assert missing["status"] == "incomplete"
    assert missing["expected_images"] == 2
    assert missing["evaluated_images"] == 0
    assert missing["missing_results"] == [
        ("geneval", "original", "2", "default", "detector")
    ]


def test_paired_entries_may_use_different_selected_guidance(tmp_path):
    row = prompt("geneval", "original", "2", "two cats", {}, [11, 13])
    publish(tmp_path, "base", row, [True, False])
    publish(tmp_path, "ema", row, [True, True])
    path = tmp_path / "images/ema/settings.json"
    recipe = json.loads(path.read_text())
    recipe["model"]["guidance"] = 3.5
    write_json(path, recipe)
    result = report(tmp_path, [row], ["base", "ema"])
    assert result["paired"]["geneval"]["guidance"] == {"base": 1.0, "ema": 3.5}


def test_catalog_slice_uses_its_judgments_from_a_larger_score_collection(tmp_path):
    first = prompt("geneval", "original", "2", "two cats", {}, [11])
    second = prompt("geneval", "original", "3", "two dogs", {}, [13])
    publish(tmp_path, "base", first, [False])
    publish(tmp_path, "base", second, [True])
    write_json(
        tmp_path / "scores/geneval/base/summary.json",
        {
            "expected": 2,
            "evaluated": 2,
            "scores": {"categories": {"counting": 0.5}, "overall": 0.5},
        },
    )
    result = report(tmp_path, [second], ["base"])
    entry = result["benchmarks"]["geneval"]["models"]["base"]
    assert entry["status"] == "complete"
    assert entry["expected_images"] == entry["evaluated_images"] == 1
    assert entry["scores"]["overall"] == 1.0


def test_published_aggregate_tolerates_float_summation_noise(tmp_path):
    row = prompt("geneval", "original", "2", "two cats", {}, [11])
    publish(tmp_path, "base", row, [True])
    write_json(
        tmp_path / "scores/geneval/base/summary.json",
        {
            "expected": 1,
            "evaluated": 1,
            "scores": {"categories": {"counting": 1.0 + 1e-15}, "overall": 1.0},
        },
    )
    result = report(tmp_path, [row], ["base"])
    assert result["benchmarks"]["geneval"]["models"]["base"]["status"] == "complete"


def test_published_aggregate_mismatch_fails(tmp_path):
    row = prompt("geneval", "original", "2", "two cats", {}, [11])
    publish(tmp_path, "base", row, [True])
    write_json(
        tmp_path / "scores/geneval/base/summary.json",
        {
            "expected": 1,
            "evaluated": 1,
            "scores": {"categories": {"counting": 0.5}, "overall": 0.5},
        },
    )
    with pytest.raises(ValueError, match="published aggregate differs"):
        report(tmp_path, [row], ["base"])


@pytest.mark.parametrize(
    "published,native", [(True, 1.0), (False, 0), (1, True), (0.0, False)]
)
def test_score_comparison_rejects_boolean_numeric_mismatch(published, native):
    from src.evaluation.report import scores_match

    assert not scores_match({"overall": published}, {"overall": native})


@pytest.mark.parametrize("change", ["judge", "inference", "prompt", "seed"])
def test_incompatible_comparisons_fail(tmp_path, change):
    row = prompt("geneval", "original", "2", "two cats", {}, [11, 13])
    publish(tmp_path, "base", row, [True, False])
    publish(tmp_path, "ema", row, [True, True])
    if change == "judge":
        path = tmp_path / "scores/geneval/ema/request.json"
        record = json.loads(path.read_text())
        record["assets"]["detector"] = "another-detector"
    elif change == "inference":
        path = tmp_path / "images/ema/settings.json"
        record = json.loads(path.read_text())
        record["model"]["steps"] = 8
    else:
        path = tmp_path / "scores/geneval/ema/scores.jsonl"
        record = json.loads(path.read_text())
        if change == "prompt":
            record["prompt"]["text"] = "three cats"
        else:
            record["samples"][0]["seed"] = 99
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        report(tmp_path, [row], ["base", "ema"])


def test_bootstrap_keeps_variants_and_dimensions_with_source(monkeypatch):
    from src.evaluation import report as module

    before = []
    for source in ("a", "b"):
        for variant in ("short", "long"):
            for dimension in ("vlm", "ocr"):
                before.append(
                    {
                        "prompt": prompt(
                            "tiif", "testmini", source, "text", {}, [3], variant
                        ),
                        "dimension": dimension,
                        "native": 1,
                    }
                )
    after = copy.deepcopy(before)
    for row in after:
        row["native"] = 4

    def summarize(rows, assets):
        for source in ("a", "b"):
            group = [r for r in rows if r["prompt"]["prompt_id"] == source]
            assert len(group) % 4 == 0
            assert {r["prompt"]["variant"] for r in group} in (set(), {"short", "long"})
            assert {r["dimension"] for r in group} in (set(), {"vlm", "ocr"})
        return {"score": sum(r["native"] for r in rows) / len(rows)}

    monkeypatch.setattr(
        module,
        "adapter",
        lambda name: type("Adapter", (), {"summarize": staticmethod(summarize)}),
    )
    result = paired_difference("tiif", before, after, {}, 30, 9)
    assert result["source_prompts"] == 2
    assert result["results"] == 8
    assert result["delta"] == {"score": 3}
    assert result["prompt_bootstrap_95_percent"] == {"score": [3, 3]}
