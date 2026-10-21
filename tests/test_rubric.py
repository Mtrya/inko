"""Judge parsing and order normalization; model judgment quality needs images."""

import re

import pytest

from src.posttrain.scoring.backends.rubric import (
    DRAW_BASIS,
    RubricJudge,
    merge_orders,
    parse_lines,
)

FORWARD = """content_alignment|A|A has the requested tree.
form_structure|B|B has coherent branches.
aesthetics|tie|Neither composition is preferable.
style_medium|A|A has more visible brush strokes.
artifacts|B|B has fewer defects."""
REVERSE = """content_alignment|B|B has the requested tree.
form_structure|A|A has coherent branches.
aesthetics|A|A has a clearer composition.
style_medium|A|A has more visible brush strokes.
artifacts|B|B has fewer defects."""


def test_merge_maps_both_orders_to_samples_before_tying_disagreements():
    values, orders = merge_orders(
        parse_lines(FORWARD), parse_lines(REVERSE), ("left", "right")
    )
    assert values == [
        {
            "rubric.content_alignment": 1.0,
            "rubric.form_structure": 0.0,
            "rubric.aesthetics": 0.5,
            "rubric.style_medium": 0.5,
            "rubric.artifacts": 0.5,
        },
        {
            "rubric.content_alignment": 0.0,
            "rubric.form_structure": 1.0,
            "rubric.aesthetics": 0.5,
            "rubric.style_medium": 0.5,
            "rubric.artifacts": 0.5,
        },
    ]
    assert orders[0]["content_alignment"]["winner"] == "left"
    assert orders[1]["content_alignment"]["winner"] == "left"
    assert orders[1]["content_alignment"]["basis"] == "B has the requested tree."


@pytest.mark.parametrize(
    "text",
    [
        FORWARD + "\ncontent_alignment|A|Duplicate.",
        FORWARD.replace("artifacts|B|B has fewer defects.", ""),
        FORWARD.replace(
            "artifacts|B|B has fewer defects.", "artifacts|invalid|Reason."
        ),
        FORWARD.replace("artifacts|B|B has fewer defects.", "artifacts|B|"),
        FORWARD.replace("artifacts|B|", "artifacts|"),
        FORWARD + "\nCorrection; final answer:\n" + REVERSE,
    ],
)
def test_invalid_judgments_are_errors(text):
    with pytest.raises(ValueError):
        parse_lines(text)


def test_judge_constrains_final_format_and_keeps_thinking(monkeypatch):
    def completion(endpoint, route, timeout, payload):
        pattern = payload["structured_outputs"]["regex"]
        assert re.fullmatch(pattern, "\n\n" + FORWARD + "\n")
        assert re.fullmatch(pattern, FORWARD.replace("tree", "树木"))
        assert not re.fullmatch(pattern, FORWARD.replace("artifacts|B|", "artifacts|"))
        assert not re.fullmatch(pattern, FORWARD + "\n" + REVERSE)
        assert payload["chat_template_kwargs"]["enable_thinking"]
        return {"choices": [{"finish_reason": "stop", "message": {"content": FORWARD}}]}

    monkeypatch.setattr("src.posttrain.scoring.client.http_json", completion)
    rubric = {
        category: [{"text": "Compare visible details.", "weight": "high"}]
        for category in (
            "content_alignment",
            "form_structure",
            "aesthetics",
            "style_medium",
            "artifacts",
        )
    }
    result = RubricJudge("http://fixture", "judge", timeout=1, max_tokens=100).judge(
        "A tree", rubric, ["candidate-a", "candidate-b"]
    )
    assert result["decisions"]["content_alignment"]["verdict"] == "A"


@pytest.mark.parametrize(
    "suffix", ["\nUnfinished correction", "\nartifacts|A|A has fewer defects."]
)
def test_incomplete_final_block_does_not_reuse_an_earlier_draft(suffix):
    with pytest.raises(ValueError):
        parse_lines(FORWARD + suffix)


RUBRIC = {
    category: [{"text": "Compare visible details.", "weight": "high"}]
    for category in (
        "content_alignment",
        "form_structure",
        "aesthetics",
        "style_medium",
        "artifacts",
    )
}


def test_judge_without_five_verdict_lines_is_a_draw(monkeypatch):
    def completion(endpoint, route, timeout, payload):
        return {
            "choices": [{"finish_reason": "length", "message": {"content": ""}}],
            "usage": {"completion_tokens": payload["max_tokens"]},
        }

    monkeypatch.setattr("src.posttrain.scoring.client.http_json", completion)
    result = RubricJudge("http://fixture", "judge", timeout=1, max_tokens=100).judge(
        "A tree", RUBRIC, ["candidate-a", "candidate-b"]
    )
    assert {category: d["verdict"] for category, d in result["decisions"].items()} == {
        category: "tie"
        for category in (
            "content_alignment",
            "form_structure",
            "aesthetics",
            "style_medium",
            "artifacts",
        )
    }
    assert result["decisions"]["artifacts"]["basis"] == DRAW_BASIS
    assert result["usage"] == {"completion_tokens": 100}
    values, _ = merge_orders(result["decisions"], parse_lines(REVERSE), ("l", "r"))
    assert values == [
        {f"rubric.{category}": 0.5 for category in result["decisions"]},
        {f"rubric.{category}": 0.5 for category in result["decisions"]},
    ]


def test_judge_with_unparseable_answer_is_a_draw(monkeypatch):
    def completion(endpoint, route, timeout, payload):
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": "no lines"}}]
        }

    monkeypatch.setattr("src.posttrain.scoring.client.http_json", completion)
    result = RubricJudge("http://fixture", "judge", timeout=1, max_tokens=100).judge(
        "A tree", RUBRIC, ["candidate-a", "candidate-b"]
    )
    assert all(
        decision == {"verdict": "tie", "basis": DRAW_BASIS}
        for decision in result["decisions"].values()
    )
