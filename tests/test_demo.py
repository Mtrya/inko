"""Demo protocol, image records, and UI state flow; no image-quality assertions."""

import importlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from demo import generation as logic


@pytest.fixture
def tokenizer():
    return Mock(
        side_effect=lambda text, **kwargs: {
            "input_ids": [0] * (3000 if "OVER_BUDGET" in text else 80)
        }
    )


@pytest.fixture
def canvases(monkeypatch):
    # These are decision inputs. The mechanism must also work with other presets.
    monkeypatch.setattr(
        logic,
        "CANVASES",
        {
            "Standard": {"1:1": (512, 512), "4:3": (640, 480)},
            "Large": {"1:1": (1024, 1024), "4:3": (1280, 960)},
        },
    )


def response(content, finish="stop"):
    return Mock(
        json=lambda: {
            "choices": [{"message": {"content": content}, "finish_reason": finish}]
        }
    )


def test_disabled_enhancement_preserves_text_and_manual_canvas(
    monkeypatch, tokenizer, canvases
):
    post = Mock(side_effect=AssertionError("Unexpected network call"))
    monkeypatch.setattr(logic.requests, "post", post)
    result = logic.enhance_prompt(
        "  一幅画。  ", False, "Custom", "Auto", 480, 1920, tokenizer
    )
    assert result == {"prompt": "一幅画。", "width": 480, "height": 1920, "notice": ""}
    post.assert_not_called()


@pytest.mark.parametrize(
    "aspect,resolution,expected",
    [
        ("Auto", "Auto", (512, 512)),
        ("4:3", "Auto", (640, 480)),
        ("Auto", "Large", (1024, 1024)),
        ("4:3", "Large", (1280, 960)),
    ],
)
def test_auto_fallback_preserves_explicit_selections(
    canvases, aspect, resolution, expected
):
    assert logic.fallback_canvas(aspect, resolution, 480, 480) == expected


def test_canvas_restrictions_reach_provider_and_unicode_is_retained(
    monkeypatch, tokenizer, canvases
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    post = Mock(
        return_value=response(
            json.dumps({"prompt": "工笔牡丹，绢本设色。", "width": 640, "height": 480})
        )
    )
    monkeypatch.setattr(logic.requests, "post", post)
    result = logic.enhance_prompt(
        "牡丹，题字“春”。", True, "4:3", "Standard", 512, 512, tokenizer
    )
    assert result == {
        "prompt": "工笔牡丹，绢本设色。",
        "width": 640,
        "height": 480,
        "notice": "",
    }
    request = post.call_args.kwargs
    assert request["headers"] == {"Authorization": "Bearer test-secret"}
    assert request["json"]["response_format"] == {"type": "json_object"}
    supplied = json.loads(request["json"]["messages"][-1]["content"])
    assert supplied["description"] == "牡丹，题字“春”。"
    assert supplied["canvas_choices"] == [
        {"width": 640, "height": 480, "resolution": "Standard", "aspect_ratio": "4:3"}
    ]
    assert tokenizer.call_args.kwargs["truncation"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "missing_key",
        "empty",
        "truncated",
        "bad_json",
        "extra_field",
        "wrong_canvas",
        "float_dimension",
        "over_budget",
        "nonobject",
    ],
)
def test_failed_enhancement_preserves_original_and_explicit_canvas(
    monkeypatch, tokenizer, canvases, failure
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    content = {"prompt": "A rewritten garden", "width": 640, "height": 480}
    if failure == "extra_field":
        content["reason"] = "unrequested"
    if failure == "wrong_canvas":
        content["width"], content["height"] = 512, 512
    if failure == "float_dimension":
        content["width"] = 640.0
    if failure == "over_budget":
        content["prompt"] = "OVER_BUDGET"
    encoded = json.dumps(content)
    if failure == "empty":
        encoded = ""
    if failure == "bad_json":
        encoded = "```json\n{}\n```"
    if failure == "nonobject":
        encoded = "[]"
    post = Mock(
        return_value=response(encoded, "length" if failure == "truncated" else "stop")
    )
    if failure == "timeout":
        post.side_effect = requests.Timeout("private service detail")
    if failure == "missing_key":
        monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.setattr(logic.requests, "post", post)
    result = logic.enhance_prompt(
        "Original garden", True, "4:3", "Standard", 512, 512, tokenizer
    )
    assert (result["prompt"], result["width"], result["height"]) == (
        "Original garden",
        640,
        480,
    )
    assert "original prompt" in result["notice"]
    assert "private service detail" not in result["notice"]
    if failure == "missing_key":
        post.assert_not_called()


def test_over_budget_original_is_not_sent_or_silently_truncated(
    monkeypatch, tokenizer, canvases
):
    post = Mock()
    monkeypatch.setattr(logic.requests, "post", post)
    with pytest.raises(ValueError, match="token limit"):
        logic.enhance_prompt("OVER_BUDGET", True, "Auto", "Auto", 512, 512, tokenizer)
    post.assert_not_called()


@pytest.mark.parametrize(
    "width,height",
    [(479, 640), (481, 640), (1936, 640), (640, float("nan")), (True, 640)],
)
def test_invalid_dimensions_fail_at_request_boundary(width, height):
    with pytest.raises(ValueError):
        logic.validate_dimensions(width, height)


def test_png_keeps_unicode_recipe_and_original_pixels(tmp_path):
    image = Image.new("RGB", (16, 32), (25, 80, 120))
    recipe = {
        "prompt": "题字“春” — blue vase",
        "width": 16,
        "height": 32,
        "seed": 17,
        "steps": 13,
        "guidance": 2.75,
        "model": "owner/model",
        "sampler": "Euler",
        "notice": "",
    }
    path = logic.save_image(image, recipe, tmp_path)
    with Image.open(path) as saved:
        assert saved.size == (16, 32)
        assert saved.getpixel((8, 16)) == (25, 80, 120)
        assert json.loads(saved.info["inko"]) == recipe
    assert logic.read_record(path, tmp_path) == recipe | {
        "image": path,
        "kind": "Generated",
    }


def test_history_fifo_does_not_mutate_old_state_or_delete_another_session(tmp_path):
    other_session = tmp_path / "other.png"
    other_session.touch()
    directory = tmp_path / "session"
    directory.mkdir()
    records = []
    for name in ("first", "second", "third"):
        path = directory / f"{name}.png"
        path.touch()
        records.append({"image": str(path), "prompt": name})
    old = records[:2]
    updated = logic.append_history(old, records[2], directory, limit=2)
    assert [record["prompt"] for record in updated] == ["second", "third"]
    assert [record["prompt"] for record in old] == ["first", "second"]
    assert not (directory / "first.png").exists()
    assert (directory / "second.png").exists()
    assert other_session.exists()
    logic.delete_results([{"image": str(other_session)}], directory)
    assert other_session.exists()


def test_result_reader_rejects_files_outside_generated_directory(tmp_path):
    with pytest.raises(ValueError, match="generated in this session"):
        logic.read_record(tmp_path / "outside.png", tmp_path / "results")


@pytest.fixture
def app(monkeypatch, tokenizer, tmp_path):
    pytest.importorskip("gradio", minversion="6.13")
    pytest.importorskip("spaces")
    monkeypatch.setenv("INKO_MODEL_ID", "local-export")
    pipe = Mock(
        config={"num_inference_steps": 13, "guidance_scale": 3.25}, tokenizer=tokenizer
    )
    pipe.to.return_value = pipe
    pipe.side_effect = lambda prompt, **kwargs: SimpleNamespace(
        images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")]
    )
    monkeypatch.setattr(
        "src.pipeline.InkoPipeline.from_pretrained", lambda *args, **kwargs: pipe
    )
    module = importlib.import_module("demo.app")
    monkeypatch.setattr(module, "pipe", pipe)
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    return module


def test_demo_requires_explicit_model_source_before_loading(monkeypatch):
    pytest.importorskip("gradio", minversion="6.13")
    pytest.importorskip("spaces")
    monkeypatch.delenv("INKO_MODEL_ID", raising=False)
    load = Mock(side_effect=AssertionError("Unexpected model load"))
    monkeypatch.setattr("src.pipeline.InkoPipeline.from_pretrained", load)
    with pytest.raises(RuntimeError, match="INKO_MODEL_ID"):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "demo/app.py"))
    load.assert_not_called()


def test_example_restores_saved_sampling_instead_of_loaded_defaults(app, monkeypatch):
    item = {
        "prompt": "A lake",
        "width": 640,
        "height": 896,
        "seed": 123,
        "steps": 17,
        "guidance": 2.5,
        "model": "example/model",
        "file": "lake.png",
    }
    monkeypatch.setattr(app, "SHOWCASE", [item])
    updates = app.select_example(False, [], SimpleNamespace(index=0))
    assert updates[app.prompt] == "A lake"
    assert updates[app.width] == 640 and updates[app.height] == 896
    assert updates[app.seed] == 123
    assert updates[app.steps] == 17 and updates[app.guidance] == 2.5
    assert updates[app.enhance] is False and updates[app.randomize_seed] is False
    assert updates[app.selected]["kind"] == "Example"
    assert app.history not in updates


def test_preparation_and_failure_leave_completed_result_untouched(app):
    changes = list(
        app.prepare_request(
            "A garden", False, "Custom", "Auto", 480, 640, 17, False, 13, 2.75
        )
    )
    for update in [*changes, app.failed_request({"prompt": "previous"})]:
        for component in (
            app.output,
            app.used_prompt,
            app.selected,
            app.history,
            app.recipe,
            app.result_notice,
        ):
            assert component not in update
    assert changes[-1][app.pending] == {
        "prompt": "A garden",
        "width": 480,
        "height": 640,
        "seed": 17,
        "steps": 13,
        "guidance": 2.75,
        "notice": "",
    }


def test_generation_and_commit_keep_recipe_atomic_and_original_input_intact(app):
    request = {
        "prompt": "A garden",
        "width": 480,
        "height": 640,
        "seed": 17,
        "steps": 13,
        "guidance": 2.75,
        "notice": "Enhancement unavailable; used your original prompt.",
    }
    path = app.generate(request)
    kwargs = app.pipe.call_args.kwargs
    assert kwargs["width"] == 480 and kwargs["height"] == 640
    assert kwargs["num_inference_steps"] == 13 and kwargs["guidance_scale"] == 2.75
    assert kwargs["generator"].initial_seed() == 17
    changes = app.commit_result(path, [])
    assert changes[app.output] == path
    assert changes[app.used_prompt] == "A garden"
    assert changes[app.selected]["seed"] == 17
    assert changes[app.history] == [changes[app.selected]]
    assert (
        changes[app.result_notice]["value"]
        == "Enhancement unavailable; used your original prompt."
    )
    assert changes[app.result_notice]["visible"] is True
    assert "value" not in changes[app.prompt]
    assert "value" not in changes[app.enhance]
    assert "value" not in changes[app.randomize_seed]
    for component, expected in ((app.width, 480), (app.height, 640), (app.seed, 17)):
        assert changes[component]["value"] == expected
        assert changes[component]["interactive"] is True


def test_prompt_reuse_copies_canvas_without_changing_sampling_controls(app):
    changes = app.reuse_prompt(
        {"prompt": "Use this exact prompt", "width": 480, "height": 1920}, False
    )
    assert changes[app.prompt] == "Use this exact prompt"
    assert changes[app.enhance] is False
    assert changes[app.aspect] == "Custom"
    assert changes[app.width] == 480 and changes[app.height] == 1920
    assert app.seed not in changes and app.randomize_seed not in changes


def test_history_selection_uses_selected_record_and_freezes_recipe(app):
    records = [
        {
            "image": "/tmp/a.png",
            "prompt": "Older painting",
            "width": 480,
            "height": 640,
            "seed": 91,
            "steps": 15,
            "guidance": 1.5,
            "kind": "Generated",
            "notice": "",
        },
        {
            "image": "/tmp/b.png",
            "prompt": "Newer painting",
            "width": 640,
            "height": 480,
            "seed": 92,
            "steps": 16,
            "guidance": 2.5,
            "kind": "Generated",
            "notice": "",
        },
    ]
    changes = app.select_history(False, records, SimpleNamespace(index=0))
    assert changes[app.selected] == records[0]
    assert changes[app.output] == "/tmp/a.png"
    assert changes[app.prompt] == "Older painting"
    assert changes[app.seed] == 91 and changes[app.steps] == 15
    assert changes[app.enhance] is False and changes[app.randomize_seed] is False


def test_invalid_generation_does_not_call_pipeline(app):
    with pytest.raises(app.gr.Error):
        app.generate(
            {
                "prompt": "Garden",
                "width": 481,
                "height": 640,
                "seed": 17,
                "steps": 20,
                "guidance": 1,
            }
        )
    app.pipe.assert_not_called()


def history_record(app, name, seed):
    path = app.RESULT_DIR / f"{name}.png"
    path.touch()
    return {
        "image": str(path),
        "prompt": name,
        "width": 480,
        "height": 640,
        "seed": seed,
        "steps": 15,
        "guidance": 2.5,
        "kind": "Generated",
    }


def test_delete_selected_removes_only_that_file_and_restores_adjacent_recipe(app):
    records = [
        history_record(app, name, seed) for seed, name in enumerate(("A", "B", "C"))
    ]
    changes = app.delete_selected(records[1], False, records)
    assert changes[app.history] == [records[0], records[2]]
    assert changes[app.selected] == records[2]
    assert changes[app.output] == records[2]["image"]
    assert changes[app.prompt] == "C"
    assert changes[app.seed] == 2
    assert changes[app.enhance] is False and changes[app.randomize_seed] is False
    assert not Path(records[1]["image"]).exists()
    assert Path(records[0]["image"]).exists() and Path(records[2]["image"]).exists()
    assert len(records) == 3


def test_delete_last_image_clears_result_and_keeps_input_for_regeneration(app):
    record = history_record(app, "Last painting", 31)
    changes = app.delete_selected(record, False, [record])
    assert changes[app.history] == []
    assert changes[app.selected] is None and changes[app.output] is None
    assert changes[app.used_prompt] == ""
    assert changes[app.recipe]["visible"] is False
    assert changes[app.result_notice]["visible"] is False
    assert changes[app.reuse_button]["interactive"] is False
    assert changes[app.delete_button]["visible"] is False
    assert app.prompt not in changes and app.seed not in changes
    assert not Path(record["image"]).exists()


def test_deletion_rejects_busy_requests_and_images_outside_history(app):
    record = history_record(app, "Kept painting", 32)
    example = history_record(app, "Example", 33) | {"kind": "Example"}
    with pytest.raises(app.gr.Error, match="Wait for"):
        app.delete_selected(record, True, [record])
    with pytest.raises(app.gr.Error, match="generation history"):
        app.delete_selected(example, False, [record])
    assert Path(record["image"]).exists() and Path(example["image"]).exists()
