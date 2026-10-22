"""Demo request routing without a GPU or external service."""

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

pytest.importorskip("gradio", minversion="6.13")
pytest.importorskip("spaces")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("INKO_MODEL_ID", "local-export")
    pipe = Mock(config={"num_inference_steps": 20, "guidance_scale": 1})
    pipe.to.return_value = pipe
    monkeypatch.setattr(
        "src.pipeline.InkoPipeline.from_pretrained", lambda *a, **k: pipe
    )
    module = importlib.import_module("demo.app")
    monkeypatch.setattr(module, "pipe", pipe)
    return module


def test_demo_requires_explicit_model_source_before_loading(monkeypatch):
    monkeypatch.delenv("INKO_MODEL_ID", raising=False)
    load = Mock(side_effect=AssertionError("Unexpected model load"))
    monkeypatch.setattr("src.pipeline.InkoPipeline.from_pretrained", load)
    with pytest.raises(RuntimeError, match="INKO_MODEL_ID"):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "demo/app.py"))
    load.assert_not_called()


def test_enhancement_off_does_not_contact_deepseek(app, monkeypatch):
    post = Mock(side_effect=AssertionError("Unexpected network call"))
    monkeypatch.setattr(app.requests, "post", post)
    assert app.prepare_prompt("  An ink landscape.  ", False) == "An ink landscape."
    post.assert_not_called()


def test_gallery_prompt_uses_loaded_model_sampling_defaults(app, monkeypatch):
    monkeypatch.setattr(
        app.pipe, "config", {"num_inference_steps": 13, "guidance_scale": 3.25}
    )
    monkeypatch.setattr(
        app,
        "SHOWCASE",
        {
            "width": 640,
            "height": 896,
            "examples": [{"prompt": "A lake", "seed": 123, "name": "lake"}],
        },
    )
    selected = app.select_example(SimpleNamespace(index=0))
    assert selected[:9] == ("A lake", 640, 896, 123, False, 13, 3.25, False, "A lake")


def test_enhancement_uses_server_secret_and_returns_prompt(app, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    post = Mock(
        return_value=Mock(
            json=lambda: {
                "choices": [
                    {
                        "message": {"content": "  Mist over ink mountains.  "},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(app.requests, "post", post)
    assert app.prepare_prompt("An ink landscape.", True) == "Mist over ink mountains."
    request = post.call_args.kwargs
    assert request["headers"]["Authorization"] == "Bearer test-secret"
    assert request["json"]["messages"][-1] == {
        "role": "user",
        "content": "An ink landscape.",
    }
    assert request["timeout"][1] > 0


@pytest.mark.parametrize("failure", ["timeout", "empty", "truncated", "missing_key"])
def test_enhancement_failure_warns_and_keeps_original(app, monkeypatch, failure):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    response = {
        "choices": [
            {
                "message": {
                    "content": None if failure == "empty" else "Partial rewrite"
                },
                "finish_reason": "length" if failure == "truncated" else "stop",
            }
        ]
    }
    post = Mock(return_value=Mock(json=lambda: response))
    if failure == "timeout":
        post.side_effect = requests.Timeout("private service detail")
    if failure == "missing_key":
        monkeypatch.delenv("DEEPSEEK_API_KEY")
    warning = Mock()
    monkeypatch.setattr(app.requests, "post", post)
    monkeypatch.setattr(app.gr, "Warning", warning)
    assert app.prepare_prompt("Original idea", True) == "Original idea"
    warning.assert_called_once()
    assert "private service detail" not in warning.call_args.args[0]


@pytest.mark.parametrize("randomize,expected_seed", [(True, 1234567), (False, 42)])
def test_generation_forwards_dimensions_and_resolved_seed(
    app, monkeypatch, randomize, expected_seed
):
    gpu = Mock(return_value="generated image")
    monkeypatch.setattr(app, "run_pipeline", gpu)
    monkeypatch.setattr(app.secrets, "randbelow", lambda n: 1234567)
    assert app.generate("A garden", 480, 1920, 42, randomize, 20, 0) == (
        "generated image",
        expected_seed,
    )
    gpu.assert_called_once_with("A garden", 480, 1920, expected_seed, 20, 0)


def test_generation_api_keeps_existing_inputs_without_rope_choice(app):
    endpoint = app.demo.get_api_info()["named_endpoints"]["/generate"]
    assert [parameter["parameter_name"] for parameter in endpoint["parameters"]] == [
        "prompt",
        "width",
        "height",
        "seed",
        "randomize_seed",
        "steps",
        "guidance",
    ]
    assert not any(
        component.get("props", {}).get("label") == "RoPE scaling"
        for component in app.demo.config["components"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": " "},
        {"width": 479},
        {"width": 481},
        {"width": 1936},
        {"height": 1936},
        {"guidance": -0.1},
        {"guidance": 10.1},
        {"guidance": float("nan")},
        {"steps": 10.5},
        {"seed": -1},
    ],
)
def test_invalid_request_does_not_allocate_gpu(app, monkeypatch, changes):
    gpu = Mock()
    monkeypatch.setattr(app, "run_pipeline", gpu)
    values = {
        "prompt": "A garden",
        "width": 896,
        "height": 896,
        "seed": 42,
        "randomize_seed": False,
        "steps": 20,
        "guidance": 1,
    }
    with pytest.raises(app.gr.Error):
        app.generate(**(values | changes))
    gpu.assert_not_called()
