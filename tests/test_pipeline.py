"""Inference state flow and export round trips; these do not assess image quality."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from scripts.export_model import export_model
from src.models import Inko
from src.pipeline import MAX_IMAGE_SIZE, InkoPipeline


class ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x, t, text, pooled, mask):
        self.calls.append((x.clone(), t.clone(), pooled.clone()))
        return pooled[:, :1, None, None].expand_as(x)


class RecordingVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = {
            "z_dim": 16,
            "latents_mean": [0.25] * 16,
            "latents_std": [2.0] * 16,
        }

    def decode(self, x):
        self.decoded = x.clone()
        pixels = torch.nn.functional.interpolate(x[:, :3, 0], scale_factor=8)
        return SimpleNamespace(sample=pixels.unsqueeze(2))


@pytest.fixture
def pipe(monkeypatch):
    def encode(
        prompts, model, tokenizer, pooling, *, exit_layer, exit_mode, fast_slice
    ):
        values = torch.tensor([2.0 if text == "positive" else -1.0 for text in prompts])
        text = values[:, None, None].expand(-1, 2, 1024)
        return text, torch.ones(len(prompts), 2, dtype=torch.long), text.mean(1)

    monkeypatch.setattr("src.pipeline.encode_text", encode)
    return InkoPipeline(
        ConstantVelocity(),
        torch.nn.Linear(1, 1),
        SimpleNamespace(),
        RecordingVAE(),
        config={
            "height": 32,
            "width": 48,
            "num_inference_steps": 3,
            "guidance_scale": 1,
            "text_exit_layer": 2,
        },
    )


@pytest.mark.parametrize(
    "guidance,velocity,forwards", [(1, 2, 3), (0, -1, 6), (2, 5, 6)]
)
def test_guidance_integrates_velocity_then_denormalizes_for_vae(
    pipe, guidance, velocity, forwards
):
    progress = []
    output = pipe(
        "positive",
        guidance_scale=guidance,
        generator=torch.Generator().manual_seed(7),
        progress_callback=lambda step, total: progress.append((step, total)),
    )
    initial = pipe.transformer.calls[0][0]
    torch.testing.assert_close(
        pipe.vae.decoded[:, :, 0], (initial + velocity) * 2 + 0.25
    )
    assert len(pipe.transformer.calls) == forwards
    assert all(t.dtype == torch.float32 for _, t, _ in pipe.transformer.calls)
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert len(output.images) == 1
    assert output.images[0].size == (48, 32)
    assert output.images[0].mode == "RGB"


def test_per_image_generators_preserve_batch_order_and_global_rng(pipe):
    before = torch.random.get_rng_state().clone()
    pipe(
        ["positive", "negative"],
        generator=[torch.Generator().manual_seed(seed) for seed in (19, 23)],
    )
    batched = pipe.vae.decoded.clone()
    for index, (prompt, seed) in enumerate(
        zip(["positive", "negative"], (19, 23), strict=True)
    ):
        pipe(prompt, generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(pipe.vae.decoded[0], batched[index])
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": []},
        {"height": 31},
        {"width": 0},
        {"width": MAX_IMAGE_SIZE + 16},
        {"height": MAX_IMAGE_SIZE + 16},
        {"num_inference_steps": 0},
        {"num_inference_steps": 2.5},
        {"guidance_scale": float("nan")},
        {"guidance_scale": -1},
        {"negative_prompt": ["a", "b"]},
        {"generator": []},
    ],
)
def test_invalid_requests_fail_before_model_forward(pipe, kwargs):
    with pytest.raises(ValueError):
        pipe(**({"prompt": "positive"} | kwargs))
    assert pipe.transformer.calls == []


def test_to_keeps_latent_statistics_in_fp32(pipe):
    assert pipe.to("cpu", torch.bfloat16) is pipe
    assert pipe.dtype == torch.bfloat16
    assert pipe.vae_mean.dtype == pipe.vae_std.dtype == torch.float32
    assert all(
        next(model.parameters()).dtype == torch.bfloat16
        for model in (pipe.transformer, pipe.text_encoder, pipe.vae)
    )


def test_requested_dimensions_drive_both_block_types_and_isolate_requests(
    pipe, small_model, monkeypatch
):
    from src.models.dit_blocks import MSRoPE

    pipe.transformer = small_model().eval().requires_grad_(False)
    for module in pipe.transformer.modules():
        if isinstance(module, MSRoPE):
            module.reference_hw = (2, 3)
    calls = []
    forward = MSRoPE.forward

    def record(self, img_hw, txt_seq_len, device, *, time=None):
        image, text = forward(self, img_hw, txt_seq_len, device, time=time)
        calls.append((img_hw, image[0].clone(), text[0].clone()))
        return image, text

    monkeypatch.setattr(MSRoPE, "forward", record)
    pipe(
        "positive",
        height=32,
        width=32,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(7),
    )
    small = pipe.vae.decoded.clone()
    calls.clear()
    pipe(
        "positive",
        height=64,
        width=96,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert len(calls) == 2  # One double-stream and one single-stream block.
    for img_hw, image, text in calls:
        assert img_hw == (4, 6)  # Each patch spans 16 output pixels.
        torch.testing.assert_close(
            image,
            math.sqrt(1 + 0.1 * math.log(2))
            * torch.exp(1j * torch.tensor([-0.75, -0.0075, -1.25, -0.0125])),
        )
        torch.testing.assert_close(
            text,
            math.sqrt(1 + 0.1 * math.log(2))
            * torch.exp(1j * torch.tensor([3.0, 0.03, 3.0, 0.03])),
        )
    pipe(
        "positive",
        height=32,
        width=32,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(pipe.vae.decoded, small, rtol=0, atol=0)


def test_export_preserves_weights_and_loads_as_pipeline(tmp_path, monkeypatch, pipe):
    original = Inko(
        hidden_size=32,
        num_heads=2,
        double_stream_depth=1,
        single_stream_depth=1,
        mlp_ratio=2,
    )
    torch.save(original.state_dict(), tmp_path / "ema_weights.pt")
    (tmp_path / "transformer_config.json").write_text(json.dumps(original.get_config()))
    config_path = tmp_path / "inference.json"
    config_path.write_text(
        json.dumps(pipe.config | {"text_encoder": "text-local", "vae": "vae-local"})
    )
    output = tmp_path / "release"
    export_model(str(tmp_path / "ema_weights.pt"), output, config_path)
    monkeypatch.setattr(
        "src.pipeline.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: pipe.text_encoder,
    )
    monkeypatch.setattr(
        "src.pipeline.AutoTokenizer.from_pretrained", lambda *a, **kw: pipe.tokenizer
    )
    monkeypatch.setattr(
        "src.pipeline.AutoencoderKLQwenImage.from_pretrained", lambda *a, **kw: pipe.vae
    )
    restored = InkoPipeline.from_pretrained(output, local_files_only=True)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(
            restored.transformer.state_dict()[name], value, rtol=0, atol=0
        )
    assert not restored.transformer.training
    assert not any(p.requires_grad for p in restored.transformer.parameters())


def test_dynamic_rope_receives_current_solver_time(pipe, small_model, monkeypatch):
    from src.models.dit_blocks import MSRoPE

    pipe.transformer = small_model().eval().requires_grad_(False)
    for module in pipe.transformer.modules():
        if isinstance(module, MSRoPE):
            module.reference_hw = (2, 3)
    calls = []
    forward = MSRoPE.forward

    def record(self, img_hw, txt_seq_len, device, *, time=None):
        calls.append(time)
        return forward(self, img_hw, txt_seq_len, device, time=time)

    def sample(velocity, noise, **kwargs):
        for time in (0.25, 0.75):
            velocity(noise, torch.full((noise.shape[0],), time))
        return noise

    monkeypatch.setattr(MSRoPE, "forward", record)
    monkeypatch.setattr("src.pipeline.sample_ode", sample)
    pipe("positive", height=64, width=96)
    assert calls == [(0.25,), (0.25,), (0.75,), (0.75,)]
    calls.clear()
    pipe("positive", height=32, width=32)
    assert calls == [None] * 4


@pytest.mark.parametrize(
    "width,height",
    [(MAX_IMAGE_SIZE, 512), (512, MAX_IMAGE_SIZE), (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)],
)
def test_dimension_limit_is_accepted_on_either_axis(pipe, monkeypatch, width, height):
    monkeypatch.setattr("src.pipeline.sample_ode", lambda velocity, noise, **kw: noise)
    shapes = []

    def decode(latents, vae):
        shapes.append(tuple(latents.shape))
        return []

    monkeypatch.setattr("src.pipeline.decode_latents", decode)
    pipe("positive", width=width, height=height)
    assert shapes == [(1, 16, height // 8, width // 8)]
