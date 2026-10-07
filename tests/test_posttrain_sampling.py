import pytest
import torch

from src.posttrain.sampling import sample, sample_noise, time_grid


@pytest.mark.parametrize("rule", ["euler", "renoise"])
def test_oracle_clean_prediction_reaches_target(rule):
    target = torch.tensor(3.0)

    def oracle(x, t):
        return (target - x) / (1 - t[:, None, None, None])

    result = sample(oracle, [7, 19], (1, 2, 2), rule=rule, grid=(0, 0.2, 0.7, 1))
    torch.testing.assert_close(result.terminal, torch.full((2, 1, 2, 2), 3.0))
    assert not result.terminal.requires_grad


def test_noise_assignment_survives_reordering_and_sharding():
    velocity = lambda x, t: x * 0.125
    kwargs = {"shape": (1, 2, 2), "rule": "renoise", "grid": (0, 0.4, 1)}
    batch = sample(velocity, [11, 23], **kwargs)
    reordered = sample(velocity, [23, 11], **kwargs)
    single = sample(velocity, [11], **kwargs)
    assert torch.equal(batch.terminal[0], reordered.terminal[1])
    assert torch.equal(batch.terminal[:1], single.terminal)
    a = sample_noise([11], (1, 2, 2), stream="transition", index=1)
    b = sample_noise([11], (1, 2, 2), stream="transition", index=2)
    assert not torch.equal(a, b)


def test_selected_input_recomputes_with_only_one_step_gradient():
    theta = torch.tensor(2.0, requires_grad=True)
    result = sample(
        lambda x, t: theta.expand_as(x),
        [3],
        (1, 1, 1),
        rule="renoise",
        grid=(0, 0.25, 1),
        selected_indices=[1],
    )
    assert not result.selected_input.requires_grad
    assert not result.terminal.requires_grad
    selected_time = result.selected_times.item()
    clean = result.selected_input + (1 - selected_time) * theta
    torch.testing.assert_close(clean.detach(), result.terminal)
    clean.sum().backward()
    assert theta.grad.item() == 0.75


def test_shift_has_explicit_endpoints_and_is_applied_once():
    assert time_grid(2, 3) == (0.0, 0.25, 1.0)
    with pytest.raises(ValueError, match="increase"):
        sample(lambda x, t: x, [1], (1, 1, 1), rule="renoise", grid=(0, 0.5, 0.4, 1))


def test_inko_model_boundary_accepts_fp32_time_with_bf16_weights():
    from src.models.inko import Inko
    from src.posttrain.models import Generator

    worker = Generator.__new__(Generator)
    worker.device = torch.device("cpu")
    worker.dtype = torch.bfloat16
    worker.model = (
        Inko(
            hidden_size=64,
            num_heads=4,
            double_stream_depth=0,
            single_stream_depth=1,
            mlp_ratio=2,
        )
        .to(torch.bfloat16)
        .eval()
    )
    conditioning = (
        torch.zeros(1, 3, 1024, dtype=torch.bfloat16),
        torch.zeros(1, 1024, dtype=torch.bfloat16),
        torch.ones(1, 3, dtype=torch.long),
    )
    # Exercise the real model's FP32 sinusoidal features and BF16 projections.
    output = worker.predict_velocity(
        torch.zeros(1, 16, 4, 4), torch.tensor([0.25]), conditioning
    )
    assert output.shape == (1, 16, 4, 4) and output.dtype == torch.bfloat16
