"""Independent NFT equations, grouped statistics and whole-round state flow."""

import math

import pytest
import torch

from src.posttrain.objectives.nft import losses, reference_losses
from src.posttrain.rewards import group_advantages
from src.posttrain.stages import refresh_decay
from src.posttrain.training_config import OldPolicy


@pytest.mark.parametrize(
    "parameterization,value,gradient",
    [
        ("clean_adaptive", 3.0, -1.5),
        ("velocity_mse", 4.48, -0.8),
    ],
)
def test_nft_detached_branch_value_gradient(parameterization, value, gradient):
    # z=.5; implicit v+=1.2,v-=.4. Clean errors=-.6,-1.2 give
    # independent detached denominators .6 and 1.2. C/beta=4.
    new = torch.tensor([[1.6]], requires_grad=True)
    old = torch.tensor([[0.8]], requires_grad=True)
    terminal = torch.tensor([[2.0]], requires_grad=True)
    noise = torch.zeros_like(terminal, requires_grad=True)
    times = torch.tensor([0.25], requires_grad=True)
    probability = torch.tensor([0.75], requires_grad=True)
    result = losses(
        new,
        old,
        terminal,
        noise,
        times,
        probability,
        parameterization=parameterization,
        beta=0.5,
        clip=2,
        epsilon=0.01,
    )
    result.sum().backward()
    assert float(result.detach()) == pytest.approx(value, abs=1e-6)
    assert float(new.grad) == pytest.approx(gradient, abs=1e-6)
    assert all(x.grad is None for x in (old, terminal, noise, times, probability))


def test_constant_group_anchors_after_student_movement_and_epsilon_floor():
    # p=.5 cancels at new=old, but new=.02,old=0 gives clean errors
    # +/-.01. Both h clamp to .1: C=2 -> loss .002, derivative .2.
    for value, loss, gradient in [(0.0, 0.0, 0.0), (0.02, 0.002, 0.2)]:
        new = torch.tensor([[value]], requires_grad=True)
        result = losses(
            new,
            torch.zeros_like(new),
            torch.zeros_like(new),
            torch.zeros_like(new),
            torch.tensor([0.5]),
            torch.tensor([0.5]),
            parameterization="clean_adaptive",
            beta=1,
            clip=2,
            epsilon=0.1,
        )
        result.sum().backward()
        assert float(result.detach()) == pytest.approx(loss, abs=1e-7)
        assert float(new.grad) == pytest.approx(gradient, abs=1e-6)


def test_beta_crossing_clean_target_and_reference_penalty():
    new = torch.tensor([[1.6]], requires_grad=True)
    result = losses(
        new,
        torch.tensor([[0.8]]),
        torch.tensor([[2.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([0.25]),
        torch.tensor([0.75]),
        parameterization="clean_adaptive",
        beta=2,
        clip=2,
        epsilon=0.01,
    )
    result.sum().backward()
    # Errors +.3 and -2.1: detached-normalized branch derivatives both +3.
    assert float(new.grad) == pytest.approx(3.0, abs=1e-6)
    value = torch.tensor([[1.0, 3.0]], requires_grad=True)
    reference = torch.tensor([[0.0, 1.0]], requires_grad=True)
    penalty = reference_losses(value, reference)
    penalty.sum().backward()
    torch.testing.assert_close(penalty, torch.tensor([2.5]), rtol=0, atol=0)
    torch.testing.assert_close(value.grad, torch.tensor([[1.0, 2.0]]), rtol=0, atol=0)
    assert reference.grad is None


def test_global_groups_clipping_constant_group_and_round_std():
    ids = ["a0", "b0", "a1", "b1", "c0", "c1"]
    values = [1.0, 5.0, 3.0, 9.0, 7.0, 7.0]
    groups = {"a0": "A", "a1": "A", "b0": "B", "b1": "B", "c0": "C", "c1": "C"}
    reward = {"sample_ids": ids, "combined": values}
    result = group_advantages(reward, groups, scale="group", clip=0.25, epsilon=1.0)
    assert result["advantages"] == pytest.approx(
        [-0.5, -2 / 3, 0.5, 2 / 3, 0, 0], abs=1e-7
    )
    assert result["probabilities"] == [0.0, 0.0, 1.0, 1.0, 0.5, 0.5]
    # Mean=16/3, population variance=65/9. Group numerators stay -1,-2,+1,+2.
    result = group_advantages(reward, groups, scale="round", clip=2, epsilon=1.0)
    denominator = math.sqrt(65 / 9) + 1
    assert result["advantages"] == pytest.approx(
        [-1 / denominator, -2 / denominator, 1 / denominator, 2 / denominator, 0, 0],
        abs=1e-7,
    )
    with pytest.raises(ValueError, match="complete"):
        group_advantages(reward, {"a0": "A"}, scale="group", clip=1, epsilon=0.1)
    with pytest.raises(ValueError, match="at least two"):
        group_advantages(
            {"sample_ids": ["a0"], "combined": [1.0]},
            {"a0": "A"},
            scale="group",
            clip=1,
            epsilon=0.1,
        )


def test_old_schedule_uses_previously_committed_rounds():
    assert [
        refresh_decay(OldPolicy("copy", 0.7, 0.2, 0.8, 2), r) for r in range(4)
    ] == [0, 0, 0, 0]
    assert [
        refresh_decay(OldPolicy("constant", 0.7, 0.2, 0.8, 2), r) for r in range(4)
    ] == [0.7, 0.7, 0.7, 0.7]
    assert [
        refresh_decay(OldPolicy("linear_ramp", 0.7, 0.2, 0.8, 2), r) for r in range(4)
    ] == pytest.approx([0.2, 0.5, 0.8, 0.8])


def test_fp32_adaptive_loss_and_gradient_against_independent_double_oracle():
    # Two dimensions have unequal signs/magnitudes, so a shared h or a derivative
    # through h changes this independently evaluated scalar/closed derivative.
    targets = [[2.0, -1.0], [0.5, 3.0]]
    noises = [[0.0, 1.0], [-0.5, 0.25]]
    olds = [[0.8, -1.5], [0.25, 1.0]]
    news = [[1.6, -2.0], [1.5, 2.5]]
    times = [0.25, 0.6]
    probabilities = [0.75, 0.2]
    beta, clip, epsilon = 1.3, 2.7, 0.05
    expected_values, expected_gradients = [], []
    for x, e, old, new, t, p in zip(
        targets, noises, olds, news, times, probabilities, strict=True
    ):
        # Python float is IEEE double. Evaluate branch errors directly in these
        # coordinates; the derivative holds the population normalizers fixed.
        ep = [
            (1 - t) * (ev + (1 - beta) * ov + beta * nv - xv)
            for xv, ev, ov, nv in zip(x, e, old, new, strict=True)
        ]
        em = [
            (1 - t) * (ev + (1 + beta) * ov - beta * nv - xv)
            for xv, ev, ov, nv in zip(x, e, old, new, strict=True)
        ]
        hp, hm = (
            max(sum(abs(v) for v in ep) / 2, epsilon),
            max(sum(abs(v) for v in em) / 2, epsilon),
        )
        expected_values.append(
            clip
            / beta
            * (
                p * sum(v * v for v in ep) / (2 * hp)
                + (1 - p) * sum(v * v for v in em) / (2 * hm)
            )
        )
        expected_gradients.append(
            [
                clip * (1 - t) * (p * vp / hp - (1 - p) * vm / hm)
                for vp, vm in zip(ep, em, strict=True)
            ]
        )
    new = torch.tensor(news, requires_grad=True)
    result = losses(
        new,
        torch.tensor(olds),
        torch.tensor(targets),
        torch.tensor(noises),
        torch.tensor(times),
        torch.tensor(probabilities),
        parameterization="clean_adaptive",
        beta=beta,
        clip=clip,
        epsilon=epsilon,
    )
    result.sum().backward()
    torch.testing.assert_close(
        result.detach(), torch.tensor(expected_values), rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        new.grad, torch.tensor(expected_gradients), rtol=2e-6, atol=2e-6
    )
    with pytest.raises(ValueError, match="terminal endpoint"):
        losses(
            new,
            torch.tensor(olds),
            torch.tensor(targets),
            torch.tensor(noises),
            torch.ones(2),
            torch.tensor(probabilities),
            parameterization="clean_adaptive",
            beta=beta,
            clip=clip,
            epsilon=epsilon,
        )


def test_tensor_hash_canonicalizes_scalar_expanded_and_bfloat_storage():
    from src.posttrain.rollout import tensor_digest

    for dtype in (torch.float32, torch.bfloat16):
        scalar = torch.tensor(1.0, dtype=dtype)
        assert (
            tensor_digest(scalar)
            == tensor_digest(scalar.expand(1, 1))
            == tensor_digest(torch.ones(1, dtype=dtype))
        )
        assert tensor_digest(scalar.expand(2, 2)) == tensor_digest(
            torch.ones(4, dtype=dtype)
        )
        assert tensor_digest(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype).t()
        ) == tensor_digest(torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=dtype))
