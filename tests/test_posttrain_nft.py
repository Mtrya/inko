"""Independent NFT equations, grouped statistics and whole-round state flow."""

import copy
import math
from dataclasses import replace

import pytest
import torch

from src.posttrain.checkpoint import assert_exact
from src.posttrain.clock import Clock, rounds_before
from src.posttrain.engine import update_round
from src.posttrain.objectives.nft import losses, reference_losses
from src.posttrain.optim import RoleOptimizer
from src.posttrain.rewards import group_advantages
from src.posttrain.roles import Roles
from src.posttrain.rollout import collect_nft
from src.posttrain.schedules import refresh_decay
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


def new_roles(config, base):
    return Roles(copy.deepcopy(base), config)


def round_rollouts(config, roles, base, round_index=0, generator_step=0):
    """Every prompt group and draw of one collection round, through the old policy."""
    cond = (
        torch.full((1, 2, 1024), 0.01),
        torch.full((1, 1024), 0.01),
        torch.ones(1, 2),
    )
    return [
        collect_nft(
            roles.old_field,
            config,
            [{"id": f"p{slot}"}],
            [f"{round_index}:{slot}:{draw}"],
            round_index,
            0,
            generator_step,
            cond,
            None,
            "cpu",
        )
        for slot in range(config.collection.prompts_per_round)
        for draw in range(config.collection.draws_per_prompt)
    ]


def probabilities(rollouts, value=0.8):
    return {
        key: (value if key.endswith(":0") else 1 - value)
        for rollout in rollouts
        for key in rollout.sample_ids
    }


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_round_update_keeps_detached_roles_and_reports_terms(
    training_config, small_model, mode
):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    config = replace(
        training_config, student=replace(training_config.student, mode=mode)
    )
    base = small_model()
    roles = new_roles(config, base)
    rollouts = round_rollouts(config, roles, base)
    optimizer = RoleOptimizer(
        roles.student, config.student_optimizer, config.training.steps
    )
    old, reference = (
        copy.deepcopy(roles.old.state_dict()),
        copy.deepcopy(roles.reference.state_dict()),
    )
    student = copy.deepcopy(roles.student.state_dict())
    evidence = update_round(
        config,
        roles,
        roles.student_field,
        rollouts,
        probabilities(rollouts),
        optimizer,
        0,
        0,
    )
    # The reference anchor never moves; the old policy moves only on commit.
    assert_exact(old, roles.old.state_dict())
    assert_exact(reference, roles.reference.state_dict())
    assert any(
        not torch.equal(student[name], value)
        for name, value in roles.student.named_parameters()
        if value.requires_grad
    )
    assert optimizer.steps == 1
    assert set(evidence["term_losses"]) == {"nft", "reference"}
    assert evidence["student_grad_norm"] >= 0
    assert len(evidence["inputs"]) == len(rollouts) * config.nft.time_draws
    assert all(record["draw"] == 0 for record in evidence["inputs"])


def test_round_update_rejects_incomplete_rewards_and_stale_policy(
    training_config, small_model
):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    config = training_config
    base = small_model()
    roles = new_roles(config, base)
    rollouts = round_rollouts(config, roles, base)
    optimizer = RoleOptimizer(
        roles.student, config.student_optimizer, config.training.steps
    )
    complete = probabilities(rollouts)
    incomplete = dict(list(complete.items())[1:])
    with pytest.raises(ValueError, match="complete global reward set"):
        update_round(
            config, roles, roles.student_field, rollouts, incomplete, optimizer, 0, 0
        )
    out_of_range = {key: 1.5 for key in complete}
    with pytest.raises(ValueError, match="invalid global reward probabilities"):
        update_round(
            config, roles, roles.student_field, rollouts, out_of_range, optimizer, 0, 0
        )
    assert optimizer.steps == 0
    with pytest.raises(ValueError, match="stale or missing"):
        update_round(
            config, roles, roles.student_field, rollouts, complete, optimizer, 0, 3
        )
    assert optimizer.steps == 0


def test_old_refresh_decay_moves_only_the_frozen_copy(training_config, small_model):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    config = replace(
        training_config,
        old_policy=replace(training_config.old_policy, strategy="constant", decay=0.25),
    )
    base = small_model()
    roles = new_roles(config, base)
    trainable = {n for n, p in roles.student.named_parameters() if p.requires_grad}
    reference = copy.deepcopy(roles.reference.state_dict())
    before = copy.deepcopy(roles.old.state_dict())
    with torch.no_grad():
        for parameter in roles.student.parameters():
            if parameter.requires_grad:
                parameter.add_(2)
    roles.refresh_old(0.25)
    for name in trainable:
        torch.testing.assert_close(
            roles.old.state_dict()[name], before[name] + 1.5, rtol=1e-6, atol=1e-6
        )
    assert_exact(reference, roles.reference.state_dict())


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_role_separation_and_lora_base_sharing(training_config, small_model, mode):
    config = replace(
        training_config, student=replace(training_config.student, mode=mode)
    )
    roles = new_roles(config, small_model())
    trainable = {n for n, p in roles.student.named_parameters() if p.requires_grad}
    assert roles.old is not None and roles.reference is not None
    for name, parameter in roles.student.named_parameters():
        old = dict(roles.old.named_parameters())[name]
        reference = dict(roles.reference.named_parameters())[name]
        assert not old.requires_grad and not reference.requires_grad
        if name in trainable:
            assert parameter is not old and parameter is not reference
            assert old is not reference
        elif mode == "lora":
            assert parameter is old is reference
    assert set(roles.ownership()) == {"student", "old", "reference"}
    without_anchor = replace(config, nft=replace(config.nft, reference_coefficient=0.0))
    assert new_roles(without_anchor, small_model()).reference is None


@pytest.mark.parametrize("targets,count", [("attention", 6), ("attention_mlp", 12)])
def test_student_lora_targets_are_zero_initialized_and_merge(
    targets, count, small_model
):
    from src.posttrain.adapters import WeightLoRA, install_adapters, merge_student

    torch.manual_seed(4)
    model = small_model()
    original = copy.deepcopy(model)
    names = install_adapters(model, rank=2, scale=3, targets=targets)
    assert len(names) == count
    args = (
        torch.randn(1, 16, 2, 2),
        torch.tensor([0.4]),
        torch.randn(1, 3, 1024),
        torch.randn(1, 1024),
        torch.ones(1, 3),
    )
    torch.testing.assert_close(model(*args), original(*args), rtol=0, atol=0)
    for module in model.modules():
        if isinstance(module, WeightLoRA):
            torch.nn.init.normal_(module.b.weight, std=0.03)
    torch.testing.assert_close(
        model(*args), merge_student(model)(*args), rtol=2e-5, atol=2e-6
    )
    assert all(
        ".a." in n or ".b." in n for n, p in model.named_parameters() if p.requires_grad
    )


def test_clock_requires_complete_rounds_and_records_its_schedule(training_config):
    config = replace(
        training_config,
        nft=replace(training_config.nft, updates_per_round=2),
        training=replace(training_config.training, steps=4, checkpoint_every=2),
    )
    assert rounds_before(config, 4) == 2
    with pytest.raises(ValueError, match="inside a collection round"):
        rounds_before(config, 1)
    clock = Clock()
    with pytest.raises(ValueError, match="complete uncommitted collection round"):
        clock.commit_round(config, 0.0)
    for _ in range(2):
        clock.update(config)
    clock.commit_round(config, 0.0)
    assert (clock.step, clock.committed_rounds, clock.old_policy_version) == (2, 1, 1)
    assert clock.samples_used == 2 * 4 and clock.samples_scored == 4
    assert clock.old_refresh_decay == 0.0
    clock.validate(config)
    corrupt = clock.state_dict()
    corrupt["old_policy_version"] = 7
    with pytest.raises(ValueError, match="clocks differ"):
        Clock.from_state(corrupt).validate(config)


def test_config_rejects_incomplete_rounds_and_degenerate_groups(training_config):
    with pytest.raises(ValueError, match="evenly"):
        replace(
            training_config,
            execution=replace(training_config.execution, learner_devices=(0, 1, 2)),
        )
    with pytest.raises(ValueError, match="two draws"):
        replace(
            training_config,
            collection=replace(training_config.collection, draws_per_prompt=1),
        )
    with pytest.raises(ValueError, match="complete rounds"):
        replace(
            training_config,
            training=replace(training_config.training, steps=4),
            nft=replace(training_config.nft, updates_per_round=3),
        )
    with pytest.raises(ValueError, match="sum to one"):
        replace(
            training_config,
            rewards=replace(
                training_config.rewards,
                channels=(replace(training_config.rewards.channels[0], weight=0.4),),
            ),
        )
