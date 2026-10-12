"""ART guidance statistics, frozen-target loss, masking and trajectory capture."""

import copy
import math
from dataclasses import replace

import pytest
import torch
from test_posttrain_nft import new_roles

from src.posttrain.checkpoint import assert_exact
from src.posttrain.clock import Clock
from src.posttrain.engine import update_art
from src.posttrain.guidance import GroupGuidance, group_guidance, round_guidance
from src.posttrain.objectives.art import KL_THRESHOLD, NOISE_LEVEL, losses
from src.posttrain.rollout import ARTRollout, art_window, collect_art, tensor_digest
from src.posttrain.sampling import sample
from src.posttrain.session import TrainingState
from src.posttrain.training_config import ART, load_training_config


def test_group_guidance_matches_the_hand_computed_direction():
    # Group 0: rewards 0,1,2 with endpoints (1,0), (0,1), (1,1).
    # Centered rewards -1,0,1 give covariance [0, 0.5] over K-1=2,
    # MAD 2/3, signal [0, 0.75], RMS 0.75/sqrt(2) and unit-RMS direction [0, sqrt(2)].
    endpoints = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]]
    )
    rewards = torch.tensor([0.0, 1.0, 2.0, 5.0, 5.0, 5.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    stats = group_guidance(endpoints, rewards, groups, 2, expected_group_size=3)
    assert stats["active"].tolist() == [True, False]
    torch.testing.assert_close(
        stats["directions"][0], torch.tensor([0.0, math.sqrt(2)]), rtol=0, atol=1e-6
    )
    # A constant-reward group has no direction, magnitude or training signal.
    torch.testing.assert_close(stats["directions"][1], torch.zeros(2), rtol=0, atol=0)
    assert stats["magnitudes"].tolist() == [1.0, 0.0]
    torch.testing.assert_close(
        stats["covariance"][0], torch.tensor([0.0, 0.5]), rtol=0, atol=1e-6
    )
    with pytest.raises(ValueError, match="incomplete prompt group"):
        group_guidance(endpoints[:5], rewards[:5], groups[:5], 2, expected_group_size=3)


@pytest.mark.parametrize("size", [8, 24])
@pytest.mark.parametrize("reward", [0.1, -0.3])
def test_exact_fractional_reward_ties_have_zero_guidance(size, reward):
    torch.manual_seed(1)
    stats = group_guidance(
        torch.randn(size, 4),
        torch.full((size,), reward),
        torch.zeros(size, dtype=torch.long),
        1,
        expected_group_size=size,
    )
    assert stats["active"].tolist() == [False]
    assert stats["magnitudes"].tolist() == [0.0]
    assert torch.equal(stats["covariance"], torch.zeros(1, 4))
    assert torch.equal(stats["directions"], torch.zeros(1, 4))


def test_small_nonzero_reward_spread_is_not_treated_as_a_tie():
    stats = group_guidance(
        torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0.0, 1e-8]),
        torch.zeros(2, dtype=torch.long),
        1,
        expected_group_size=2,
    )
    assert stats["active"].tolist() == [True]
    torch.testing.assert_close(
        stats["directions"], torch.tensor([[0.0, math.sqrt(2)]]), rtol=0, atol=1e-6
    )


def test_covariance_is_stable_under_a_representable_reward_offset():
    endpoints = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0], [7.0, -2.0]])
    # Centered rewards [-1.5,-0.5,0.5,1.5] give covariance [10/3,-10/3].
    for offset in (0.0, 1_000_000.0):
        stats = group_guidance(
            endpoints,
            torch.tensor([0.0, 1.0, 2.0, 3.0]) + offset,
            torch.zeros(4, dtype=torch.long),
            1,
            expected_group_size=4,
        )
        torch.testing.assert_close(
            stats["covariance"], torch.tensor([[10 / 3, -10 / 3]])
        )
        torch.testing.assert_close(stats["directions"], torch.tensor([[1.0, -1.0]]))


def test_art_windows_shuffle_before_splitting(art_config, monkeypatch):
    from types import SimpleNamespace

    config = replace(
        art_config, collection=replace(art_config.collection, prompts_per_round=4)
    )
    rollout = SimpleNamespace(
        draw_indices=list(range(8)),
        sample_ids=[f"0:{i // 2}:{i % 2}" for i in range(8)],
        collection_round=0,
        generator_step=0,
    )

    def permutation(size, *, generator):
        assert size == 8 and isinstance(generator, torch.Generator)
        return torch.tensor([5, 0, 7, 2, 6, 1, 4, 3])

    monkeypatch.setattr(torch, "randperm", permutation)
    assert [i for _, i in art_window(config, [rollout], 0)] == [5, 0, 7, 2]
    assert [i for _, i in art_window(config, [rollout], 1)] == [6, 1, 4, 3]


def test_art_windows_are_balanced_and_independent_of_microbatch_packing(art_config):
    from types import SimpleNamespace

    config = replace(
        art_config,
        collection=replace(
            art_config.collection, prompts_per_round=12, draws_per_prompt=8
        ),
        execution=replace(art_config.execution, learner_devices=tuple(range(12))),
    )
    rng_before = torch.get_rng_state().clone()
    for rank in range(12):
        draws = list(range(rank, 96, 12))

        def pack(indices):
            return SimpleNamespace(
                draw_indices=indices,
                sample_ids=[f"3:{i // 8}:{i % 8}" for i in indices],
                collection_round=3,
                generator_step=2,
            )

        whole = [pack(draws)]
        reordered = [pack([i]) for i in reversed(draws)]
        windows = []
        for window in (0, 1):
            expected = [r.draw_indices[i] for r, i in art_window(config, whole, window)]
            actual = [
                r.draw_indices[i] for r, i in art_window(config, reordered, window)
            ]
            assert actual == expected and len(actual) == 4
            windows.append(set(actual))
        assert not windows[0] & windows[1]
        assert windows[0] | windows[1] == set(draws)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_group_guidance_marks_an_unreduced_partial_group_as_incomplete():
    endpoints = torch.randn(4, 3)
    rewards = torch.tensor([0.0, 1.0, 0.0, 2.0])
    groups = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="incomplete prompt group"):
        group_guidance(endpoints, rewards, groups, 2, expected_group_size=4)


def test_round_guidance_groups_by_draw_slot_not_prompt_identity(art_config):
    # A prompt repeated within one round (the cursor can cross an epoch
    # boundary) still forms two independent groups keyed by collection slot.
    config = art_config
    terminals = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[2.0, 1.0]]),
    ]
    rewards = {"0:0:0": 0.0, "0:0:1": 1.0, "0:1:0": 0.0, "0:1:1": 2.0}
    rollouts = [
        ARTRollout(
            [f"0:{slot}:{draw}"],
            ["p0"],
            [slot * config.collection.draws_per_prompt + draw],
            [0],
            0,
            terminals[slot * 2 + draw],
            torch.zeros(1, config.collection.steps, 2),
            torch.zeros(1, config.collection.steps, 2),
            (0.0, 0.5, 1.0),
            (),
            0,
        )
        for slot in range(2)
        for draw in range(2)
    ]
    guidance = round_guidance(config, rollouts, rewards)
    # Slot 0: endpoints (1,0),(0,1) with rewards 0,1 give covariance
    # (-0.5,0.5), MAD 0.5, signal (-1,1) and unit-RMS direction (-1,1).
    # Slot 1: endpoints (1,1),(2,1) with rewards 0,2 give covariance (1,0),
    # MAD 1, signal (1,0) and direction (sqrt(2),0).
    assert guidance.active.tolist() == [True, True]
    assert guidance.magnitudes.tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        guidance.directions[0], torch.tensor([-1.0, 1.0]), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        guidance.directions[1],
        torch.tensor([math.sqrt(2), 0.0]),
        rtol=0,
        atol=1e-6,
    )


def test_loss_regresses_the_frozen_cached_target_plus_guidance():
    cached = torch.tensor([[0.4]])
    times, deltas = torch.tensor([0.5]), torch.tensor([0.1])
    direction, magnitude = torch.tensor([[1.0]]), torch.tensor([1.0])
    new = cached.clone().requires_grad_(True)
    result, keep, masked = losses(new, cached, times, deltas, direction, magnitude)
    # residual = target - new = (cached + G) - cached = G, weight = 1 - t.
    assert keep.item() and not masked.item()
    assert float(result.detach()) == pytest.approx((1 - 0.5) * 1.0**2, abs=1e-7)
    result.sum().backward()
    assert float(new.grad) == pytest.approx(-(1 - 0.5) * 2 * 1.0, abs=1e-6)


def test_second_window_residual_is_guidance_minus_the_move():
    cached = torch.tensor([[0.2]])
    times, deltas = torch.tensor([0.5]), torch.tensor([0.1])
    guidance, magnitude = torch.tensor([[0.3]]), torch.tensor([1.0])
    for move in (0.0, 0.05, -0.04):
        new = (cached + move).clone().requires_grad_(True)
        result, _, masked = losses(new, cached, times, deltas, guidance, magnitude)
        if masked:
            assert float(result.detach()) == 0
            continue
        # d = new - cached and residual = G - d; the loss derivative pulls the
        # prediction back toward the frozen target by twice the residual.
        residual = 0.3 - move
        assert float(result.detach()) == pytest.approx(0.5 * residual**2, abs=1e-7)
        result.sum().backward()
        assert float(new.grad) == pytest.approx(-2 * 0.5 * residual, abs=1e-6)


def test_mask_blocks_only_outward_moves_above_the_proxy_threshold():
    cached = torch.zeros(1, 1)
    times, deltas = torch.tensor([0.5]), torch.tensor([0.1])

    def evaluate(direction, move):
        new = (cached + move).clone().requires_grad_(True)
        result, keep, masked = losses(
            new, cached, times, deltas, direction, torch.ones(1)
        )
        return float(result.detach()), bool(keep), bool(masked)

    # coefficient = dt * (1 + 0.5 * 0.8^2); std = 0.8 * sqrt((1-t)/t * dt).
    coefficient = 0.1 * (1 + 0.5 * NOISE_LEVEL**2)
    std = NOISE_LEVEL * math.sqrt((1 - 0.5) / 0.5 * 0.1)
    outward = 0.05
    proxy = coefficient**2 * outward**2 / (2 * std**2)
    assert proxy >= KL_THRESHOLD
    assert evaluate(torch.full((1, 1), 0.1), outward) == (0.0, False, True)
    # The same move is kept when it corrects outward past the guidance target.
    loss, keep, masked = evaluate(torch.full((1, 1), 0.01), outward)
    assert not masked and keep and loss == pytest.approx(0.5 * (0.01 - outward) ** 2)
    # A small same-direction move stays below the proxy threshold.
    small = 0.001
    assert coefficient**2 * small**2 / (2 * std**2) < KL_THRESHOLD
    loss, keep, masked = evaluate(torch.full((1, 1), 0.1), small)
    assert keep and not masked and loss == pytest.approx(0.5 * (0.1 - small) ** 2)


def test_initial_state_uses_the_reference_noise_level():
    cached = torch.zeros(1, 1)
    times, deltas = torch.tensor([0.0]), torch.tensor([0.05])
    direction, magnitude = torch.tensor([[1.0]]), torch.ones(1)
    # At t = 0 the proxy uses std = 0.8 and coefficient = dt; the paper's
    # non-initial coefficient dt * (1 + 0.5 * 0.8^2) would cross the threshold.
    small = 0.06
    initial = 0.05**2 * small**2 / (2 * NOISE_LEVEL**2)
    general = (0.05 * (1 + 0.5 * NOISE_LEVEL**2)) ** 2 * small**2 / (2 * NOISE_LEVEL**2)
    assert initial < KL_THRESHOLD <= general
    result, keep, masked = losses(
        cached + small, cached, times, deltas, direction, magnitude
    )
    assert keep.item() and not masked.item()
    assert float(result.detach()) == pytest.approx((1 - small) ** 2)
    # A large move is masked, so the initial standard deviation is finite at 0.8.
    result, keep, masked = losses(
        cached + 0.4, cached, times, deltas, direction, magnitude
    )
    assert masked.item() and not keep.item()
    assert float(result.detach()) == 0


def test_timestep_weight_scales_the_unmasked_loss():
    cached = torch.zeros(1, 1)
    direction, magnitude = torch.tensor([[0.1]]), torch.ones(1)
    move = 1e-4
    values = []
    for time in (0.25, 0.75):
        new = (cached + move).clone().requires_grad_(True)
        result, keep, _ = losses(
            new,
            cached,
            torch.tensor([time]),
            torch.tensor([0.1]),
            direction,
            magnitude,
        )
        assert keep.item()
        values.append(float(result.detach()))
    assert values[0] / values[1] == pytest.approx((1 - 0.25) / (1 - 0.75), rel=1e-6)


def test_item_weights_and_gradients_stay_on_the_item_axis():
    # Two latent items with different timesteps keep their own (1 - t) weight and
    # their own guidance magnitude; a [N] weight broadcast along the latent axis
    # would instead mix the items and multiply the loss tensor's element count.
    cached = torch.zeros(2, 1, 2, 2)
    new = cached.clone().requires_grad_(True)
    direction = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1).expand(2, 1, 2, 2)
    result, keep, masked = losses(
        new,
        cached,
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.1, 0.1]),
        direction,
        torch.ones(2),
    )
    assert result.shape == (2,) and keep.tolist() == [True, True]
    assert not masked.any()
    # residual is 1 and 2, so the weighted mean losses are 0.75 and 1.0.
    assert float(result.detach().sum()) == pytest.approx(1.75, abs=1e-6)
    result.sum().backward()
    # Each item's gradient is its own residual over the latent element count.
    elements = cached[0].numel()
    torch.testing.assert_close(
        new.grad,
        torch.tensor([-2 * 0.75 * 1.0 / elements, -2 * 0.25 * 2.0 / elements])
        .reshape(2, 1, 1, 1)
        .expand_as(new),
        rtol=0,
        atol=1e-6,
    )


def test_inactive_and_invalid_items_are_rejected_or_dropped():
    cached = torch.zeros(2, 1)
    new = cached.clone().requires_grad_(True)
    times, deltas = torch.tensor([0.5, 0.5]), torch.tensor([0.1, 0.1])
    direction = torch.tensor([[1.0], [0.0]])
    result, keep, _ = losses(new, cached, times, deltas, direction, torch.ones(2))
    assert keep.tolist() == [True, False]  # zero direction: inactive, no signal
    assert result[1].item() == 0
    with pytest.raises(ValueError, match="data endpoint"):
        losses(new, cached, torch.tensor([1.0, 0.5]), deltas, direction, torch.ones(2))
    with pytest.raises(ValueError, match="toward data"):
        losses(new, cached, times, torch.tensor([0.0, 0.1]), direction, torch.ones(2))
    with pytest.raises(ValueError, match="shapes differ"):
        losses(new, cached, times, deltas, torch.zeros(2, 2), torch.ones(2))


def test_collection_caches_aligned_trajectory_and_leaves_the_nft_path_lean(
    art_config, small_model
):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    config = art_config
    roles = new_roles(config, small_model())
    cond = (
        torch.randn(1, 2, 1024),
        torch.randn(1, 1024),
        torch.ones(1, 2),
    )
    rollout = collect_art(
        roles.student_field,
        config,
        [{"id": "p0"}],
        ["0:0:0"],
        [0],
        0,
        0,
        cond,
        "cpu",
    )
    steps = config.collection.steps
    assert rollout.draw_indices == [0] and rollout.seeds
    assert rollout.states.shape == (1, steps, 16, 2, 2)
    assert rollout.predictions.shape == rollout.states.shape
    assert rollout.states.dtype == torch.float32
    assert rollout.predictions.dtype == torch.float32
    assert rollout.grid[0] == 0.0 and rollout.grid[-1] == 1.0
    # The cached anchor is the same forward the training step would make there.
    with torch.no_grad():
        for step in range(steps):
            z = rollout.states[0, step].unsqueeze(0)
            t = torch.full((1,), rollout.grid[step])
            fresh = roles.student_field(z, t, cond)
            torch.testing.assert_close(
                fresh,
                rollout.predictions[0, step].unsqueeze(0),
                rtol=0,
                atol=0,
            )
    # The sampled terminal ends the cached trajectory by one Euler step.
    step_size = rollout.grid[-1] - rollout.grid[-2]
    torch.testing.assert_close(
        rollout.terminal[0],
        rollout.states[0, -1] + step_size * rollout.predictions[0, -1],
        rtol=0,
        atol=0,
    )
    lean = sample(lambda x, t: x, [1, 2], (1, 2, 2), grid=(0, 0.5, 1))
    assert lean.states is None and lean.predictions is None


def test_art_round_updates_two_disjoint_windows_and_round_trips_state(
    art_config, small_model
):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    config = art_config
    roles = new_roles(config, small_model())
    cond = (torch.randn(1, 1, 1024), torch.randn(1, 1024), torch.ones(1, 1))
    rollouts = [
        collect_art(
            roles.student_field,
            config,
            [{"id": f"p{slot}"}],
            [f"0:{slot}:{draw}"],
            [slot * config.collection.draws_per_prompt + draw],
            0,
            0,
            cond,
            "cpu",
        )
        for slot in range(config.collection.prompts_per_round)
        for draw in range(config.collection.draws_per_prompt)
    ]
    guidance = round_guidance(
        config,
        rollouts,
        {
            key: float(index)
            for index, rollout in enumerate(rollouts)
            for key in rollout.sample_ids
        },
    )
    assert isinstance(guidance, GroupGuidance)
    assert guidance.active.tolist() == [True, True]
    assert guidance.directions.shape == (2, 16, 2, 2)
    no_anchor = replace(config, art=ART(reference_coefficient=0.0))
    assert new_roles(no_anchor, small_model()).reference is None
    state = TrainingState(config, roles, None, Clock())
    assert roles.old is None and roles.reference is not None
    frozen = copy.deepcopy(roles.reference.state_dict())
    student = copy.deepcopy(roles.student.state_dict())
    cached = rollouts[0].predictions.clone()
    by_id = {key: rollout for rollout in rollouts for key in rollout.sample_ids}
    windows = {0: set(), 1: set()}
    for window in (0, 1):
        evidence = update_art(
            config,
            roles,
            roles.student_field,
            rollouts,
            guidance,
            state.student_optimizer,
            0,
            window,
        )
        state.clock.update(config)
        assert evidence["window"] == window
        assert evidence["term_losses"]["art"] >= 0
        assert evidence["art_items"] == (
            config.collection.draws_per_prompt * config.collection.steps
        )
        assert evidence["art_kept"] + evidence["art_masked"] <= evidence["art_items"]
        for record in evidence["inputs"]:
            assert record["window"] == window
            rollout = by_id[record["id"]]
            index = rollout.sample_ids.index(record["id"])
            start, stop = record["steps"]
            # The training inputs name the cached trajectory state it regresses.
            assert record["state"] == tensor_digest(rollout.states[index, start:stop])
            assert record["cached"] == tensor_digest(
                rollout.predictions[index, start:stop]
            )
            assert record["time"] == tensor_digest(
                torch.tensor(rollout.grid[start:stop])
            )
            windows[window].add(record["id"])
    assert state.student_optimizer.steps == 2
    # Both windows regressed one frozen target: reference and cache never moved.
    assert_exact(frozen, roles.reference.state_dict())
    assert_exact(cached, rollouts[0].predictions)
    assert any(
        not torch.equal(student[name], value)
        for name, value in roles.student.named_parameters()
        if value.requires_grad
    )
    # Every draw trains in exactly one window, and the two windows cover the round.
    assert windows[0] & windows[1] == set()
    assert windows[0] | windows[1] == set(by_id)
    state.commit_round()
    assert state.clock.old_refresh_decay is None
    assert state.clock.old_policy_version == 0
    saved = state.state_dict()
    assert saved["old"] is None and saved["old_schedule"] is None
    with pytest.raises(ValueError, match="stale or missing ART"):
        update_art(
            config,
            roles,
            roles.student_field,
            rollouts,
            guidance,
            state.student_optimizer,
            1,
            0,
        )
    assert state.student_optimizer.steps == 2


def test_art_reward_barrier_returns_the_combined_rewards(tmp_path, art_config):
    from src.posttrain.records import atomic_json, file_hash
    from src.posttrain.round_scoring import (
        collection_path,
        reward_path,
        wait_for_rewards,
    )

    output, attempt, step = tmp_path, "a", 0
    local = [{"sample_id": "0:0:0"}, {"sample_id": "0:0:1"}]
    collection = collection_path(output, attempt, step, 0)
    atomic_json(collection, {"step": step, "rank": 0, "records": local})
    packet = {
        "step": step,
        "policy_version": step,
        "collections": [file_hash(collection)],
        "reward": {
            "sample_ids": ["0:0:0", "0:0:1", "0:1:0", "0:1:1"],
            "combined": [1.0, -1.0, 0.5, -0.5],
        },
    }
    atomic_json(reward_path(output, attempt, step), packet)
    values, reward = wait_for_rewards(art_config, output, attempt, step, 0, local)
    # ART guidance consumes the combined reward; no optimality probability exists.
    assert values == {"0:0:0": 1.0, "0:0:1": -1.0, "0:1:0": 0.5, "0:1:1": -0.5}
    assert reward == packet["reward"]
    with pytest.raises(ValueError, match="missing local reward"):
        wait_for_rewards(art_config, output, attempt, step, 0, [{"sample_id": "0:9:9"}])


def test_art_config_rejects_the_other_objective_sections(art_config, training_config):
    nft = load_training_config("configs/posttrain.toml")
    art = load_training_config("configs/posttrain_art.toml")
    assert nft.updates_per_round == 1 and nft.objective == "nft" and nft.art is None
    assert art.updates_per_round == 2 and art.objective == "art"
    assert art.nft is None and art.old_policy is None
    with pytest.raises(ValueError, match="own config section"):
        replace(art_config, nft=training_config.nft)
    with pytest.raises(ValueError, match="old-policy refresh belongs"):
        replace(art_config, old_policy=training_config.old_policy)
    with pytest.raises(ValueError, match="own config section"):
        replace(training_config, objective="art")
    with pytest.raises(ValueError, match="cannot use CFG"):
        replace(art_config, collection=replace(art_config.collection, guidance=1.5))
    with pytest.raises(ValueError, match="two update windows"):
        replace(
            art_config,
            collection=replace(
                art_config.collection, prompts_per_round=3, draws_per_prompt=3
            ),
        )
    with pytest.raises(ValueError, match="two draws"):
        replace(
            art_config,
            collection=replace(art_config.collection, draws_per_prompt=1),
        )
    with pytest.raises(ValueError, match="length/seed/EMA"):
        replace(art_config, training=replace(art_config.training, rounds=0))
    with pytest.raises(ValueError, match="reference coefficient"):
        replace(art_config, art=ART(reference_coefficient=float("nan")))
    with pytest.raises(ValueError, match="unknown training objective"):
        replace(art_config, objective="dmd")
