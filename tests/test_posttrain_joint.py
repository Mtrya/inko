"""Role separation, grouped admission and whole-round recovery, never quality."""

import copy
from dataclasses import asdict, replace

import pytest
import torch
from PIL import Image
from test_posttrain_dmd import config as dmd_config
from test_posttrain_dmd import small_model
from test_posttrain_scoring import Endpoint

from src.posttrain.checkpoint import assert_exact, load, publish
from src.posttrain.engine import update_joint, update_round
from src.posttrain.initialization import boundary_manifest, load_boundary
from src.posttrain.records import file_hash
from src.posttrain.roles import Roles
from src.posttrain.rollout import collect, collect_nft, nft_inputs
from src.posttrain.round_scoring import acquire_round
from src.posttrain.scoring.contracts import Channel
from src.posttrain.scoring.lifecycle import Services
from src.posttrain.session import TrainingState
from src.posttrain.stages import Clock, rounds_before
from src.posttrain.training_config import Joint, RewardChannel, Rewards, Stages


def joint_config(spec, mode="full", dmd=1, nft=1, reference=0.1):
    c = dmd_config(mode)
    return replace(
        c,
        collection=replace(c.collection, draws_per_prompt=2),
        training=replace(c.training, steps=4, checkpoint_every=2),
        stages=Stages(0, 4, "cold", "2" * 64),
        joint=Joint(dmd, nft, reference if nft else 0),
        nft=replace(c.nft, updates_per_round=2, time_draws=2),
        scorers=(spec,) if nft else (),
        rewards=Rewards(
            (
                (
                    RewardChannel(
                        spec.id + "." + spec.outputs[0].name,
                        spec.outputs[0].direction,
                        1,
                        "none",
                    ),
                )
                if nft
                else ()
            ),
            1e-4,
            "mean_one_exp",
            1,
        ),
    )


def new_state(c, base, *, stage="joint", generator_steps=0, fake_steps=0):
    roles = Roles(
        copy.deepcopy(base),
        copy.deepcopy(base) if c.dmd_enabled(stage) else None,
        c,
        stage=stage,
    )
    if stage == "joint":
        roles.enter_joint()
    clock = Clock.initial(c, generator_steps, fake_steps)
    return TrainingState(
        c,
        roles,
        roles.student_field,
        roles.fake_field,
        clock,
        initial_generator_steps=generator_steps,
        initial_fake_steps=fake_steps,
    )


def nft_round(state):
    c, clock = state.config, state.clock
    # Fixed prompt-group conditioning, with each group's draws split across batches.
    rows = []
    for slot in range(2):
        cond = (
            torch.full((1, 2, 1024), 0.01 * (slot + 1)),
            torch.full((1, 1024), 0.01 * (slot + 1)),
            torch.ones(1, 2),
        )
        for draw in range(2):
            key = f"{clock.committed_rounds}:{slot}:{draw}"
            rows.append(
                collect_nft(
                    state.roles.old_field,
                    c,
                    [{"id": f"p{slot}"}],
                    [key],
                    clock.committed_rounds,
                    clock.old_policy_version,
                    clock.global_generator_steps,
                    cond,
                    None,
                    "cpu",
                )
            )
    return rows


def fresh_dmd(state, rows, inner):
    clock = state.clock
    return [
        collect(
            state.roles.student_field,
            state.config,
            [{"id": p} for p in r.prompt_ids],
            [f"{key}:dmd:{inner}" for key in r.sample_ids],
            clock.committed_rounds,
            r.conditioning,
            r.unconditional,
            "cpu",
            stage="joint",
            generator_step=clock.global_generator_steps,
            stream="dmd_collection",
        )
        for r in rows
    ]


def run_inner(state, rows, inner):
    dmd = fresh_dmd(state, rows, inner) if state.config.joint.dmd_coefficient else None
    p = {key: 0.2 if key.endswith(":0") else 0.8 for r in rows for key in r.sample_ids}
    report = update_joint(
        state.config,
        state.roles,
        state.student,
        state.fake,
        rows,
        dmd,
        p,
        state.student_optimizer,
        state.fake_optimizer,
        state.clock.global_generator_steps,
        state.clock.old_policy_version,
    )
    state.clock.update(state.config)
    return report, dmd


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_old_reference_independence_refresh_and_separate_ema(sampling_config, mode):
    torch.manual_seed(6)
    c = joint_config(sampling_config.scorers[0], mode)
    c = replace(
        c, training=replace(c.training, ema=(replace(c.training.ema[0], decay=0.75),))
    )
    state = new_state(c, small_model())
    roles = state.roles
    old0, ref0 = (
        copy.deepcopy(roles.old.state_dict()),
        copy.deepcopy(roles.reference.state_dict()),
    )
    trainable = {n for n, p in roles.student.named_parameters() if p.requires_grad}
    for name, p in roles.student.named_parameters():
        old, ref = (
            dict(roles.old.named_parameters())[name],
            dict(roles.reference.named_parameters())[name],
        )
        assert not old.requires_grad and not ref.requires_grad
        if name in trainable:
            assert p is not old and p is not ref and old is not ref
            with torch.no_grad():
                p.add_(2)
        elif mode == "lora":
            assert p is old is ref
    assert_exact(old0, roles.old.state_dict())
    assert_exact(ref0, roles.reference.state_dict())
    roles.update_ema()
    for name in trainable:
        torch.testing.assert_close(
            roles.ema["ema"][name], old0[name] + 0.5, rtol=0, atol=1e-6
        )
    roles.refresh_old(0.25)
    for name in trainable:
        torch.testing.assert_close(
            roles.old.state_dict()[name], old0[name] + 1.5, rtol=0, atol=1e-6
        )
    assert_exact(ref0, roles.reference.state_dict())
    with pytest.raises(ValueError, match="valid decay"):
        roles.refresh_old(1)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_whole_round_rollback_recollection_and_exact_resume(
    tmp_path, sampling_config, mode
):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    c = joint_config(sampling_config.scorers[0], mode)
    base = small_model()
    state = new_state(c, base, generator_steps=6, fake_steps=12)
    initial = copy.deepcopy(state.state_dict())
    rows = nft_round(state)
    old = copy.deepcopy(state.roles.old.state_dict())
    report0, dmd0 = run_inner(state, rows, 0)
    assert_exact(old, state.roles.old.state_dict())
    assert state.clock.old_policy_version == 0
    with pytest.raises(ValueError, match="inside a joint collection round"):
        state.state_dict()
    before_fake = copy.deepcopy(state.roles.fake.state_dict())
    with pytest.raises(ValueError, match="stale"):
        update_joint(
            c,
            state.roles,
            state.student,
            state.fake,
            rows,
            dmd0,
            {r.sample_ids[0]: 0.5 for r in rows},
            state.student_optimizer,
            state.fake_optimizer,
            7,
            0,
        )
    assert_exact(before_fake, state.roles.fake.state_dict())
    report1, dmd1 = run_inner(state, rows, 1)
    assert dmd0[0].student_version == 6 and dmd1[0].student_version == 7
    assert not torch.equal(dmd0[0].terminal, dmd1[0].terminal)
    assert report0["inputs"] != report1["inputs"]
    assert state.commit_round() == 0
    assert state.clock.old_policy_version == state.clock.joint_rounds == 1
    assert state.clock.fake_steps == 16 and state.clock.global_generator_steps == 8
    assert state.clock.samples_used == 16 and state.clock.samples_scored == 4
    for n, p in state.roles.student.named_parameters():
        torch.testing.assert_close(
            p, dict(state.roles.old.named_parameters())[n], rtol=0, atol=0
        )
    expected = copy.deepcopy(state.state_dict())
    # Discard incomplete work, load the committed initial boundary and recollect.
    restored = new_state(c, base, generator_steps=6, fake_steps=12)
    restored.load_state_dict(initial)
    recollected = nft_round(restored)
    for a, b in zip(rows, recollected, strict=True):
        assert a.seeds == b.seeds
        torch.testing.assert_close(a.terminal, b.terminal, rtol=0, atol=0)
    assert run_inner(restored, recollected, 0)[0] == report0
    assert run_inner(restored, recollected, 1)[0] == report1
    restored.commit_round()
    assert_exact(expected, restored.state_dict())
    point = publish(
        tmp_path / "checkpoints",
        2,
        {"case": mode},
        restored.state_dict(),
        {"cursor": 7},
        0,
        torch.device("cpu"),
    )
    saved, _ = load(point, {"case": mode}, 0)
    recovered = new_state(c, base, generator_steps=6, fake_steps=12)
    recovered.load_state_dict(saved)
    for s in (state, recovered):
        next_rows = nft_round(s)
        run_inner(s, next_rows, 0)
        run_inner(s, next_rows, 1)
        s.commit_round()
    assert_exact(state.state_dict(), recovered.state_dict())
    corrupt = copy.deepcopy(saved)
    corrupt["old_policy_version"] = 19
    with pytest.raises(ValueError, match="old-policy clock"):
        recovered.load_state_dict(corrupt)


@pytest.mark.parametrize("dmd,nft", [(1, 0), (0, 1), (1, 1)])
def test_inactive_roles_and_partial_round_rejection(sampling_config, dmd, nft):
    c = joint_config(sampling_config.scorers[0], dmd=dmd, nft=nft)
    state = new_state(c, small_model())
    assert (state.roles.teacher is not None) == bool(dmd)
    assert (state.roles.fake is not None) == bool(dmd)
    assert (state.fake_optimizer is not None) == bool(dmd)
    assert (state.roles.old is not None) == bool(nft)
    assert (state.roles.reference is not None) == bool(nft)
    if nft:
        with pytest.raises(ValueError, match="complete rounds"):
            replace(
                c,
                stages=replace(c.stages, joint_steps=3),
                training=replace(c.training, steps=3),
            )
        with pytest.raises(ValueError, match="inside a joint collection round"):
            rounds_before(c, 1)
    else:
        assert rounds_before(c, 1) == 1


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("dmd", [0, 1])
def test_stage_transition_retains_weights_resets_optimizers_and_ema(
    sampling_config, mode, dmd
):
    c = joint_config(sampling_config.scorers[0], mode, dmd=dmd)
    c = replace(c, stages=Stages(1, 4, "", ""), training=replace(c.training, steps=5))
    base = small_model()
    state = new_state(c, base, stage="cold")
    cond = (torch.zeros(1, 2, 1024), torch.zeros(1, 1024), torch.ones(1, 2))
    rows = [
        collect(
            state.roles.student_field,
            c,
            [{"id": f"p{slot}"}],
            [f"0:{slot}:{draw}"],
            0,
            cond,
            None,
            "cpu",
            stage="cold",
            generator_step=0,
        )
        for slot in range(2)
        for draw in range(2)
    ]
    update_round(
        c,
        state.roles,
        state.student,
        state.fake,
        rows,
        state.student_optimizer,
        state.fake_optimizer,
        0,
    )
    state.clock.update(c)
    state.commit_round()
    assert state.student_optimizer.steps == 1 and state.fake_optimizer.steps == 2
    assert state.student_optimizer.optimizers[0].state
    student = copy.deepcopy(state.roles.student.state_dict())
    fake = copy.deepcopy(state.roles.fake.state_dict())
    saved = copy.deepcopy(state.state_dict())
    assert saved["next_action"] == "transition"
    state.enter_joint()
    assert_exact(student, state.roles.student.state_dict())
    if dmd:
        assert_exact(fake, state.roles.fake.state_dict())
        assert state.fake_optimizer.steps == 0
        assert not state.fake_optimizer.optimizers[0].state
    else:
        assert (
            state.roles.fake
            is state.roles.teacher
            is state.fake
            is state.fake_optimizer
            is None
        )
    assert_exact(student, state.roles.old.state_dict())
    assert_exact(student, state.roles.reference.state_dict())
    assert state.student_optimizer.steps == 0
    assert not state.student_optimizer.optimizers[0].state
    for n, p in state.roles.student.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(state.roles.ema["ema"][n], p, rtol=0, atol=0)
    assert (
        state.clock.stage_generator_steps == 0 and state.clock.old_policy_version == 0
    )
    recovered = new_state(c, base, stage="cold")
    recovered.load_state_dict(saved)
    recovered.enter_joint()
    assert_exact(state.state_dict(), recovered.state_dict())


def test_fresh_time_noise_and_terminal_grid_exclusion(sampling_config):
    c = joint_config(sampling_config.scorers[0])
    state = new_state(c, small_model())
    r = nft_round(state)[0]
    a, b = nft_inputs(c, r, 0, 0), nft_inputs(c, r, 0, 1)
    q = nft_inputs(c, r, 1, 0)
    for other in (b, q):
        assert not torch.equal(a[0], other[0]) and not torch.equal(a[1], other[1])
    torch.testing.assert_close(nft_inputs(c, r, 0, 0)[0], a[0], rtol=0, atol=0)
    # Two Euler/renoise calls -> the deployed grid is {0,.5,1}; NFT excludes1.
    grid = replace(
        c,
        nft=replace(c.nft, time_distribution="grid"),
        collection=replace(c.collection, shift="none"),
    )
    for draw in range(12):
        assert float(nft_inputs(grid, r, 0, draw)[0]) in {0, 0.5}


def test_config_only_http_scorer_substitution_through_grouped_nft(
    tmp_path, sampling_config
):
    images = []
    for i in range(4):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i * 50, 20, 30)).save(path)
        images.append(path)
    for identifier, metric, direction, expected in (
        ("rubric", "match", "higher", [0, 1, 0.5, 0.5]),
        ("latency", "cost", "lower", [1, 0, 0.5, 0.5]),
    ):
        spec = replace(
            sampling_config.scorers[0],
            id=identifier,
            outputs=(Channel(metric, direction, (), "Fixture"),),
        )

        class Backend:
            def score(self, inputs, channel=metric):
                return [
                    {
                        channel: (
                            float(Image.open(i.image_path).getpixel((0, 0))[0]) / 50
                            if i.prompt == "first"
                            else 2
                        )
                    }
                    for i in inputs
                ]

        endpoint = Endpoint(tmp_path / identifier, spec, Backend())
        try:
            c = joint_config(spec, dmd=0, reference=0)
            c = replace(c, nft=replace(c.nft, advantage_clip=0.25))
            records = [
                {
                    "sample_id": f"0:{slot}:0" if i % 2 == 0 else f"0:{slot}:1",
                    "prompt": {
                        "id": f"p{slot}",
                        "text": "first" if slot == 0 else "second",
                        "structured": {"rubric": "identify content"},
                    },
                    "image_path": str(path),
                    "image_sha256": file_hash(path),
                }
                for i, path in enumerate(images)
                for slot in [i // 2]
            ]
            with Services(
                c.scorers,
                {identifier: {"endpoints": [endpoint.url], "launches": []}},
                tmp_path / identifier / "caller",
                c.execution.learner_devices,
                10,
            ) as services:
                result = acquire_round(
                    c, services.clients, records[::-1], stage="joint"
                )
            assert result["reward"]["probabilities"] == expected
            assert result["reward"]["groups"]["p1"]["std"] == 0
            state = new_state(c, small_model())
            rows = nft_round(state)
            p = dict(
                zip(
                    result["reward"]["sample_ids"],
                    result["reward"]["probabilities"],
                    strict=True,
                )
            )
            before = copy.deepcopy(state.state_dict())
            with pytest.raises(ValueError, match="complete global"):
                update_joint(
                    c,
                    state.roles,
                    state.student,
                    None,
                    rows,
                    None,
                    {"0:0:0": 0.5},
                    state.student_optimizer,
                    None,
                    0,
                    0,
                )
            assert_exact(before, state.state_dict())
            update_joint(
                c,
                state.roles,
                state.student,
                None,
                rows,
                None,
                p,
                state.student_optimizer,
                None,
                0,
                0,
            )
            assert state.student_optimizer.steps == 1
        finally:
            endpoint.close()


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_explicit_cold_initialization_checks_pins_capacity_and_resets(
    tmp_path, sampling_config, mode
):
    torch.set_num_threads(1)
    source = dmd_config(mode)
    base = small_model()
    cold = new_state(source, base, stage="cold")
    # Build one real mechanism update; source fixture's clock says three planned steps.
    cond = (torch.zeros(1, 2, 1024), torch.zeros(1, 1024), torch.ones(1, 2))
    rows = [
        collect(
            cold.roles.student_field,
            source,
            [{"id": f"p{i}"}],
            [str(i)],
            0,
            cond,
            None,
            "cpu",
            stage="cold",
            generator_step=0,
        )
        for i in range(2)
    ]
    update_round(
        source,
        cold.roles,
        cold.student,
        cold.fake,
        rows,
        cold.student_optimizer,
        cold.fake_optimizer,
        0,
    )
    cold.clock.update(source)
    cold.commit_round()
    path = publish(
        tmp_path / "source",
        1,
        {"config": asdict(source)},
        cold.state_dict(),
        {},
        0,
        torch.device("cpu"),
    )
    c = joint_config(sampling_config.scorers[0], mode)
    c = replace(
        c, stages=replace(c.stages, initial_sha256=file_hash(path / "manifest.json"))
    )
    assets = {"training_checkpoints": {"cold": str(path)}}
    expected = boundary_manifest(c, assets)
    roles = Roles(copy.deepcopy(base), copy.deepcopy(base), c, stage="joint")
    assert load_boundary(c, assets, roles, expected) == (1, 2)
    assert_exact(cold.roles.student.state_dict(), roles.student.state_dict())
    assert_exact(cold.roles.fake.state_dict(), roles.fake.state_dict())
    roles.enter_joint()
    state = TrainingState(
        c,
        roles,
        roles.student_field,
        roles.fake_field,
        Clock.initial(c, 1, 2),
        initial_generator_steps=1,
        initial_fake_steps=2,
    )
    assert state.student_optimizer.steps == state.fake_optimizer.steps == 0
    assert not state.student_optimizer.optimizers[0].state
    with pytest.raises(ValueError, match="manifest bytes"):
        boundary_manifest(
            replace(c, stages=replace(c.stages, initial_sha256="9" * 64)), assets
        )
    if mode == "lora":
        with pytest.raises(ValueError, match="tuning/capacity"):
            boundary_manifest(replace(c, student=replace(c.student, rank=3)), assets)
    with pytest.raises(ValueError, match="fake capacity"):
        boundary_manifest(replace(c, fake=replace(c.fake, rank=3)), assets)
