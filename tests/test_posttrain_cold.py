"""Cold reward/replay/schedule contracts; no generation-quality expectations."""

import copy
import itertools
import math
from dataclasses import replace

import pytest
import torch
from PIL import Image
from test_posttrain_dmd import config as dmd_config
from test_posttrain_dmd import small_model
from test_posttrain_scoring import Endpoint

from src.posttrain.adapters import OutputQV, score_view
from src.posttrain.checkpoint import assert_exact, load, publish
from src.posttrain.engine import update_round
from src.posttrain.objectives.rtdm import verify_replay, weighted_losses
from src.posttrain.optim import RoleOptimizer
from src.posttrain.records import file_hash
from src.posttrain.rewards import transform_round
from src.posttrain.roles import Roles
from src.posttrain.rollout import collect, score_inputs
from src.posttrain.round_scoring import acquire_round
from src.posttrain.schedules import cold_point, physical_times
from src.posttrain.scoring.contracts import Channel
from src.posttrain.scoring.lifecycle import Services
from src.posttrain.training_config import DynaDG, DynaRS, RewardChannel, Rewards


def rewarded(config, specs, channels, weighting="mean_one_exp"):
    return replace(
        config,
        objective=replace(config.objective, mode="rtdm"),
        scorers=specs,
        rewards=Rewards(channels, 1e-4, weighting, 1),
    )


def rows(values, service="unfamiliar", channel="fit"):
    return [
        {
            "sample_id": str(i),
            "scorer": service,
            "status": "ok",
            "values": {channel: value},
        }
        for i, value in enumerate(values)
    ]


def test_population_orientation_mixture_and_constant_channel(sampling_config):
    spec = replace(
        sampling_config.scorers[0],
        outputs=(
            Channel("fit", "higher"),
            Channel("cost", "lower"),
        ),
    )
    c = rewarded(
        dmd_config(),
        (spec,),
        (
            RewardChannel("unfamiliar.fit", "higher", 0.25, "round"),
            RewardChannel("unfamiliar.cost", "lower", 0.75, "round"),
        ),
    )
    records = rows([1, 2, 3])
    for r in records:
        r["values"]["cost"] = 2
    result = transform_round(["0", "1", "2"], records[::-1], c)
    std = math.sqrt(2 / 3)
    expected = torch.tensor([-0.25 / (std + 1e-4), 0, 0.25 / (std + 1e-4)])
    torch.testing.assert_close(torch.tensor(result["combined"]), expected)
    assert result["statistics"]["unfamiliar.fit"]["mean"] == 2
    assert result["statistics"]["unfamiliar.fit"]["std"] == pytest.approx(std)
    assert result["statistics"]["unfamiliar.cost"]["transformed"] == [0, 0, 0]
    assert result["raw"]["unfamiliar.cost"] == [2, 2, 2]
    assert transform_round(["2", "0", "1"], records, c) == result


def test_raw_orientation_exponential_and_paper_weights(sampling_config):
    spec = replace(
        sampling_config.scorers[0], outputs=(Channel("fit", "lower"),)
    )
    c = rewarded(
        dmd_config(), (spec,), (RewardChannel("unfamiliar.fit", "lower", 1, "none"),)
    )
    result = transform_round(["0", "1", "2"], rows([0, -math.log(2), -math.log(4)]), c)
    torch.testing.assert_close(
        torch.tensor(result["weights"]), torch.tensor([3 / 7, 6 / 7, 12 / 7])
    )
    extreme = transform_round(["0", "1", "2"], rows([10000, 0, -10000]), c)
    assert extreme["weights"] == [0, 0, 3]
    spec = replace(spec, outputs=(Channel("fit", "higher"),))
    c = rewarded(
        dmd_config(),
        (spec,),
        (RewardChannel("unfamiliar.fit", "higher", 1, "none"),),
        "paper_max_exp",
    )
    result = transform_round(["0", "1", "2"], rows([0, 0.5, 1]), c)
    torch.testing.assert_close(
        torch.tensor(result["weights"]), torch.tensor([1, math.exp(0.5), math.e])
    )
    for values in ([0, 0, 0], [-0.1, 0.5, 1]):
        with pytest.raises(ValueError, match="paper-max"):
            transform_round(["0", "1", "2"], rows(values), c)


def test_missing_failed_duplicate_and_nonfinite_reward_abort(sampling_config):
    c = rewarded(
        dmd_config(),
        sampling_config.scorers,
        (RewardChannel("unfamiliar.fit", "higher", 1, "none"),),
    )
    good = rows([0.5, 0.6])
    for bad in (
        good[:1],
        good + [good[0]],
        [*good[:1], {**good[1], "status": "failed"}],
        rows([float("nan"), 0.5]),
    ):
        with pytest.raises(ValueError):
            transform_round(["0", "1"], bad, c)


def test_detached_weights_and_selected_replay_mutation():
    y = torch.tensor([1.0, 2.0], requires_grad=True)
    weights = torch.tensor([0.5, 1.5], requires_grad=True)
    weighted_losses(0.5 * y.square(), weights).mean().backward()
    torch.testing.assert_close(y.grad, torch.tensor([0.25, 1.5]))
    assert weights.grad is None
    verify_replay(y, y.detach().clone())
    with pytest.raises(RuntimeError, match="replay changed"):
        verify_replay(y + 0.001, y.detach())


def test_dynamic_clock_endpoints_and_inverse_noise_coordinate():
    c = replace(
        dmd_config(), dyna_dg=DynaDG(True, 2, 0.8), dyna_rs=DynaRS(True, 2, 4, 1)
    )
    assert [
        cold_point(c, step)["real_scale"] for step in (0, 1, 2, 3)
    ] == pytest.approx([0.8, 0.4, 0, 0])
    assert [cold_point(c, step)["alpha"] for step in (0, 1, 2, 3)] == pytest.approx(
        [4, 2.5, 1, 1]
    )
    torch.testing.assert_close(
        physical_times(torch.tensor([0.0, 0.25, 0.5, 1.0]), 0.1, 0.9),
        torch.tensor([0.9, 0.7, 0.5, 0.1]),
    )
    with pytest.raises(ValueError):
        replace(c, fake=replace(c.fake, strategy="full"))
    with pytest.raises(ValueError):
        replace(c, dyna_rs=DynaRS(True, 4, 4, 1))


def test_qv_linear_real_views_and_exception_restoration():
    base = torch.nn.Linear(2, 6, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(2).repeat(3, 1))
    model = torch.nn.Sequential(OutputQV(base, 1, 1))
    adapter = model[0]
    with torch.no_grad():
        for layer in (adapter.qa, adapter.va):
            layer.weight.fill_(1)
        for layer in (adapter.qb, adapter.vb):
            layer.weight.fill_(0.5)
    original = copy.deepcopy(base.state_dict())
    for scale, expected in (
        (0, [1, 2, 1, 2, 1, 2]),
        (0.5, [1.75, 2.75, 1, 2, 1.75, 2.75]),
        (1, [2.5, 3.5, 1, 2, 2.5, 3.5]),
    ):
        with score_view(model, scale):
            torch.testing.assert_close(
                model(torch.tensor([[1.0, 2.0]])),
                torch.tensor([expected], dtype=torch.float32),
            )
    with pytest.raises(RuntimeError, match="deliberate"), score_view(model, 0.2):
        with pytest.raises(RuntimeError, match="overlapping"), score_view(model, 0.7):
            pass
        raise RuntimeError("deliberate")
    assert adapter.view_scale == 1
    assert_exact(base.state_dict(), original)


def rollouts_for(c, roles, step):
    result = []
    for slot in range(2):
        cond = (
            torch.ones(1, 3, 1024) * (0.2 + slot),
            torch.ones(1, 1024) * 0.3,
            torch.ones(1, 3),
        )
        result.append(
            collect(
                roles.student_field,
                c,
                [{"id": str(slot)}],
                [f"{step}:{slot}:0"],
                step,
                cond,
                None,
                "cpu",
                stage="cold",
                generator_step=step,
            )
        )
    return result


@pytest.mark.parametrize("rt,dg,rs", list(itertools.product((False, True), repeat=3)))
def test_every_cold_toggle_and_committed_clock(rt, dg, rs, sampling_config, tmp_path):
    c = replace(
        dmd_config("lora" if dg else "full"),
        dyna_dg=DynaDG(dg, 1, 1),
        dyna_rs=DynaRS(rs, 1, 4, 1),
    )
    if rt:
        c = rewarded(
            c,
            sampling_config.scorers,
            (RewardChannel("unfamiliar.fit", "higher", 1, "none"),),
        )
    roles = Roles(small_model(), small_model(), c)
    so = RoleOptimizer(roles.student, c.student_optimizer, 3)
    fo = RoleOptimizer(roles.fake, c.fake_optimizer, 6)
    for step in (0, 1):
        batches = rollouts_for(c, roles, step)
        weights = {f"{step}:0:0": 0.5, f"{step}:1:0": 1.5} if rt else None
        report = update_round(
            c,
            roles,
            roles.student_field,
            roles.fake_field,
            batches,
            so,
            fo,
            step,
            weights,
        )
        assert report["selected_replay_exact"]
        assert report["cold_schedule"]["generator_step"] == step
        assert so.steps == step + 1 and fo.steps == 2 * (step + 1)
        assert all(p.grad is None for p in roles.fake.parameters())
    path = publish(
        tmp_path,
        2,
        {"input": "cold"},
        {
            "cold_schedule": cold_point(c, 2),
            "student": roles.student.state_dict(),
            "fake": roles.fake.state_dict(),
            "so": so.state_dict(),
            "fo": fo.state_dict(),
        },
        {},
        0,
        torch.device("cpu"),
    )
    saved, _ = load(path, {"input": "cold"}, 0)
    assert saved["cold_schedule"] == {
        "generator_step": 2,
        "real_scale": 0.0,
        "distribution": "beta_noise_coordinate" if rs else "uniform_physical",
        "alpha": 1.0,
        "beta": 1.0,
        "dyna_dg": {"enabled": dg, "horizon": 1, "start_scale": 1},
        "dyna_rs": {
            "enabled": rs,
            "horizon": 1,
            "alpha_start": 4,
            "beta_start": 1,
        },
    }
    assert_exact(saved["student"], roles.student.state_dict())


def test_incomplete_rewards_cannot_step_fake(sampling_config):
    c = rewarded(
        dmd_config(),
        sampling_config.scorers,
        (RewardChannel("unfamiliar.fit", "higher", 1, "none"),),
    )
    roles = Roles(small_model(), small_model(), c)
    so = RoleOptimizer(roles.student, c.student_optimizer, 3)
    fo = RoleOptimizer(roles.fake, c.fake_optimizer, 6)
    batches = rollouts_for(c, roles, 0)
    before = copy.deepcopy(roles.fake.state_dict())
    with pytest.raises(ValueError, match="complete.*global"):
        update_round(
            c,
            roles,
            roles.student_field,
            roles.fake_field,
            batches,
            so,
            fo,
            0,
            {"0:0:0": 1},
        )
    assert fo.steps == so.steps == 0
    assert_exact(before, roles.fake.state_dict())


def test_actual_selected_prediction_and_independent_time_noise_streams():
    c = replace(dmd_config(), dyna_rs=DynaRS(True, 2, 4, 1))
    roles = Roles(small_model(), small_model(), c)
    r = rollouts_for(c, roles, 0)[0]
    t, e = score_inputs(c, r, "fake", 0)
    u, f = score_inputs(c, r, "fake", 1)
    assert not torch.equal(t, u) and not torch.equal(e, f)
    assert score_inputs(c, r, "fake", 0)[0].equal(t)
    assert cold_point(c, 0)["alpha"] == 4
    with torch.no_grad():
        next(p for p in roles.student.parameters() if p.requires_grad).add_(1)
    from src.posttrain.objectives.dmd import clean_prediction

    prediction = clean_prediction(
        r.selected_input,
        r.selected_time,
        roles.student_field(r.selected_input, r.selected_time, r.conditioning),
    )
    with pytest.raises(RuntimeError, match="replay changed"):
        verify_replay(prediction, r.selected_prediction)


def test_config_only_http_scorer_substitution_through_cold_update(
    tmp_path, sampling_config
):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)
    for identifier, metric, direction, expected in (
        ("alien", "rubric_fit", "higher", [2 / 3, 4 / 3]),
        ("other", "cost", "lower", [4 / 3, 2 / 3]),
    ):
        spec = replace(
            sampling_config.scorers[0],
            id=identifier,
            outputs=(Channel(metric, direction),),
        )

        class Backend:
            def score(self, inputs, channel=metric):
                return [
                    {channel: math.log(2) if i.prompt == "second" else 0}
                    for i in inputs
                ]

        endpoint = Endpoint(tmp_path / identifier, spec, Backend())
        try:
            c = rewarded(
                dmd_config(),
                (spec,),
                (RewardChannel(identifier + "." + metric, direction, 1, "none"),),
            )
            records = [
                {
                    "sample_id": f"0:{slot}:0",
                    "prompt": {"text": text},
                    "image_path": str(image),
                    "image_sha256": file_hash(image),
                }
                for slot, text in enumerate(("first", "second"))
            ]
            with Services(
                c.scorers,
                {identifier: {"endpoints": [endpoint.url], "launches": []}},
                tmp_path / identifier / "caller",
                c.execution.learner_devices,
                10,
            ) as services:
                result = acquire_round(c, services.clients, records, stage="cold")
            torch.testing.assert_close(
                torch.tensor(result["reward"]["weights"]), torch.tensor(expected)
            )
            roles = Roles(small_model(), small_model(), c)
            so = RoleOptimizer(roles.student, c.student_optimizer, 3)
            fo = RoleOptimizer(roles.fake, c.fake_optimizer, 6)
            weights = dict(
                zip(
                    result["reward"]["sample_ids"],
                    result["reward"]["weights"],
                    strict=True,
                )
            )
            report = update_round(
                c,
                roles,
                roles.student_field,
                roles.fake_field,
                rollouts_for(c, roles, 0),
                so,
                fo,
                0,
                weights,
            )
            assert report["selected_replay_exact"] and so.steps == 1
        finally:
            endpoint.close()


def test_concurrent_score_scope_and_fake_forward_are_rejected():
    import threading

    model = torch.nn.Sequential(OutputQV(torch.nn.Linear(2, 6, bias=False), 1, 1))
    failures = []

    def competing():
        for action in (
            lambda: model(torch.ones(1, 2)),
            lambda: score_view(model, 0.7).__enter__(),
        ):
            try:
                action()
            except RuntimeError as exc:
                failures.append(str(exc))

    with score_view(model, 0.5):
        thread = threading.Thread(target=competing)
        thread.start()
        thread.join()
        assert model[0].view_scale == 0.5
    assert len(failures) == 2 and model[0].view_scale == 1


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cold_dynamic_optimizer_and_next_input_recovery(
    tmp_path, sampling_config, mode
):
    c = rewarded(
        replace(
            dmd_config(mode),
            dyna_dg=DynaDG(True, 2, 0.8),
            dyna_rs=DynaRS(True, 2, 4, 1),
        ),
        sampling_config.scorers,
        (RewardChannel("unfamiliar.fit", "higher", 1, "none"),),
    )
    initial_student, initial_teacher = small_model(), small_model()
    roles = Roles(copy.deepcopy(initial_student), copy.deepcopy(initial_teacher), c)
    so = RoleOptimizer(roles.student, c.student_optimizer, 3)
    fo = RoleOptimizer(roles.fake, c.fake_optimizer, 6)

    def advance(roles, so, fo, step):
        return update_round(
            c,
            roles,
            roles.student_field,
            roles.fake_field,
            rollouts_for(c, roles, step),
            so,
            fo,
            step,
            {f"{step}:0:0": 0.5, f"{step}:1:0": 1.5},
        )

    advance(roles, so, fo, 0)
    path = publish(
        tmp_path,
        1,
        {"input": "cold-recovery"},
        {
            "student": roles.student.state_dict(),
            "fake": roles.fake.state_dict(),
            "ema": roles.ema,
            "so": so.state_dict(),
            "fo": fo.state_dict(),
            "step": 1,
            "cold_schedule": cold_point(c, 1),
        },
        {},
        0,
        torch.device("cpu"),
    )
    expected = advance(roles, so, fo, 1)
    restored = Roles(copy.deepcopy(initial_student), copy.deepcopy(initial_teacher), c)
    new_so = RoleOptimizer(restored.student, c.student_optimizer, 3)
    new_fo = RoleOptimizer(restored.fake, c.fake_optimizer, 6)
    saved, _ = load(path, {"input": "cold-recovery"}, 0)
    restored.student.load_state_dict(saved["student"])
    restored.fake.load_state_dict(saved["fake"])
    restored.ema = saved["ema"]
    new_so.load_state_dict(saved["so"])
    new_fo.load_state_dict(saved["fo"])
    actual = advance(restored, new_so, new_fo, saved["step"])
    assert actual == expected
    assert_exact(roles.student.state_dict(), restored.student.state_dict())
    assert_exact(roles.fake.state_dict(), restored.fake.state_dict())
    assert_exact(so.state_dict(), new_so.state_dict())
    assert_exact(fo.state_dict(), new_fo.state_dict())
    assert_exact(roles.ema, restored.ema)


def test_cfg_selected_replay_retains_both_conditioning_branches():
    from src.posttrain.objectives.dmd import clean_prediction

    c = replace(dmd_config(), collection=replace(dmd_config().collection, guidance=1.5))
    roles = Roles(small_model(), small_model(), c)
    conditioning = (
        torch.ones(1, 3, 1024) * 0.2,
        torch.ones(1, 1024) * 0.3,
        torch.ones(1, 3),
    )
    empty = (torch.zeros(1, 3, 1024), torch.zeros(1, 1024), torch.ones(1, 3))
    r = collect(
        roles.student_field,
        c,
        [{"id": "prompt"}],
        ["0:0:0"],
        0,
        conditioning,
        empty,
        "cpu",
        stage="cold",
        generator_step=0,
    )
    predicted = clean_prediction(
        r.selected_input,
        r.selected_time,
        roles.student_field(
            r.selected_input, r.selected_time, r.conditioning, r.unconditional, 1.5
        ),
    )
    verify_replay(predicted, r.selected_prediction)
    wrong = clean_prediction(
        r.selected_input,
        r.selected_time,
        roles.student_field(r.selected_input, r.selected_time, r.conditioning),
    )
    with pytest.raises(RuntimeError, match="replay changed"):
        verify_replay(wrong, r.selected_prediction)
    predicted.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.count_nonzero()
        for p in roles.student.parameters()
    )


def test_real_view_cannot_enter_during_an_existing_fake_forward():
    import threading

    from src.posttrain.roles import Velocity

    entered, release = threading.Event(), threading.Event()

    class Blocked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qv = OutputQV(torch.nn.Linear(2, 6, bias=False), 1, 1)

        def forward(self, x, t):
            entered.set()
            assert release.wait(5), "test forward gate expired"
            return self.qv(x)

    model = Blocked()
    field = Velocity(model, "float32")
    failures = []

    def forward():
        try:
            field(torch.ones(1, 2), torch.ones(1), ())
        except (RuntimeError, AssertionError) as exc:
            failures.append(exc)

    thread = threading.Thread(target=forward)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="overlapping"), score_view(model, 0.5):
            pass
    finally:
        release.set()
        thread.join()
    assert not failures


def test_shared_score_forward_admission_and_exception_restoration():
    from src.posttrain.roles import Velocity

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qv = OutputQV(torch.nn.Linear(2, 6, bias=False), 1, 1)
            self.fail = False

        def forward(self, x, t):
            if self.fail:
                raise ValueError("deliberate forward failure")
            return self.qv(x)

    model = Model()
    fake, real = Velocity(model, "float32"), Velocity(model, "float32", real=True)
    x, t = torch.ones(1, 2), torch.ones(1)
    fake(x, t, ())
    with pytest.raises(RuntimeError, match="role"):
        real(x, t, ())
    with score_view(model, 0.5):
        real(x, t, ())
        with pytest.raises(RuntimeError, match="role"):
            fake(x, t, ())
    fake(x, t, ())
    model.fail = True
    with pytest.raises(ValueError, match="deliberate"):
        fake(x, t, ())
    model.fail = False
    with score_view(model, 0):
        real(x, t, ())
    assert model.qv.view_scale == 1
