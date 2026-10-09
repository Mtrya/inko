"""Analytic flow/DMD, role ownership, adapters and complete-round recovery."""

import copy
from dataclasses import asdict, replace

import pytest
import torch

from scripts.posttrain.export import export
from src.models.inko import Inko
from src.posttrain.adapters import (
    OutputQV,
    WeightLoRA,
    install_adapters,
    merge_student,
    score_view,
)
from src.posttrain.checkpoint import (
    assert_exact,
    inspect,
    load,
    publish,
    restore_rng,
    rng_state,
)
from src.posttrain.config import Checkpoint
from src.posttrain.engine import update_round
from src.posttrain.objectives.dmd import (
    clean_prediction,
    dmd_gradient,
    fake_loss,
    surrogate,
)
from src.posttrain.optim import RoleOptimizer
from src.posttrain.roles import Roles
from src.posttrain.rollout import PromptCursor, collect, score_inputs
from src.posttrain.training_config import (
    NFT,
    Collection,
    DynaDG,
    DynaRS,
    Fake,
    FakeProbe,
    Joint,
    Objective,
    OldPolicy,
    Optimizer,
    Rewards,
    Stages,
    Student,
    StudentEMA,
    Training,
    TrainingConfig,
    TrainingExecution,
)


@pytest.mark.parametrize(
    "estimator,expected",
    [("clean_difference", -2.7), ("teacher_residual", -2.7 / 1.9), ("score_kl", -0.3)],
)
def test_gaussian_score_direction(estimator, expected):
    # Unit-variance Gaussians with means +2 and -1 at t=1/4, z=0:
    # posterior clean means are +1.8 and -0.9. Reverse KL must move y upward.
    y = torch.zeros((1, 2), requires_grad=True)
    real = torch.full_like(y, 1.8, requires_grad=True)
    fake = torch.full_like(y, -0.9, requires_grad=True)
    g = dmd_gradient(
        y, real, fake, torch.tensor([0.25]), estimator=estimator, epsilon=0.1
    )
    surrogate(y, g).sum().backward()
    torch.testing.assert_close(y.grad, torch.full_like(y, expected / 2))
    assert real.grad is None and fake.grad is None and not g.requires_grad
    assert (y.detach() - y.grad).min() > 0


def test_affine_clean_and_fake_velocity():
    x = torch.tensor([[3.0, 3.0]])
    torch.testing.assert_close(
        clean_prediction(x, torch.tensor([0.25]), torch.full_like(x, 2)),
        torch.full_like(x, 4.5),
    )
    pred = torch.tensor([[0.0, 2.0]], requires_grad=True)
    target = torch.tensor([[3.0, 5.0]], requires_grad=True)
    fake_loss(pred, target, torch.ones_like(pred)).sum().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[-2.0, -2.0]]))
    assert target.grad is None


def small_model():
    model = Inko(
        hidden_size=16,
        num_heads=2,
        double_stream_depth=1,
        single_stream_depth=1,
        mlp_ratio=2,
    )
    torch.nn.init.normal_(model.final_layer[1].weight, std=0.1)
    # Exercise active transformer branches, not Inko's deliberately closed fresh-init gates.
    for name, parameter in model.named_parameters():
        if "modulation" in name and name.endswith("bias"):
            torch.nn.init.constant_(parameter, 0.2)
    return model


def config(mode="full", strategy="teacher_qv"):
    # Mechanism inputs are independent of the mutable coverage recipes. Changing
    # a recipe's CFG, optimizer or EMA decision must not change these contracts.
    optimizer = Optimizer(
        kind="adamw",
        lr=1e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0,
        conditioning_weight_decay=0,
        max_grad_norm=1,
        muon_lr=1e-3,
        muon_weight_decay=0,
        muon_momentum=0.95,
        schedule="constant",
        warmup_steps=0,
        minimum_ratio=0.1,
    )
    checkpoint = Checkpoint("fixture", "live")
    return TrainingConfig(
        initialization=checkpoint,
        teacher=checkpoint,
        student=Student(mode, 2, 2, "attention_mlp"),
        fake=Fake(strategy, 2, 1, 2, "selected_prediction", 0, 1, "round"),
        fake_probe=FakeProbe(0, 1),
        objective=Objective("dmd", "teacher_residual", 1e-8, 0.02, 0.98, 1),
        collection=Collection(
            prompts="fixture.jsonl",
            prompts_per_round=2,
            draws_per_prompt=1,
            height=16,
            width=16,
            rule="renoise",
            steps=2,
            guidance=1,
            shift="shape",
            selected_pmf=(0.5, 0.5),
            text_exit_layer=1,
        ),
        student_optimizer=optimizer,
        fake_optimizer=optimizer,
        execution=TrainingExecution(
            learner_devices=(0,),
            microbatch=1,
            cpu_threads=1,
            collective_timeout=30,
            startup_timeout=30,
            round_timeout=30,
            precision="float32",
            activation_checkpointing=False,
            intra_host_roce=False,
            deterministic_algorithms=True,
        ),
        training=Training(7, 3, 1, (StudentEMA("ema", 0.9, "learner"),)),
        scorers=(),
        rewards=Rewards((), 1e-4, "mean_one_exp", 1),
        dyna_dg=DynaDG(False, 3, 1),
        dyna_rs=DynaRS(False, 3, 4, 1),
        stages=Stages(3, 0, ""),
        joint=Joint(0, 0, 0),
        nft=NFT("clean_adaptive", "group", 5, 1, 1e-5, 1, 1, "uniform", 0.02, 0.98),
        old_policy=OldPolicy("copy", 0.9, 0, 0.9, 3),
    )


@pytest.mark.parametrize("targets,count", [("attention", 6), ("attention_mlp", 12)])
def test_student_targets_zero_init_and_merge(targets, count):
    torch.manual_seed(4)
    model = small_model()
    original = copy.deepcopy(model)
    names = install_adapters(model, kind="weight", rank=2, scale=3, targets=targets)
    assert len(names) == count
    args = (
        torch.randn(1, 16, 2, 2),
        torch.tensor([0.4]),
        torch.randn(1, 3, 1024),
        torch.randn(1, 1024),
        torch.ones(1, 3),
    )
    torch.testing.assert_close(model(*args), original(*args), rtol=0, atol=0)
    for m in model.modules():
        if isinstance(m, WeightLoRA):
            torch.nn.init.normal_(m.b.weight, std=0.03)
    torch.testing.assert_close(
        model(*args), merge_student(model)(*args), rtol=2e-5, atol=2e-6
    )
    assert all(
        ".a." in n or ".b." in n for n, p in model.named_parameters() if p.requires_grad
    )


def test_qv_placement_and_exception_restore():
    base = torch.nn.Linear(4, 12)
    adapter = OutputQV(base, 2, 2)
    model = torch.nn.Sequential(adapter)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(adapter(x), base(x), rtol=0, atol=0)
    torch.nn.init.ones_(adapter.qb.weight)
    torch.nn.init.ones_(adapter.vb.weight)
    q, k, v = adapter(x).chunk(3, -1)
    oq, ok, ov = base(x).chunk(3, -1)
    torch.testing.assert_close(k, ok, rtol=0, atol=0)
    torch.testing.assert_close(q, oq + 2 * adapter.qb(adapter.qa(oq)))
    torch.testing.assert_close(v, ov + 2 * adapter.vb(adapter.va(ov)))
    with pytest.raises(RuntimeError), score_view(model, 0):
        torch.testing.assert_close(adapter(x), base(x), rtol=0, atol=0)
        with score_view(model, 1):
            pass
    assert adapter.view_scale == 1
    with pytest.raises(ValueError, match="export"):
        merge_student(model)


@pytest.mark.parametrize(
    "mode,strategy,target",
    [("full", "teacher_qv", "selected_prediction"), ("lora", "full", "terminal")],
)
def test_round_roles_and_resume(tmp_path, mode, strategy, target):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    c = config(mode, strategy)
    c = replace(c, fake=replace(c.fake, target=target))
    base = small_model()
    roles = Roles(copy.deepcopy(base), copy.deepcopy(base), c)
    so = RoleOptimizer(roles.student, c.student_optimizer, 3)
    fo = RoleOptimizer(roles.fake, c.fake_optimizer, 6)
    frozen = {
        n: p.clone() for n, p in roles.teacher.named_parameters() if not p.requires_grad
    }
    cond = (torch.randn(1, 3, 1024), torch.randn(1, 1024), torch.ones(1, 3))

    def round_at(step):
        rows = [
            collect(
                roles.student_field,
                c,
                [],
                [str(i)],
                step,
                cond,
                None,
                "cpu",
                stage="cold",
                generator_step=step,
            )
            for i in range(2)
        ]
        return update_round(
            c, roles, roles.student_field, roles.fake_field, rows, so, fo, step
        )

    initial = copy.deepcopy(roles.student.state_dict())
    round_at(0)
    assert so.steps == 1 and fo.steps == 2
    assert any(
        not torch.equal(v, roles.student.state_dict()[n]) for n, v in initial.items()
    )
    assert all(
        torch.equal(v, dict(roles.teacher.named_parameters())[n])
        for n, v in frozen.items()
    )
    saved = {
        "student": copy.deepcopy(roles.student.state_dict()),
        "fake": copy.deepcopy(roles.fake.state_dict()),
        "so": copy.deepcopy(so.state_dict()),
        "fo": copy.deepcopy(fo.state_dict()),
        "ema": copy.deepcopy(roles.ema),
    }
    path = publish(
        tmp_path, 1, {"case": mode}, saved, {"cursor": 3}, 0, torch.device("cpu")
    )
    expected_evidence = round_at(1)
    expected = copy.deepcopy(roles.student.state_dict())
    state, rank = load(path, {"case": mode}, 0)
    roles.student.load_state_dict(state["student"])
    roles.fake.load_state_dict(state["fake"])
    so.load_state_dict(state["so"])
    fo.load_state_dict(state["fo"])
    roles.ema = state["ema"]
    restore_rng(rank["rng"], torch.device("cpu"))
    assert_exact(rank["rng"], rng_state(torch.device("cpu")))
    assert round_at(1) == expected_evidence
    assert_exact(expected, roles.student.state_dict())
    with pytest.raises(ValueError, match="contract"):
        inspect(path, {"case": "changed"})


def test_nonfinite_optimizer_state_rejects_whole_role():
    model = small_model()
    c = config()
    opt = RoleOptimizer(model, replace(c.student_optimizer, kind="muon_adamw"), 3)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = copy.deepcopy(model.state_dict())
    param = opt.optimizers[-1].param_groups[0]["params"][0]
    opt.optimizers[-1].state[param]["poison"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError):
        opt.step(torch.tensor(1.0))
    assert_exact(before, model.state_dict())
    assert opt.steps == 0


def test_cursor_and_training_noise_recovery():
    cursor = PromptCursor(5, 123)
    cursor.take(4)
    state = cursor.state_dict()
    expected = cursor.take(7)
    cursor.load_state_dict(state)
    assert cursor.take(7) == expected
    c = config()
    roles = Roles(small_model(), small_model(), c)
    cond = (torch.randn(1, 2, 1024), torch.randn(1, 1024), torch.ones(1, 2))
    a = collect(
        roles.student_field,
        c,
        [],
        ["0"],
        0,
        cond,
        None,
        "cpu",
        stage="cold",
        generator_step=0,
    )
    b = collect(
        roles.student_field,
        c,
        [],
        ["0"],
        1,
        cond,
        None,
        "cpu",
        stage="cold",
        generator_step=1,
    )
    assert a.seeds != b.seeds
    t, e = score_inputs(c, a, "fake", 0)
    u, f = score_inputs(c, a, "fake", 1)
    assert not torch.equal(e, f) and not torch.equal(t, u)


def test_checkpoint_unpublished_failure(tmp_path, monkeypatch):
    import src.posttrain.checkpoint as module

    def fail(*args, **kwargs):
        raise OSError("publication fault")

    monkeypatch.setattr(module, "atomic_json", fail)
    with pytest.raises(OSError):
        publish(tmp_path, 1, {}, {}, {}, 0, torch.device("cpu"))
    assert not (tmp_path / "step-000001").exists()


def test_manifest_in_staging_is_not_a_published_checkpoint(tmp_path):
    device = torch.device("cpu")
    committed = publish(tmp_path, 1, {"case": "publication"}, {}, {}, 0, device)
    with pytest.raises(ValueError, match="overwrite"):
        publish(tmp_path, 1, {"case": "publication"}, {}, {}, 0, device)
    staging = next(tmp_path.glob(".incomplete-*"))
    assert (staging / "manifest.json").exists()
    with pytest.raises(ValueError, match="unpublished"):
        inspect(staging)
    assert inspect(committed)["step"] == 1


def test_role_schedule_uses_its_successful_step_clock():
    c = config()
    optimizer = RoleOptimizer(
        small_model(),
        replace(
            c.student_optimizer, schedule="cosine", warmup_steps=2, minimum_ratio=0.2
        ),
        5,
    )
    for step, expected in enumerate([0.5, 1.0, 1.0, 0.6, 0.2]):
        optimizer.steps = step
        assert optimizer.rate_factor() == pytest.approx(expected)


def test_checkpointed_blocks_preserve_forward_and_gradients():
    torch.manual_seed(7)
    a = small_model()
    b = copy.deepcopy(a)
    b.activation_checkpointing = True
    args = (
        torch.randn(1, 16, 2, 2),
        torch.tensor([0.4]),
        torch.randn(1, 3, 1024),
        torch.randn(1, 1024),
        torch.ones(1, 3),
    )
    for model in (a, b):
        model(*args).square().sum().backward()
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)


def test_config_rejects_invalid_cold_objectives_and_uneven_rounds():
    c = config()
    with pytest.raises(ValueError, match="cold objective"):
        replace(c, objective=replace(c.objective, mode="nft"))
    with pytest.raises(ValueError, match="evenly"):
        replace(c, execution=replace(c.execution, learner_devices=(0, 1, 2)))
    with pytest.raises(ValueError, match="PMF"):
        replace(c, collection=replace(c.collection, selected_pmf=(0.1, 0.2)))


@pytest.mark.parametrize(
    "mode,view", [("full", "live"), ("full", "ema"), ("lora", "live"), ("lora", "ema")]
)
def test_export_actual_student(tmp_path, mode, view):
    c = config(mode)
    roles = Roles(small_model(), small_model(), c)
    with torch.no_grad():
        for p in roles.student.parameters():
            if p.requires_grad:
                p.add_(0.01)
    state = {
        "student": roles.student.state_dict(),
        "ema": roles.ema,
        "architecture": roles.student.get_config(),
        "step": 2,
        "ownership": roles.ownership(),
    }
    path = publish(
        tmp_path / "checkpoints",
        2,
        {"config": asdict(c)},
        state,
        {},
        0,
        torch.device("cpu"),
    )
    export(path, view, tmp_path / "export")
    model = Inko.from_single_file(str(tmp_path / "export" / "weights.pt"))
    if view == "ema":
        with torch.no_grad():
            for n, p in roles.student.named_parameters():
                if n in roles.ema["ema"]:
                    p.copy_(roles.ema["ema"][n])
    assert_exact(merge_student(roles.student).state_dict(), model.state_dict())
