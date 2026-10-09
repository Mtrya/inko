"""Campaign state and readout contracts. These tests make no quality claim."""

import copy
import json
from dataclasses import asdict, replace

import pytest
import torch
from test_posttrain_dmd import config, small_model

from src.evaluation.benchmarks.diversity import grouped_pairs, summarize
from src.posttrain.checkpoint import assert_exact, load, publish
from src.posttrain.engine import fake_updates, update_round
from src.posttrain.evaluation_config import (
    IntervalEvaluation,
    boundaries,
    evaluation_due,
)
from src.posttrain.probes import fake_regression_probe
from src.posttrain.roles import Roles
from src.posttrain.rollout import collect
from src.posttrain.session import TrainingState
from src.posttrain.stages import Clock
from src.posttrain.training_config import FakeProbe, StudentEMA


@pytest.mark.parametrize("fit_stop", [None, 2])
def test_launcher_stop_arguments_are_accepted_by_learner_cli(
    tmp_path, monkeypatch, fit_stop
):
    import sys

    from src.posttrain import learner, train

    c = config()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {"id": "p", "scene_id": "s", "text": "A tree", "seed": 1, "aspect": "1:1"}
        )
        + "\n"
    )
    c = replace(
        c,
        fake=replace(c.fake, initial_steps=3),
        collection=replace(c.collection, prompts=str(prompts)),
    )
    config_path, assets_path = tmp_path / "config.json", tmp_path / "assets.json"
    config_path.write_text(json.dumps(asdict(c)))
    assets = {"learner": {"python_paths": []}, "device_type": "cpu"}
    assets_path.write_text(json.dumps(assets))
    received = []
    monkeypatch.setattr(learner, "learner", lambda *args: received.append(args))

    class AcceptedLearner:
        returncode = 0

        def __init__(self, argv, **kwargs):
            with monkeypatch.context() as child:
                child.setattr(sys, "argv", argv[2:])
                train.main()

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(train.subprocess, "Popen", AcceptedLearner)
    result = train.launch(
        c,
        assets,
        {"boundary": None},
        tmp_path / "run",
        None,
        c.training.steps,
        config_path,
        assets_path,
        fit_stop,
    )
    assert len(received) == len(c.execution.learner_devices)
    assert all(args[-1] == fit_stop for args in received)
    assert all(args[6] == c.training.steps for args in received)
    assert result["completed_step"] == (c.training.steps if fit_stop is None else 0)


def state_for(c, base):
    roles = Roles(copy.deepcopy(base), copy.deepcopy(base), c)
    return TrainingState(
        c, roles, roles.student_field, roles.fake_field, Clock.initial(c)
    )


@pytest.mark.parametrize(
    "startup", ["empty", "staging", "published", "changed_contract"]
)
def test_launcher_restart_requires_resume_only_after_checkpoint_publication(
    tmp_path, monkeypatch, startup
):
    from src.posttrain import train

    c = config()
    root = tmp_path / "training"
    root.mkdir()
    contract = {"boundary": None}
    recorded = contract if startup != "changed_contract" else {"boundary": "changed"}
    (root / "manifest.json").write_text(json.dumps(recorded))
    if startup in {"staging", "published"}:
        checkpoint = (
            root
            / "checkpoints"
            / (".uncommitted" if startup == "staging" else "step-000000")
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text("{}")
    started = []

    class CompletedLearner:
        returncode = 0

        def __init__(self, argv, **kwargs):
            started.append(argv)

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(train.subprocess, "Popen", CompletedLearner)

    def restart():
        return train.launch(
            c,
            {"learner": {"python_paths": []}, "device_type": "cpu"},
            contract,
            root,
            None,
            c.training.steps,
            tmp_path / "config.json",
            tmp_path / "assets.json",
        )

    if startup in {"empty", "staging"}:
        restart()
        assert len(started) == len(c.execution.learner_devices)
        assert all("--resume" not in argv for argv in started)
    else:
        error = "explicit --resume" if startup == "published" else "contract changed"
        with pytest.raises(ValueError, match=error):
            restart()
        assert started == []
    assert json.loads((root / "manifest.json").read_text()) == recorded


def collection(state):
    rows = []
    for slot in range(2):
        rows.append(
            collect(
                state.roles.student_field,
                state.config,
                [{"id": f"p{slot}"}],
                [f"0:{slot}:0"],
                0,
                (torch.zeros(1, 2, 1024), torch.zeros(1, 1024), torch.ones(1, 2)),
                None,
                "cpu",
                stage="cold",
                generator_step=0,
            )
        )
    return rows


def fit_one(state, rows):
    evidence = fake_updates(
        state.config,
        state.roles,
        state.fake,
        rows,
        state.fake_optimizer,
        count=1,
        start=state.clock.initial_fit_steps,
    )
    state.clock.fit(state.config)
    return evidence


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_initial_fit_recovery_keeps_entry_student_and_counts_total_updates(
    tmp_path, mode
):
    c = config(mode)
    c = replace(c, fake=replace(c.fake, initial_steps=3))
    base = small_model()
    state = state_for(c, base)
    rows = collection(state)
    student = copy.deepcopy(state.roles.student.state_dict())
    ema = copy.deepcopy(state.roles.ema)
    with pytest.raises(ValueError, match="stage boundary"):
        state.clock.update(c)
    first = fit_one(state, rows)
    contract = json.loads(json.dumps({"config": asdict(c)}))
    path = publish(
        tmp_path,
        0,
        contract,
        state.state_dict(),
        {"cursor": "unchanged"},
        0,
        torch.device("cpu"),
    )
    assert path.name == "fit-000001"
    saved, rank_saved = load(path, contract, 0)
    recovered = state_for(c, base)
    recovered.load_state_dict(saved)
    assert rank_saved["cursor"] == "unchanged"
    for _ in range(2):
        a, b = fit_one(state, rows), fit_one(recovered, rows)
        assert_exact(a, b)
    assert_exact(state.state_dict(), recovered.state_dict())
    assert_exact(student, state.roles.student.state_dict())
    assert_exact(ema, state.roles.ema)
    assert state.clock.state_dict() == {
        "stage": "cold",
        "step": 0,
        "stage_generator_steps": 0,
        "global_generator_steps": 0,
        "fake_steps": 3,
        "initial_fit_steps": 3,
        "committed_rounds": 0,
        "joint_rounds": 0,
        "old_policy_version": None,
        "samples_used": 0,
        "samples_scored": 0,
        "old_refresh_decay": None,
    }
    assert state.student_optimizer.steps == 0
    assert first["inputs"][0]["update"] == 0
    evidence = update_round(
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
    assert evidence["fake_losses"] == []
    assert state.clock.fake_steps == state.fake_optimizer.steps == 3
    assert state.student_optimizer.steps == 1
    state.state_dict()  # Reject a clock/optimizer disagreement at publication.


def test_named_emas_have_independent_recurrences_and_exact_recovery():
    c = config("lora")
    c = replace(
        c,
        training=replace(
            c.training,
            ema=(StudentEMA("short", 0.5, "learner"), StudentEMA("long", 0.75, "cpu")),
        ),
    )
    base = small_model()
    state = state_for(c, base)
    entry = copy.deepcopy(state.roles.ema["short"])
    for increment in (2, 2):
        with torch.no_grad():
            for p in state.roles.student.parameters():
                if p.requires_grad:
                    p.add_(increment)
        state.roles.update_ema()
    for name, initial in entry.items():
        torch.testing.assert_close(
            state.roles.ema["short"][name], initial + 2.5, rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            state.roles.ema["long"][name], initial + 1.375, rtol=0, atol=1e-6
        )
        assert state.roles.ema["short"][name] is not state.roles.ema["long"][name]
    restored = state_for(c, base)
    restored.roles.load_state_dict(state.roles.state_dict())
    assert_exact(state.roles.state_dict(), restored.roles.state_dict())


def test_fresh_probe_changes_neither_roles_nor_global_rng():
    c = replace(config(), fake_probe=FakeProbe(1, 2))
    state = state_for(c, small_model())
    rows = collection(state)
    before = copy.deepcopy(state.state_dict())
    rng = torch.get_rng_state().clone()
    result = fake_regression_probe(c, state.roles, rows, 0)
    assert result["samples"] == 4
    assert set(result) == {
        "samples",
        "student_version",
        "velocity_mse",
        "score_weighted_mse",
        "scope",
    }
    assert_exact(before, state.state_dict())
    assert torch.equal(rng, torch.get_rng_state())


def test_muon_routes_actual_adapters_and_zero_gradient_has_no_first_movement():
    from src.optim.muon import Muon

    c = config("lora")
    c = replace(
        c,
        student_optimizer=replace(c.student_optimizer, kind="muon_adamw"),
        fake_optimizer=replace(c.fake_optimizer, kind="muon_adamw"),
    )
    state = state_for(c, small_model())
    for optimizer, model in (
        (state.student_optimizer, state.roles.student),
        (state.fake_optimizer, state.roles.fake),
    ):
        assert len(optimizer.optimizers) == 1 and isinstance(
            optimizer.optimizers[0], Muon
        )
        owned = [
            p for group in optimizer.optimizers[0].param_groups for p in group["params"]
        ]
        assert {id(p) for p in owned} == {
            id(p) for p in model.parameters() if p.requires_grad
        }
        initial = [p.detach().clone() for p in owned]
        for p in owned:
            p.grad = torch.zeros_like(p)
        optimizer.step(torch.tensor(0.0))
        for p, before in zip(owned, initial, strict=True):
            assert torch.equal(p, before)
        for p in owned:
            p.grad = torch.ones_like(p)
        optimizer.step(torch.tensor(1.0))
        assert all(
            not torch.equal(p, before) for p, before in zip(owned, initial, strict=True)
        )


def test_interval_stops_are_complete_rounds_and_final_evaluation_is_explicit():
    c = replace(
        config(),
        stages=replace(config().stages, cold_steps=5),
        training=replace(config().training, steps=5),
    )
    interval = IntervalEvaluation(
        "sampling.toml", 2, ("monitor",), ("live", "ema"), False, 7
    )
    assert list(boundaries(c, interval)) == [2, 4, 5]
    assert [s for s in (0, 1, 2, 3, 4, 5) if evaluation_due(c, interval, s)] == [
        2,
        4,
        5,
    ]
    with pytest.raises(ValueError, match="absent EMA"):
        list(boundaries(c, replace(interval, weights=("unknown",))))


@pytest.mark.parametrize("direction,disagreements", [("higher", 1), ("lower", 0)])
def test_blind_join_uses_channel_direction_and_rejects_unfilled_ballots(
    tmp_path, sampling_config, direction, disagreements
):
    from PIL import Image

    from scripts.posttrain.readouts import join, panel
    from src.posttrain.records import atomic_json, file_hash
    from src.posttrain.scoring.client import request_for_record
    from src.posttrain.scoring.contracts import ScoreResult, score_record

    spec = replace(
        sampling_config.scorers[0],
        outputs=(replace(sampling_config.scorers[0].outputs[0], direction=direction),),
    )
    c = replace(sampling_config, scorers=(spec,))
    run = tmp_path / "benchmark"
    atomic_json(run / "manifest.json", {"config": asdict(c)})
    for index, arm in enumerate(("reference", "candidate")):
        image = tmp_path / f"{arm}.png"
        Image.new("RGB", (32, 32), (index * 100, 20, 30)).save(image)
        record = {
            "sample_id": arm,
            "arm": arm,
            "prompt": {
                "id": "p",
                "scene_id": "s",
                "text": "fixture",
                "seed": 1,
                "aspect": "1:1",
            },
            "image_path": str(image),
            "image_sha256": file_hash(image),
        }
        atomic_json(run / "records" / f"{arm}.json", record)
        score = ScoreResult.for_request(
            request_for_record(record, spec), values={"fit": 0.2 if index == 0 else 0.8}
        )
        atomic_json(
            run / "scores" / spec.id / f"{arm}.json", score_record(spec.id, score, 1)
        )
    blind = tmp_path / "blind"
    panel(run, blind, 7)
    key = json.loads((blind / "answer_key.json").read_text())
    with pytest.raises(ValueError, match="empty"):
        join(run, blind / "answer_key.json", blind / "ballot.json")
    winner = next(label for label, arm in key["p"].items() if arm == "reference")
    atomic_json(blind / "ballot.json", {"p": winner})
    panel(run, blind, 7)  # Restart must preserve an already completed ballot.
    result = join(run, blind / "answer_key.json", blind / "ballot.json")
    metric = result["comparisons"]["candidate"]["unfamiliar.fit"]
    assert metric["complete"] and metric["human_pair_comparable"] == 1
    assert metric["scorer_up_human_down"] == disagreements
    assert metric["rows"][0]["raw_delta"] == pytest.approx(0.6)
    assert json.loads((blind / "ballot.json").read_text()) == {"p": winner}


def test_interval_recovery_evaluation_failure_blocks_further_training(
    tmp_path, monkeypatch
):
    from scripts.posttrain import run_campaign
    from src.posttrain.records import atomic_json

    source = tmp_path / "config.json"
    atomic_json(source, asdict(config()))
    assets = tmp_path / "assets.json"
    atomic_json(assets, {})
    sampling = tmp_path / "sampling.json"
    interval = tmp_path / "interval.json"
    atomic_json(
        interval,
        asdict(
            IntervalEvaluation(
                str(sampling),
                2,
                ("bringup",),
                ("live", "ema"),
                True,
                7,
            )
        ),
    )
    committed = tmp_path / "committed"
    monkeypatch.setattr(
        run_campaign, "committed_checkpoints", lambda root: [(2, 0, committed)]
    )
    monkeypatch.setattr(
        run_campaign, "inspect", lambda path: {"contract": {"fixture": "contract"}}
    )
    monkeypatch.setattr(run_campaign, "load_prompts", lambda *args: [{"id": "p"}])
    monkeypatch.setattr(
        "src.posttrain.provenance.training_contract",
        lambda *args: {"fixture": "contract"},
    )
    training_calls = []
    monkeypatch.setattr(
        run_campaign.subprocess,
        "run",
        lambda *args, **kwargs: training_calls.append(args),
    )

    def failed_evaluation(*args, **kwargs):
        raise RuntimeError("fixture scorer failure")

    monkeypatch.setattr(run_campaign, "evaluate_checkpoint", failed_evaluation)
    with pytest.raises(RuntimeError, match="fixture scorer failure"):
        run_campaign.execute(source, interval, assets, tmp_path / "campaign")
    assert training_calls == []
    assert not (tmp_path / "campaign" / "complete.json").exists()


def test_diversity_pairs_only_exact_caption_seeds_and_equal_scene_aggregation():
    records = []
    for text, lang, seeds in (("same", "en", (1, 2, 3)), ("translated", "zh", (1, 2))):
        for seed in seeds:
            records.append(
                {
                    "sample_id": f"{lang}-{seed}",
                    "arm": "a",
                    "prompt": {
                        "scene_id": "scene",
                        "text": text,
                        "lang": lang,
                        "aspect": "1:1",
                        "seed": seed,
                    },
                }
            )
    groups = list(grouped_pairs(records))
    assert [len(g[3]) for g in groups] == [3, 1]
    for _, _, _, pairs in groups:
        assert all(a["prompt"]["text"] == b["prompt"]["text"] for a, b in pairs)
    assert summarize(
        [
            {"arm": "a", "scene_id": "s1", "mean_lpips": 1, "pairs": 3},
            {"arm": "a", "scene_id": "s1", "mean_lpips": 3, "pairs": 1},
            {"arm": "a", "scene_id": "s2", "mean_lpips": 6, "pairs": 1},
        ]
    ) == {"a": {"mean_by_scene": 4, "scenes": 2, "exact_captions": 3, "pairs": 5}}
    with pytest.raises(ValueError, match="distinct seeds"):
        list(grouped_pairs([records[0], {**records[0], "sample_id": "duplicate"}]))
