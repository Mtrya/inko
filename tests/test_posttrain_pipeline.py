from dataclasses import replace

import pytest

from src.evaluation.benchmarks.report import build_report, paired_interval
from src.posttrain.workers import run_pipeline


def _shutdown_worker(incoming, ready, path):
    from pathlib import Path

    ready.set()
    assert incoming.get() is None
    Path(path).write_text("normal shutdown")


def test_completed_workers_exit_normally(tmp_path):
    import multiprocessing as mp
    from types import SimpleNamespace

    from src.posttrain.workers import ResidentWorkers

    ctx = mp.get_context("spawn")
    incoming, outgoing, ready = ctx.Queue(2), ctx.Queue(2), ctx.Event()
    evidence = tmp_path / "shutdown.txt"
    process = ctx.Process(
        target=_shutdown_worker, args=(incoming, ready, str(evidence))
    )
    workers = ResidentWorkers.__new__(ResidentWorkers)
    workers.services = SimpleNamespace(close=lambda: None)
    workers.executor = None
    workers.processes = [process]
    workers.incoming, workers.outgoing = incoming, outgoing
    process.start()
    assert ready.wait(10)
    workers.close(graceful=True)
    assert process.exitcode == 0
    assert evidence.read_text() == "normal shutdown"


class Workers:
    def __init__(self, corrupt=False):
        self.messages = []
        self.unfinished = set()
        self.max_unfinished = 0
        self.scored = {}
        self.corrupt = corrupt

    def submit(self, role, task):
        key = task["task_id"]
        if role == "learner":
            self.unfinished.add(key)
            self.max_unfinished = max(self.max_unfinished, len(self.unfinished))
            record = {
                "sample_id": task["sample_ids"][0],
                "prompt": {"text": "text"},
                "image_path": "image",
                "image_sha256": "bytes",
            }
            result = {"records": [record], "timings": {}}
        else:
            result = {
                "scores": [
                    {
                        "sample_id": "wrong"
                        if self.corrupt
                        else task["records"][0]["sample_id"]
                    }
                ]
            }
        self.messages.append(
            {"event": "result", "task_id": key, "role": role, "result": result}
        )

    def receive(self, deadline):
        # Reverse order intentionally exercises completion-order independence.
        message = self.messages.pop()
        if message["role"] != "learner":
            key = message["task_id"]
            self.scored.setdefault(key, set()).add(message["role"])
            if len(self.scored[key]) == 2:
                self.unfinished.remove(key)
        return message


def test_bound_is_held_until_both_scores_return(sampling_config):
    workers = Workers()
    generated, scored = [], []
    tasks = [{"task_id": str(i), "sample_ids": [str(i)]} for i in range(7)]
    config = replace(
        sampling_config,
        scorers=(
            sampling_config.scorers[0],
            replace(sampling_config.scorers[0], id="another"),
        ),
    )
    run_pipeline(
        tasks, workers, config, generated.extend, lambda role, rows: scored.extend(rows)
    )
    assert workers.max_unfinished <= 2
    assert len(generated) == 7 and len(scored) == 14
    assert not workers.unfinished


def test_wrong_sample_response_aborts_pipeline(sampling_config):
    config = sampling_config
    with pytest.raises(ValueError, match="different sample"):
        run_pipeline(
            [{"task_id": "a", "sample_ids": ["a"]}],
            Workers(corrupt=True),
            config,
            lambda rows: None,
            lambda role, rows: None,
        )


def test_paired_interval_and_missing_score_coverage(sampling_config):
    assert paired_interval([("s1", 2), ("s2", 2)], 100, 0.95, 0) == [2, 2]
    assert paired_interval([("s1", 2), ("s1", 3)], 100, 0.95, 0) is None
    config = sampling_config
    records = [
        {"sample_id": arm, "arm": arm, "prompt": {"id": "p", "scene_id": "p"}}
        for arm in ("reference", "candidate")
    ]
    scores = [
        {
            "sample_id": "reference",
            "scorer": "unfamiliar",
            "status": "ok",
            "values": {"fit": 1},
        }
    ]
    report = build_report(records, scores, config)
    comparison = report["paired"]["candidate"]["unfamiliar.fit"]
    assert comparison["paired"] == 0 and comparison["expected"] == 1
    assert not comparison["coverage_complete"] and comparison["mean_delta"] is None
