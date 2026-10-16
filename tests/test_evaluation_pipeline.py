import pytest

from src.evaluation.benchmarks.report import build_report, paired_interval
from src.evaluation.benchmarks.workers import run_pipeline


def _shutdown_worker(incoming, ready, path):
    from pathlib import Path

    ready.set()
    assert incoming.get() is None
    Path(path).write_text("normal shutdown")


def test_completed_workers_exit_normally(tmp_path):
    import multiprocessing as mp

    from src.evaluation.benchmarks.workers import ResidentWorkers

    ctx = mp.get_context("spawn")
    incoming, outgoing, ready = ctx.Queue(2), ctx.Queue(2), ctx.Event()
    evidence = tmp_path / "shutdown.txt"
    process = ctx.Process(
        target=_shutdown_worker, args=(incoming, ready, str(evidence))
    )
    workers = ResidentWorkers.__new__(ResidentWorkers)
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

    def submit(self, task):
        key = task["task_id"]
        self.unfinished.add(key)
        self.max_unfinished = max(self.max_unfinished, len(self.unfinished))
        record = {"sample_id": "wrong" if self.corrupt else task["sample_ids"][0]}
        self.messages.append(
            {
                "event": "result",
                "task_id": key,
                "role": "learner",
                "result": {"records": [record], "timings": {}},
            }
        )

    def receive(self, deadline):
        # Reverse order intentionally exercises completion-order independence.
        message = self.messages.pop()
        self.unfinished.remove(message["task_id"])
        return message


def test_generation_bound_and_out_of_order_completion(sampling_config):
    workers, generated = Workers(), []
    tasks = [{"task_id": str(i), "sample_ids": [str(i)]} for i in range(7)]
    run_pipeline(tasks, workers, sampling_config, generated.extend)
    assert workers.max_unfinished == 2
    assert {r["sample_id"] for r in generated} == {str(i) for i in range(7)}
    assert not workers.unfinished


def test_wrong_sample_response_aborts_pipeline(sampling_config):
    config = sampling_config
    with pytest.raises(ValueError, match="different sample"):
        run_pipeline(
            [{"task_id": "a", "sample_ids": ["a"]}],
            Workers(corrupt=True),
            config,
            lambda rows: None,
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
            "values": {"hpsv3.mu": 1},
        }
    ]
    report = build_report(records, scores, config)
    comparison = report["paired"]["candidate"]["hpsv3.mu"]
    assert comparison["paired"] == 0 and comparison["expected"] == 1
    assert not comparison["coverage_complete"] and comparison["mean_delta"] is None


def test_rubric_report_distinguishes_agreed_ties_from_order_disagreement(
    sampling_config,
):
    from dataclasses import replace

    config = replace(
        sampling_config,
        evaluation=replace(
            sampling_config.evaluation, channels=("rubric.content_alignment",)
        ),
    )
    records, scores = [], []
    for prompt, credit, winners in (
        ("win", 1, ("candidate", "candidate")),
        ("tie", 0.5, (None, None)),
        ("disagreement", 0.5, ("reference", "candidate")),
        ("loss", 0, ("reference", "reference")),
    ):
        for arm in ("reference", "candidate"):
            sample = f"{prompt}:{arm}"
            records.append(
                {
                    "sample_id": sample,
                    "arm": arm,
                    "prompt": {"id": prompt, "scene_id": prompt},
                }
            )
            scores.append(
                {
                    "sample_id": sample,
                    "values": {
                        "rubric.content_alignment": (
                            credit if arm == "candidate" else 1 - credit
                        )
                    },
                    "judgment": {
                        "orders": [
                            {"content_alignment": {"winner": winner}}
                            for winner in winners
                        ]
                    },
                }
            )
    result = build_report(records, scores, config)["paired"]["candidate"][
        "rubric.content_alignment"
    ]
    assert (result["wins"], result["ties"], result["losses"]) == (1, 2, 1)
    assert result["win_credit"] == 0.5
    assert result["order_disagreements"] == 1
