"""Online run tracking from complete learner records and benchmark reports."""

import json
import statistics
from pathlib import Path

from src.pretrain.tracking import read_tracking_record, validate_run_id

from .records import atomic_json
from .schema import require

PROJECT = "inko-posttrain"


class Tracker:
    def __init__(self, output, run_name, *, resume=None, config=None):
        self.output = Path(output)
        self.run_name, self.resume, self.config = run_name, resume, config

    def __enter__(self):
        import swanlab

        identity = None
        if self.resume is not None:
            identity = read_tracking_record(self.resume)
        if (self.output / "tracking.json").exists():
            local = read_tracking_record(self.output)
            require(
                identity is None or identity == local, "SwanLab run identity differs"
            )
            identity = local
        require(
            identity is None or identity["swanlab_project"] == PROJECT,
            "checkpoint belongs to another SwanLab project",
        )
        kwargs = {
            "project": PROJECT,
            "name": self.run_name,
            "mode": "online",
            "resume": "never" if identity is None else "must",
            "log_dir": str(self.output / "swanlog"),
            "config": self.config,
            "settings": swanlab.Settings(interactive=False),
        }
        if identity is not None:
            kwargs["id"] = identity["swanlab_run_id"]
        self.sdk = swanlab
        try:
            run = swanlab.init(**kwargs)
            require(run.mode == "online", "SwanLab must remain online")
            run_id = validate_run_id(run.id)
            require(
                identity is None or run_id == identity["swanlab_run_id"],
                "SwanLab resumed a different experiment",
            )
            self.output.mkdir(parents=True, exist_ok=True)
            atomic_json(
                self.output / "tracking.json",
                {"swanlab_project": PROJECT, "swanlab_run_id": run_id},
            )
        except BaseException:
            if swanlab.get_run() is not None:
                swanlab.finish(state="crashed")
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.sdk.finish(state="success" if exc_type is None else "crashed")

    def log(self, metrics, step):
        self.sdk.log(metrics, step=step)

    def evaluation(self, report, panels, step):
        metrics = {}
        prefix = f"evaluation/{report['suite']}"
        for arm, channels in report.get("arms", {}).items():
            for channel, values in channels.items():
                for field in ("mean_on_successes", "population_stddev", "scored"):
                    if values[field] is not None:
                        metrics[f"{prefix}/{arm}/{channel}/{field}"] = values[field]
        for arm, values in report.get("diversity", {}).get("arms", {}).items():
            metrics[f"{prefix}/{arm}/lpips"] = values["mean_by_scene"]
        metrics[f"{prefix}/panels"] = [self.sdk.Image(str(p)) for p in panels]
        self.log(metrics, step)


def optimizer_rates(optimizer):
    if optimizer is None:
        return {}
    return {
        f"{type(opt).__name__.lower()}/{index}": group["lr"]
        for opt in optimizer.optimizers
        for index, group in enumerate(opt.param_groups)
    }


def update_metrics(rows):
    """Loss records use world/global_samples scaling; their rank mean is global."""
    first = rows[0]
    metrics = {
        "clock/student_updates": first["step"],
        "runtime/seconds": max(r["seconds"] for r in rows),
        "runtime/peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
    }
    for field in ("student_loss", "student_grad_norm"):
        if field in first:
            metrics[f"train/{field}"] = statistics.mean(r[field] for r in rows)
    for term in first.get("term_losses", {}):
        metrics[f"train/{term}_loss"] = statistics.mean(
            r["term_losses"][term] for r in rows
        )
    for role, values in first["learning_rates"].items():
        metrics.update({f"lr/{role}/{key}": value for key, value in values.items()})
    reward = first.get("reward")
    if reward is not None:
        for channel, values in reward["statistics"].items():
            for key in ("mean", "std"):
                metrics[f"reward/{channel}/{key}"] = values[key]
        for key in ("advantages", "probabilities"):
            if key in reward:
                metrics[f"reward/{key}_mean"] = statistics.mean(reward[key])
                metrics[f"reward/{key}_std"] = statistics.pstdev(reward[key])
    return metrics


class TrainingMetrics:
    def __init__(self, tracker, config, output, attempt, saved=None):
        from .clock import rounds_before

        self.tracker, self.config, self.output, self.attempt = (
            tracker,
            config,
            Path(output),
            attempt,
        )
        self.ranks = len(config.execution.learner_devices)
        self.step = 0 if saved is None else saved["step"]
        self.round = rounds_before(config, self.step)

    def rows(self, directory, suffix):
        paths = [
            self.output / directory / f"{self.attempt}-{suffix}-rank-{rank:03d}.json"
            for rank in range(self.ranks)
        ]
        if not all(p.exists() for p in paths):
            return None
        return [json.loads(p.read_text()) for p in paths]

    def poll(self):
        while self.rows("rounds", f"round-{self.round:06d}") is not None:
            for inner in range(self.config.nft.updates_per_round):
                rows = self.rows("updates", f"round-{self.round:06d}-inner-{inner:03d}")
                require(rows is not None, "committed round lacks update telemetry")
                self.step += 1
                self.tracker.log(update_metrics(rows), self.step)
            self.round += 1
