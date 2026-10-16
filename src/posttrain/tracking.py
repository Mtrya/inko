"""Online run tracking from complete learner records and benchmark reports."""

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from src.pretrain.tracking import read_tracking_record, validate_run_id

from .records import atomic_json
from .schema import require

PROJECT = "inko"


class Tracker:
    def __init__(self, output, run_name, *, resume=None, config=None, phase="training"):
        self.output = Path(output)
        self.run_name, self.resume, self.config = run_name, resume, config
        self.phase = phase

    def __enter__(self):
        import swanlab

        self.last_time = time.monotonic()
        self.scoring_counts = {}
        state_path = self.output / "telemetry.json"
        if not state_path.exists() and self.resume is not None:
            state_path = Path(self.resume) / "telemetry.json"
        self.state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"execution_seconds": 0.0, "evaluation_seconds": 0.0, "baselines": []}
        )
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
        run = None
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
            self.save_state()
            for group in ("monitor", "evaluation", "train", "runtime"):
                swanlab.define_metric(f"{group}/*", section_name=group)
            swanlab.define_metric("runtime/*", x_axis="runtime/execution_seconds")
            for key in (
                "train/reward/*",
                "train/nft_constant_group_fraction",
                "train/art_active_group_fraction",
            ):
                swanlab.define_metric(key, x_axis="runtime/collection_round")
            for key in ("runtime/collection_round", "runtime/execution_seconds"):
                swanlab.define_metric(key, hidden=True, x_axis="_step")
        except BaseException:
            if run is not None:
                swanlab.finish(state="crashed")
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.runtime({})
        finally:
            self.sdk.finish(state="success" if exc_type is None else "crashed")

    def save_state(self):
        atomic_json(self.output / "telemetry.json", self.state)

    def runtime(self, metrics):
        now = time.monotonic()
        elapsed = now - self.last_time
        self.state["execution_seconds"] += elapsed
        if self.phase == "evaluation":
            self.state["evaluation_seconds"] += elapsed
        self.last_time = now
        self.save_state()
        self.sdk.log(
            {
                "runtime/execution_seconds": self.state["execution_seconds"],
                "runtime/evaluation_seconds": self.state["evaluation_seconds"],
                **metrics,
            }
        )

    def scoring(self, counts):
        metrics = {
            f"runtime/scoring/{scorer}/{field}": value
            for scorer, fields in counts.items()
            for field, value in fields.items()
        }
        changed = {
            key: value
            for key, value in metrics.items()
            if self.scoring_counts.get(key) != value
        }
        self.scoring_counts.update(metrics)
        if changed:
            self.runtime(changed)

    def log(self, metrics, step):
        self.sdk.log(metrics, step=step)

    def collection(self, rows, reward, number):
        metrics = {"runtime/collection_round": number}
        for channel, values in reward["raw"].items():
            metrics[f"train/reward/{channel}/mean"] = statistics.mean(values)
            metrics[f"train/reward/{channel}/std"] = statistics.pstdev(values)
        metrics["train/art_active_group_fraction"] = rows[0]["active_group_fraction"]
        self.log(metrics, number)
        stages = ["collection", "submission_wait", "scoring_wait", "guidance", "round"]
        self.runtime(
            {
                **{
                    f"runtime/{stage}_seconds": max(r["timings"][stage] for r in rows)
                    for stage in stages
                },
                "runtime/training_images_scored": rows[0]["clock"]["samples_scored"],
                "runtime/skipped_collections": rows[0]["clock"]["skipped_rounds"],
                "runtime/learner_process_peak_gib": max(
                    r["peak_allocated_bytes"] for r in rows
                )
                / 1024**3,
            }
        )

    def evaluation(self, report, run, config, step):
        from src.evaluation.benchmarks.monitor import scene_grids

        suite = report["suite"]
        metadata = json.loads(
            json.dumps(
                {
                    "sampling": asdict(config.sampling),
                    "suites": {
                        name: getattr(config.evaluation, field)
                        for name, field in (
                            ("development", "suite"),
                            ("monitor", "monitor_suite"),
                        )
                    },
                    "arms": [asdict(arm) for arm in config.arms],
                    "channels": config.evaluation.channels,
                    "confidence": config.evaluation.confidence,
                    "prompt_limit": config.evaluation.prompt_limit,
                    "draws_per_prompt": config.evaluation.draws_per_prompt,
                    "diversity": asdict(config.evaluation.diversity),
                }
            )
        )
        if self.sdk.config.get("evaluation") != metadata:
            self.sdk.config.update({"evaluation": metadata})
        metrics = {}
        prefix = f"evaluation/{suite}"
        reference = config.arms[0].name
        for arm, channels in report.get("arms", {}).items():
            for channel, values in channels.items():
                key = f"{prefix}/{arm}/{channel}/mean"
                if (
                    values["complete"]
                    and values["mean_on_successes"] is not None
                    and (arm != reference or key not in self.state["baselines"])
                ):
                    metrics[key] = values["mean_on_successes"]
                    if arm == reference:
                        self.state["baselines"].append(key)
        for arm, channels in report.get("paired", {}).items():
            for channel, values in channels.items():
                if values["coverage_complete"] and values["mean_delta"] is not None:
                    key = f"{prefix}/{arm}/{channel}"
                    sign = 1
                    metrics[f"{key}/delta"] = sign * values["mean_delta"]
                    if "win_credit" in values:
                        metrics[f"{key}/win_credit"] = values["win_credit"]
                        for field in ("wins", "ties", "losses", "order_disagreements"):
                            metrics[f"{key}/{field}"] = values[field]
                    if values["interval"] is not None:
                        lower, upper = sorted(sign * v for v in values["interval"])
                        metrics[f"{key}/delta_lower"] = lower
                        metrics[f"{key}/delta_upper"] = upper
        for arm, values in report.get("diversity", {}).get("arms", {}).items():
            key = f"monitor/lpips/{suite}/{arm}"
            if arm != reference or key not in self.state["baselines"]:
                metrics[key] = values["mean_by_scene"]
                if arm == reference:
                    self.state["baselines"].append(key)
        if suite == "monitor":
            for scene, grids in scene_grids(run, config.arms, step).items():
                metrics[f"monitor/images/{scene}"] = [
                    self.sdk.Image(str(path), caption=caption, file_type="jpeg")
                    for path, caption in grids
                ]
        if metrics:
            self.log(metrics, step)
        self.runtime(
            {
                f"runtime/evaluation/{suite}/failed_scores": sum(
                    values["expected"] - values["scored"]
                    for channels in report.get("arms", {}).values()
                    for values in channels.values()
                )
            }
        )


def optimizer_rates(optimizer):
    rates = {}
    for opt in optimizer.optimizers:
        values = {group["lr"] for group in opt.param_groups}
        require(len(values) == 1, "optimizer component has multiple learning rates")
        rates[type(opt).__name__.lower()] = values.pop()
    return rates


def update_metrics(rows):
    """Loss records use world/global_samples scaling; their rank mean is global."""
    first = rows[0]
    metrics = {"train/grad_norm": statistics.mean(r["student_grad_norm"] for r in rows)}
    for term in first["term_losses"]:
        if term == "reference" and not first["reference_enabled"]:
            continue
        metrics[f"train/{term}_loss"] = statistics.mean(
            r["term_losses"][term] for r in rows
        )
    items = sum(r["art_items"] for r in rows)
    metrics["train/art_rejected_fraction"] = sum(r["art_masked"] for r in rows) / items
    metrics["train/art_inactive_fraction"] = (
        sum(r["art_items"] - r["art_kept"] - r["art_masked"] for r in rows) / items
    )
    metrics.update(
        {f"train/lr/{key}": value for key, value in first["learning_rates"].items()}
    )
    return metrics


class TrainingMetrics:
    def __init__(self, tracker, config, output, attempt, world_size, saved=None):
        self.tracker, self.config, self.output, self.attempt = (
            tracker,
            config,
            Path(output),
            attempt,
        )
        self.ranks = world_size
        self.step = 0 if saved is None else saved["step"]
        self.round = 0 if saved is None else saved["committed_rounds"]

    def rows(self, directory, suffix):
        paths = [
            self.output / directory / f"{self.attempt}-{suffix}-rank-{rank:03d}.json"
            for rank in range(self.ranks)
        ]
        if not all(p.exists() for p in paths):
            return None
        return [json.loads(p.read_text()) for p in paths]

    def poll(self):
        while (rounds := self.rows("rounds", f"round-{self.round:06d}")) is not None:
            if not rounds[0]["skipped_zero_guidance"]:
                for inner in range(self.config.updates_per_round):
                    rows = self.rows(
                        "updates", f"round-{self.round:06d}-inner-{inner:03d}"
                    )
                    require(rows is not None, "committed round lacks update telemetry")
                    self.step += 1
                    self.tracker.log(update_metrics(rows), self.step)
                    self.tracker.runtime(
                        {
                            "runtime/update_seconds": max(
                                r["update_seconds"] for r in rows
                            )
                        }
                    )
            from .rewards import reward_path

            reward = json.loads(
                reward_path(self.output, self.attempt, self.round).read_text()
            )["reward"]
            self.tracker.collection(rounds, reward, self.round + 1)
            self.round += 1
