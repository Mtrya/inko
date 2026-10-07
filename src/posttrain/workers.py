"""Bounded single-host resident role workers with explicit startup and failure."""

import importlib.metadata
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def _worker(role, index, config, assets, identities, incoming, outgoing):
    # Dependency overlays are role-local; scorer versions need not change pretraining.
    overlays = assets.get(role, {}).get("python_paths", [])
    sys.path[:0] = overlays
    bins = [str(Path(p) / "bin") for p in overlays if (Path(p) / "bin").is_dir()]
    os.environ["PATH"] = os.pathsep.join([*bins, os.environ.get("PATH", "")])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # These are standalone inference processes. An enclosing launcher must not
    # make a scorer's HF TrainingArguments initialize a training collective.
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        os.environ.pop(key, None)
    try:
        from .models import Generator, peak_memory, set_device, synchronize
        from .records import ScoreRequest
        from .scoring import HPSv3, VQAScore, acquire

        start = time.perf_counter()
        device = set_device(
            assets["device_type"], index, config["execution"]["cpu_threads"]
        )
        os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)
        if role == "learner":
            backend = Generator(config, assets, device)
        else:
            backend = {"hpsv3": HPSv3, "vqascore": VQAScore}[role](
                config["scorers"][role], assets[role], device
            )
        synchronize(device)
        outgoing.put(
            {
                "event": "ready",
                "role": role,
                "device": index,
                "load_seconds": time.perf_counter() - start,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "diffusers")
                },
            }
        )
        while True:
            task = incoming.get()
            if task is None:
                return
            start = time.perf_counter()
            if role == "learner":
                result = backend.generate(task)
            else:
                requests = [ScoreRequest(**r) for r in task["requests"]]
                result = {
                    "scores": [
                        asdict(r)
                        for r in acquire(
                            backend,
                            requests,
                            scorer=role,
                            scorer_identity=identities[role],
                            attempts=config["execution"]["score_attempts"],
                        )
                    ]
                }
            synchronize(device)
            outgoing.put(
                {
                    "event": "result",
                    "role": role,
                    "device": index,
                    "task_id": task["task_id"],
                    "result": result,
                    "service_seconds": time.perf_counter() - start,
                    "queue_seconds": start - task["enqueued_at"],
                    "peak_memory_bytes": peak_memory(device),
                }
            )
    except BaseException:  # noqa: BLE001 - propagate worker traceback to coordinator
        outgoing.put(
            {
                "event": "fatal",
                "role": role,
                "device": index,
                "error": traceback.format_exc(),
            }
        )


class ResidentWorkers:
    """Processes own models until close; the coordinator owns all result matching."""

    def __init__(self, config, assets, identities, roles=None):
        self.config = config
        context = mp.get_context("spawn")
        capacity = config["execution"]["max_inflight_batches"]
        self.incoming = {
            role: context.Queue(maxsize=capacity)
            for role in (roles or ("learner", "hpsv3", "vqascore"))
        }
        self.outgoing = context.Queue(maxsize=capacity * 3 + 32)
        self.processes = []
        for role, incoming in self.incoming.items():
            for index in config["execution"][role + "_devices"]:
                process = context.Process(
                    target=_worker,
                    args=(
                        role,
                        index,
                        config,
                        assets,
                        identities,
                        incoming,
                        self.outgoing,
                    ),
                )
                self.processes.append(process)
        self.startup = []

    def __enter__(self):
        try:
            for process in self.processes:
                process.start()
            deadline = time.monotonic() + self.config["execution"]["startup_timeout"]
            while len(self.startup) < len(self.processes):
                message = self.receive(deadline)
                if message["event"] != "ready":
                    raise RuntimeError("unexpected worker startup message")
                self.startup.append(message)
            return self
        except BaseException:
            self.close()
            raise

    def submit(self, role, task):
        self.incoming[role].put({**task, "enqueued_at": time.perf_counter()}, timeout=5)

    def receive(self, deadline):
        while time.monotonic() < deadline:
            try:
                message = self.outgoing.get(
                    timeout=min(1, max(0.01, deadline - time.monotonic()))
                )
            except queue.Empty:
                dead = [
                    p
                    for p in self.processes
                    if p.pid is not None and p.exitcode is not None
                ]
                if dead:
                    raise RuntimeError(
                        f"resident worker exited: {[p.exitcode for p in dead]}"
                    )
                continue
            if message["event"] == "fatal":
                raise RuntimeError(
                    f"{message['role']} device {message['device']}: {message['error']}"
                )
            return message
        raise TimeoutError("resident worker deadline exceeded")

    def close(self):
        for role, incoming in self.incoming.items():
            for _ in self.config["execution"][role + "_devices"]:
                try:
                    incoming.put_nowait(None)
                except queue.Full:
                    break
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
        for q in [*self.incoming.values(), self.outgoing]:
            q.cancel_join_thread()
            q.close()

    def __exit__(self, *exc):
        self.close()


def run_pipeline(tasks, workers, config, on_generated, on_scored):
    """Overlap generation/scoring, bounding total batches until both scores join.

    Tasks with existing records skip generation. Each may request only the scorer
    results missing from a durable previous attempt. The worker interface also
    permits deterministic mechanism tests without loading any model.
    """
    iterator = iter(tasks)
    inflight, exhausted = {}, False
    telemetry = []
    limit = config["execution"]["max_inflight_batches"]
    timeout = config["execution"]["task_timeout"]

    def score(task, records):
        requests = [
            {
                "sample_id": r["sample_id"],
                "image_path": r["image_path"],
                "image_sha256": r["image_sha256"],
                "prompt": r["prompt"]["text"],
            }
            for r in records
        ]
        task["phase"] = "score"
        task["expected_samples"] = {r["sample_id"] for r in records}
        task["waiting"] = set(task.get("missing_scorers", ["hpsv3", "vqascore"]))
        task["deadline"] = time.monotonic() + timeout
        for role in task["waiting"]:
            workers.submit(role, {"task_id": task["task_id"], "requests": requests})

    while inflight or not exhausted:
        while not exhausted and len(inflight) < limit:
            try:
                task = dict(next(iterator))
            except StopIteration:
                exhausted = True
                break
            key = task["task_id"]
            if key in inflight:
                raise ValueError("duplicate in-flight task ID")
            inflight[key] = task
            if "records" in task:
                score(task, task["records"])
                if not task["waiting"]:
                    del inflight[key]
            else:
                task.update(phase="generate", deadline=time.monotonic() + timeout)
                workers.submit("learner", task)
        if not inflight:
            continue
        message = workers.receive(min(t["deadline"] for t in inflight.values()))
        if message.get("event") != "result" or message.get("task_id") not in inflight:
            raise ValueError("unexpected worker result")
        task = inflight[message["task_id"]]
        role, result = message["role"], message["result"]
        if role == "learner":
            if task["phase"] != "generate":
                raise ValueError("duplicate generation result")
            if [r["sample_id"] for r in result["records"]] != task["sample_ids"]:
                raise ValueError("generation returned different sample identities")
            on_generated(result["records"])
            score(task, result["records"])
            if not task["waiting"]:
                del inflight[task["task_id"]]
        else:
            if task["phase"] != "score" or role not in task["waiting"]:
                raise ValueError("unexpected/duplicate scorer response")
            returned = [r["sample_id"] for r in result["scores"]]
            if (
                len(returned) != len(set(returned))
                or set(returned) != task["expected_samples"]
            ):
                raise ValueError("scorer returned different sample identities")
            on_scored(role, result["scores"])
            task["waiting"].remove(role)
            if not task["waiting"]:
                del inflight[task["task_id"]]
        telemetry.append({k: v for k, v in message.items() if k != "result"})
        if role == "learner":
            telemetry[-1]["timings"] = result["timings"]
    return telemetry
