"""Resident generation and bounded HTTP acquisition, without backend dispatch."""

import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .artifacts import package_versions
from .scoring.client import request_for_record
from .scoring.lifecycle import Services, isolated_environment


def _worker(index, config, assets, incoming, outgoing):
    paths = assets["learner"]["python_paths"]
    sys.path[:0] = paths
    os.environ.update(isolated_environment(paths))
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

        started = time.perf_counter()
        device = set_device(assets["device_type"], index, config.execution.cpu_threads)
        backend = Generator(config, assets, device)
        synchronize(device)
        outgoing.put(
            {
                "event": "ready",
                "role": "learner",
                "device": index,
                "load_seconds": time.perf_counter() - started,
                "packages": package_versions(),
            }
        )
        while True:
            task = incoming.get()
            if task is None:
                return
            started = time.perf_counter()
            result = backend.generate(task)
            outgoing.put(
                {
                    "event": "result",
                    "role": "learner",
                    "device": index,
                    "task_id": task["task_id"],
                    "result": result,
                    "service_seconds": time.perf_counter() - started,
                    "queue_seconds": started - task["enqueued_at"],
                    "peak_memory_bytes": peak_memory(device),
                }
            )
    except BaseException:  # noqa: BLE001 - propagate worker failures
        outgoing.put(
            {
                "event": "fatal",
                "role": "learner",
                "device": index,
                "error": traceback.format_exc(),
            }
        )


class ResidentWorkers:
    def __init__(self, config, assets, output, *, generate=True, score=True):
        self.config, self.assets, self.output = config, assets, output
        context = mp.get_context("spawn")
        capacity = config.execution.max_inflight_batches
        self.incoming = context.Queue(maxsize=capacity)
        self.outgoing = context.Queue(maxsize=capacity * (len(config.scorers) + 1) + 32)
        self.processes = (
            [
                context.Process(
                    target=_worker,
                    args=(i, config, assets, self.incoming, self.outgoing),
                )
                for i in config.execution.learner_devices
            ]
            if generate
            else []
        )
        self.services = Services(
            config.scorers if score else (),
            assets.get("services", {}),
            output,
            config.execution.learner_devices,
            config.execution.startup_timeout,
        )
        self.executor = None
        self.startup = []

    def __enter__(self):
        try:
            self.services.__enter__()
            self.startup.extend(self.services.startup)
            self.executor = ThreadPoolExecutor(
                max_workers=max(1, sum(s.concurrency for s in self.config.scorers))
            )
            for p in self.processes:
                p.start()
            deadline = time.monotonic() + self.config.execution.startup_timeout
            for _ in self.processes:
                message = self.receive(deadline)
                if message["event"] != "ready":
                    raise RuntimeError("unexpected generation startup message")
                self.startup.append(message)
            return self
        except BaseException:
            self.close()
            raise

    def submit(self, role, task):
        if role == "learner":
            self.incoming.put({**task, "enqueued_at": time.perf_counter()}, timeout=5)
            return
        self.executor.submit(self._score, role, task, time.perf_counter())

    def _score(self, role, task, enqueued):
        try:
            client = self.services.clients[role]
            started = time.perf_counter()
            requests = [request_for_record(r, client.spec) for r in task["records"]]
            scores = client.acquire(requests)
            self.outgoing.put(
                {
                    "event": "result",
                    "role": role,
                    "task_id": task["task_id"],
                    "result": {"scores": scores},
                    "service_seconds": time.perf_counter() - started,
                    "queue_seconds": started - enqueued,
                }
            )
        except BaseException:  # noqa: BLE001 - propagate acquisition failures
            self.outgoing.put(
                {"event": "fatal", "role": role, "error": traceback.format_exc()}
            )

    def receive(self, deadline):
        while time.monotonic() < deadline:
            self.services.check()
            try:
                message = self.outgoing.get(
                    timeout=min(1, max(0.01, deadline - time.monotonic()))
                )
            except queue.Empty:
                if any(
                    p.pid is not None and p.exitcode is not None for p in self.processes
                ):
                    raise RuntimeError("resident generation worker exited")
                continue
            if message["event"] == "fatal":
                raise RuntimeError(f"{message['role']}: {message['error']}")
            return message
        raise TimeoutError("pipeline task deadline exceeded")

    def close(self, *, graceful=False):
        self.services.close()
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        alive = [p for p in self.processes if p.pid is not None and p.is_alive()]
        if graceful:
            # Completed workers return normally so accelerator compiler children
            # and multiprocessing resources run their own shutdown hooks.
            for _ in alive:
                self.incoming.put(None, timeout=1)
            deadline = time.monotonic() + 30
            for p in alive:
                p.join(timeout=max(0, deadline - time.monotonic()))
        for p in self.processes:
            if p.pid is not None and p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
        for q in (self.incoming, self.outgoing):
            q.cancel_join_thread()
            q.close()

    def __exit__(self, *exc):
        self.close(graceful=exc[0] is None)


def run_pipeline(tasks, workers, config, on_generated, on_scored):
    """Overlap generation/scoring, bounding total batches until both scores join.

    Tasks with existing records skip generation. Each may request only the scorer
    results missing from a durable previous attempt. The worker interface also
    permits deterministic mechanism tests without loading any model.
    """
    iterator = iter(tasks)
    inflight, exhausted = {}, False
    telemetry = []
    limit = config.execution.max_inflight_batches
    timeout = config.execution.task_timeout

    def score(task, records):
        task["phase"] = "score"
        task["expected_samples"] = {r["sample_id"] for r in records}
        task["waiting"] = set(
            task.get("missing_scorers", [s.id for s in config.scorers])
        )
        task["deadline"] = time.monotonic() + timeout
        for role in task["waiting"]:
            workers.submit(role, {"task_id": task["task_id"], "records": records})

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
