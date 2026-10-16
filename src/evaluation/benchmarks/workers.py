"""Resident generation and bounded HTTP acquisition, without backend dispatch."""

import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

from src.posttrain.artifacts import package_versions
from src.posttrain.scoring.lifecycle import isolated_environment
from src.utils.device import device_count


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
        from src.utils.device import peak_memory, set_device, synchronize

        from .generation import Generator

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
    def __init__(self, config, assets):
        self.config, self.assets = config, assets
        context = mp.get_context("spawn")
        capacity = config.execution.max_inflight_batches
        self.incoming = context.Queue(maxsize=capacity)
        self.outgoing = context.Queue(maxsize=capacity + 32)
        self.processes = [
            context.Process(
                target=_worker, args=(i, config, assets, self.incoming, self.outgoing)
            )
            for i in range(device_count(assets["device_type"]))
        ]
        self.startup = []

    def __enter__(self):
        try:
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

    def submit(self, task):
        self.incoming.put({**task, "enqueued_at": time.perf_counter()}, timeout=5)

    def receive(self, deadline):
        while time.monotonic() < deadline:
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


def run_pipeline(tasks, workers, config, on_generated):
    """Bound generation batches; the callback streams each result to scoring."""
    iterator = iter(tasks)
    inflight, exhausted, telemetry = {}, False, []
    while inflight or not exhausted:
        while not exhausted and len(inflight) < config.execution.max_inflight_batches:
            try:
                task = next(iterator)
            except StopIteration:
                exhausted = True
                break
            key = task["task_id"]
            if key in inflight:
                raise ValueError("duplicate generation task ID")
            inflight[key] = (task, time.monotonic() + config.execution.task_timeout)
            workers.submit(task)
        if not inflight:
            continue
        message = workers.receive(min(deadline for _, deadline in inflight.values()))
        if message.get("event") != "result" or message.get("task_id") not in inflight:
            raise ValueError("unexpected generation result")
        task, _ = inflight.pop(message["task_id"])
        result = message["result"]
        if [r["sample_id"] for r in result["records"]] != task["sample_ids"]:
            raise ValueError("generation returned different sample identities")
        on_generated(result["records"])
        telemetry.append(
            {
                **{k: v for k, v in message.items() if k != "result"},
                "timings": result["timings"],
            }
        )
    return telemetry
