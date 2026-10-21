"""Own model processes and their isolated Python environments on the scoring node."""

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .backends.rubric import REFERENCE, SAMPLING, TEMPLATE, RubricJudge
from .backends.vqascore import SCORING_OPTIONS
from .client import http_json


def isolated_environment(paths=None):
    env = dict(os.environ)
    if paths is None:
        paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    repo = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join([*paths, repo])
    env["PATH"] = os.pathsep.join(
        [
            *(str(Path(p) / "bin") for p in paths if (Path(p) / "bin").is_dir()),
            env.get("PATH", ""),
        ]
    )
    env.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GLOO_SOCKET_IFNAME",
    ):
        env.pop(key, None)
    return env


class Replicas:
    def __init__(self, workers, concurrency):
        self.available = queue.Queue()
        for _ in range(concurrency):
            for worker in workers:
                self.available.put(worker)

    def call(self, method, *args):
        worker = self.available.get()
        try:
            return getattr(worker, method)(*args)
        finally:
            self.available.put(worker)

    def score(self, requests):
        return self.call("score", requests)

    def judge(self, prompt, rubric, images):
        return self.call("judge", prompt, rubric, images)


class Auxiliary:
    def __init__(self, endpoint, timeout):
        self.endpoint, self.timeout = endpoint, timeout

    def score(self, requests):
        return http_json(
            self.endpoint,
            "/score",
            self.timeout,
            {"inputs": [vars(r) for r in requests]},
        )["values"]


class Runtime:
    def __init__(self, config, assets, output):
        self.config, self.assets, self.output = config, assets, Path(output)
        self.processes, self.logs, self.ready = [], [], []
        self.backends, self.concurrency = {}, {}
        self.info = {
            "semantics": {
                "rubric": {
                    "weights": asdict(config.rubric.weights),
                    "prompt": TEMPLATE,
                    "reference": REFERENCE,
                    "max_tokens": config.rubric.max_tokens,
                    "max_model_len": config.rubric.max_model_len,
                    "thinking": True,
                    **SAMPLING,
                },
                "hpsv3": {
                    "source": asdict(config.hpsv3.source),
                    "weights": asdict(config.hpsv3.weights),
                    "backbone": asdict(config.hpsv3.backbone),
                },
                "vqascore": {
                    "source": asdict(config.vqascore.source),
                    "weights": asdict(config.vqascore.weights),
                    **SCORING_OPTIONS,
                },
            },
            "workers": [],
        }

    def spawn(self, name, arguments, environment):
        self.output.mkdir(parents=True, exist_ok=True)
        log = (self.output / f"{name}.log").open("a")
        self.logs.append(log)
        process = subprocess.Popen(
            arguments,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes.append(process)

    @staticmethod
    def port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def __enter__(self):
        try:
            config, assets = self.config, self.assets
            capture_sizes = [
                1 << i for i in range(config.rubric.concurrency.bit_length())
            ]
            if capture_sizes[-1] != config.rubric.concurrency:
                capture_sizes.append(config.rubric.concurrency)
            judges = []
            for index, devices in enumerate(config.rubric.replicas):
                port = self.port()
                endpoint = f"http://127.0.0.1:{port}"
                deploy = assets["rubric"]
                env = isolated_environment(deploy.get("python_paths"))
                mask = (
                    "ASCEND_RT_VISIBLE_DEVICES"
                    if assets["device_type"] == "npu"
                    else "CUDA_VISIBLE_DEVICES"
                )
                visible = os.environ.get(mask, "").split(",")
                env[mask] = ",".join(
                    visible[d] if visible != [""] else str(d) for d in devices
                )
                self.spawn(
                    f"rubric-{index}",
                    [
                        deploy.get("python", sys.executable),
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                        "--model",
                        deploy["weights"],
                        "--served-model-name",
                        config.rubric.model_name,
                        "--tensor-parallel-size",
                        str(len(devices)),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--max-model-len",
                        str(config.rubric.max_model_len),
                        "--trust-remote-code",
                        "--reasoning-parser",
                        "qwen3",
                        "--compilation-config",
                        json.dumps({"cudagraph_capture_sizes": capture_sizes}),
                    ],
                    env,
                )
                judges.append(
                    RubricJudge(
                        endpoint,
                        config.rubric.model_name,
                        timeout=config.rubric.request_timeout,
                        max_tokens=config.rubric.max_tokens,
                    )
                )
                self.ready.append((endpoint, "/v1/models", f"rubric-{index}"))
            self.backends["rubric"] = Replicas(judges, config.rubric.concurrency)
            self.concurrency["rubric"] = len(judges) * config.rubric.concurrency
            for model in ("hpsv3", "vqascore"):
                spec, deploy = getattr(config, model), assets[model]
                workers = []
                for index, device in enumerate(spec.devices):
                    port = self.port()
                    endpoint = f"http://127.0.0.1:{port}"
                    worker_assets = self.output / f"{model}-{index}.json"
                    worker_assets.write_text(
                        json.dumps({**deploy, "device_type": assets["device_type"]})
                    )
                    self.spawn(
                        f"{model}-{index}",
                        [
                            deploy.get("python", sys.executable),
                            *(["-S"] if model == "vqascore" else []),
                            "-m",
                            "src.posttrain.scoring.backends.runtime",
                            "--backend",
                            model,
                            "--assets",
                            str(worker_assets.resolve()),
                            "--device",
                            str(device),
                            "--cpu-threads",
                            str(spec.cpu_threads),
                            "--port",
                            str(port),
                            "--timeout",
                            str(spec.request_timeout),
                        ],
                        isolated_environment(deploy.get("python_paths", ())),
                    )
                    workers.append(Auxiliary(endpoint, spec.request_timeout))
                    self.ready.append((endpoint, "/info", f"{model}-{index}"))
                self.backends[model] = Replicas(workers, 1)
                self.concurrency[model] = len(workers)
            deadline = time.monotonic() + config.server.startup_timeout
            pending = list(self.ready)
            while pending:
                self.check()
                for endpoint, route, name in list(pending):
                    try:
                        info = http_json(
                            endpoint,
                            route,
                            min(2, max(0.1, deadline - time.monotonic())),
                        )
                    except (OSError, TimeoutError):
                        continue
                    self.info["workers"].append({"name": name, **info})
                    pending.remove((endpoint, route, name))
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"scoring startup timed out: {[p[2] for p in pending]}"
                    )
                if pending:
                    time.sleep(0.5)
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        if any(p.poll() is not None for p in self.processes):
            raise RuntimeError(f"scorer worker exited; inspect {self.output}")

    def close(self):
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in self.logs:
            log.close()

    def __exit__(self, *exc):
        self.close()
