"""Optional owned service processes; external endpoints are never stopped."""

import os
import signal
import subprocess
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from ..schema import decode, require
from .client import ScoreClient


@dataclass(frozen=True)
class Launch:
    config: str
    assets: str
    device: int
    bind: str
    python_paths: tuple[str, ...]


@dataclass(frozen=True)
class Deployment:
    endpoints: tuple[str, ...]
    launches: tuple[Launch, ...]


def isolated_environment(paths):
    env = dict(os.environ)
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
    ):
        env.pop(key, None)
    return env


class Services:
    def __init__(self, specs, deployments, output, learner_devices, startup_timeout):
        self.specs = specs
        self.deployments = {s.id: decode(Deployment, deployments[s.id]) for s in specs}
        self.output, self.timeout = Path(output), startup_timeout
        devices = list(learner_devices)
        for deployment in self.deployments.values():
            for launch in deployment.launches:
                require(
                    launch.device >= 0
                    and f"http://{launch.bind}" in deployment.endpoints,
                    "launch device/endpoint mismatch",
                )
                devices.append(launch.device)
        require(
            len(devices) == len(set(devices)),
            "learner/owned service devices must be disjoint",
        )
        self.processes, self.logs, self.clients, self.startup = [], [], {}, []

    def __enter__(self):
        try:
            for spec in self.specs:
                deployment = self.deployments[spec.id]
                for index, launch in enumerate(deployment.launches):
                    directory = self.output / "services" / spec.id
                    directory.mkdir(parents=True, exist_ok=True)
                    log = (directory / f"replica-{index}.log").open("a")
                    self.logs.append(log)
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "src.posttrain.scoring.service",
                            "--config",
                            launch.config,
                            "--assets",
                            launch.assets,
                            "--bind",
                            launch.bind,
                            "--device",
                            str(launch.device),
                            "--cache",
                            str(directory / "cache"),
                        ],
                        env=isolated_environment(launch.python_paths),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.processes.append(process)
                client = ScoreClient(
                    spec, deployment.endpoints, self.output / "acquisition" / spec.id
                )
                deadline = time.monotonic() + self.timeout
                while True:
                    self.check()
                    try:
                        cap = client.handshake(deadline=deadline)
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"{spec.id}: service startup deadline exceeded"
                            ) from None
                        time.sleep(0.5)
                self.clients[spec.id] = client
                self.startup.append(
                    {
                        "service": spec.id,
                        "endpoints": deployment.endpoints,
                        "capabilities": cap,
                    }
                )
            return self
        except BaseException:
            self.close()
            raise

    def check(self):
        if any(p.poll() is not None for p in self.processes):
            raise RuntimeError("owned scorer exited; inspect service logs")

    def close(self):
        for p in self.processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in self.processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in self.logs:
            log.close()

    def __exit__(self, *exc):
        self.close()
