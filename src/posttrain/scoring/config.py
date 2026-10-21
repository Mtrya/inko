"""Client transport settings and the server-owned model deployment."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..schema import decode, require


@dataclass(frozen=True)
class ClientConfig:
    timeout: float
    attempts: int
    max_pending: int

    def __post_init__(self):
        require(
            min(self.timeout, self.attempts, self.max_pending) > 0,
            "scoring client limits must be positive",
        )


@dataclass(frozen=True)
class Source:
    repository: str


@dataclass(frozen=True)
class Model:
    repository: str
    files: tuple[str, ...]

    def __post_init__(self):
        require(bool(self.files), "model file patterns required")


@dataclass(frozen=True)
class ServerLimits:
    max_collections: int
    max_samples: int
    max_image_bytes: int
    max_body_bytes: int
    startup_timeout: float

    def __post_init__(self):
        require(
            min(
                self.max_collections,
                self.max_samples,
                self.max_image_bytes,
                self.max_body_bytes,
                self.startup_timeout,
            )
            > 0,
            "invalid scoring server limits",
        )


@dataclass(frozen=True)
class JudgeConfig:
    weights: Model
    model_name: str
    replicas: tuple[tuple[int, ...], ...]
    concurrency: int
    max_tokens: int
    max_model_len: int
    request_timeout: float

    def __post_init__(self):
        require(
            bool(self.replicas) and all(self.replicas), "judge replica devices required"
        )
        require(
            min(
                self.concurrency,
                self.max_tokens,
                self.max_model_len,
                self.request_timeout,
            )
            > 0,
            "invalid judge execution settings",
        )
        require(
            self.max_tokens < self.max_model_len,
            "judge context must leave room for inputs",
        )


@dataclass(frozen=True)
class AuxiliaryConfig:
    source: Source
    weights: Model
    devices: tuple[int, ...]
    batch_size: int
    request_timeout: float
    cpu_threads: int
    backbone: Model | None = None

    def __post_init__(self):
        require(
            bool(self.devices)
            and min(self.batch_size, self.request_timeout, self.cpu_threads) > 0,
            "invalid auxiliary worker settings",
        )


@dataclass(frozen=True)
class ServerConfig:
    server: ServerLimits
    rubric: JudgeConfig
    hpsv3: AuxiliaryConfig
    vqascore: AuxiliaryConfig

    def __post_init__(self):
        devices = [d for replica in self.rubric.replicas for d in replica]
        devices.extend((*self.hpsv3.devices, *self.vqascore.devices))
        require(
            min(devices) >= 0 and len(devices) == len(set(devices)),
            "scoring workers require disjoint devices",
        )
        require(
            self.hpsv3.backbone is not None and self.vqascore.backbone is None,
            "only HPS requires a separate backbone",
        )


def load_server_config(path):
    with Path(path).open("rb") as stream:
        return decode(ServerConfig, tomllib.load(stream))
