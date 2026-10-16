"""Scorer-owned source, model and output contracts."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..schema import decode, require
from .backends.hpsv3 import OUTPUTS as HPS_OUTPUTS
from .backends.vqascore import OUTPUTS as VQA_OUTPUTS
from .contracts import Capabilities


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
class HPSOptions:
    backbone: Model


@dataclass(frozen=True)
class Semantics:
    backend: str
    source: Source
    weights: Model

    def __post_init__(self):
        require(self.backend in {"hpsv3", "vqascore"}, "unknown backend")


@dataclass(frozen=True)
class Limits:
    max_batch_size: int
    queue_capacity: int
    request_timeout: float
    max_body_bytes: int
    cpu_threads: int

    def __post_init__(self):
        require(
            min(
                self.max_batch_size,
                self.queue_capacity,
                self.request_timeout,
                self.max_body_bytes,
                self.cpu_threads,
            )
            > 0,
            "service limits must be positive",
        )


@dataclass(frozen=True)
class ServiceConfig:
    semantics: Semantics
    options: HPSOptions | None
    limits: Limits

    @property
    def capabilities(self):
        return Capabilities(
            HPS_OUTPUTS if self.semantics.backend == "hpsv3" else VQA_OUTPUTS,
            self.limits.max_batch_size,
            ("shared_file", "png_base64"),
        )


def load_service_config(path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    semantics = decode(Semantics, raw["semantics"])
    sections = {"semantics", "limits"}
    if semantics.backend == "hpsv3":
        sections.add("options")
    require(set(raw) == sections, "invalid service sections")
    options = decode(HPSOptions, raw["options"]) if "options" in sections else None
    return ServiceConfig(semantics, options, decode(Limits, raw["limits"]))
