"""Scorer-owned source, model, preprocessing and output contracts."""

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from ..records import identity
from ..schema import decode, digest, require
from .contracts import PROTOCOL, Capabilities, Channel


@dataclass(frozen=True)
class Source:
    repository: str
    revision: str

    def __post_init__(self):
        digest(self.revision, length=40)


@dataclass(frozen=True)
class Model:
    repository: str
    revision: str
    files: tuple[str, ...]

    def __post_init__(self):
        digest(self.revision, length=40)
        require(bool(self.files), "model file patterns required")


@dataclass(frozen=True)
class HPSOptions:
    backbone: Model
    head_precision: str

    def __post_init__(self):
        require(self.head_precision == "float32", "HPS head requires float32")


@dataclass(frozen=True)
class VQAOptions:
    release: str
    model: str
    question_template: str
    answer_template: str
    max_new_tokens: int
    temperature: float

    def __post_init__(self):
        require(
            self.release == "3.1" and self.model == "qwen3-vl-8b",
            "unsupported VQA adapter",
        )
        require(
            self.question_template.count("{}") == 1
            and self.answer_template == "Yes"
            and self.max_new_tokens == 1
            and self.temperature > 0,
            "invalid VQA scoring contract",
        )


@dataclass(frozen=True)
class Semantics:
    backend: str
    source: Source
    weights: Model
    precision: str
    attention: str
    preprocessing: str
    implementation: str
    input_schema: str
    structured_fields: tuple[str, ...]
    outputs: tuple[Channel, ...]
    packages: dict[str, str]
    runtime_files: dict[str, str]

    def __post_init__(self):
        require(self.backend in {"hpsv3", "vqascore"}, "unknown backend")
        for value in self.runtime_files.values():
            digest(value)
        require(
            {
                "torch",
                "transformers",
                "tokenizers",
                "qwen-vl-utils",
                "pillow",
                "torchvision",
                "numpy",
                "accelerate",
                "safetensors",
            }
            <= self.packages.keys(),
            "explicit numeric/preprocessing runtime versions required",
        )
        require(
            self.precision == "bfloat16" and self.attention == "sdpa",
            "unsupported numeric contract",
        )
        require(
            self.preprocessing == "upstream"
            and self.implementation == "inko-resident-v1",
            "unsupported adapter semantics",
        )
        require(
            self.input_schema == "image-text-v1" and not self.structured_fields,
            "initial backends accept image/text only",
        )


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
    options: HPSOptions | VQAOptions
    limits: Limits

    @property
    def fingerprint(self):
        return identity(
            {
                "protocol": PROTOCOL,
                "semantics": asdict(self.semantics),
                "options": asdict(self.options),
            }
        )

    @property
    def capabilities(self):
        s = self.semantics
        return Capabilities(
            PROTOCOL,
            self.fingerprint,
            s.input_schema,
            s.structured_fields,
            s.outputs,
            self.limits.max_batch_size,
            ("shared_file", "png_base64"),
            True,
            True,
        )


def load_service_config(path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    require(set(raw) == {"semantics", "options", "limits"}, "invalid service sections")
    semantics = decode(Semantics, raw["semantics"])
    options = decode(
        HPSOptions if semantics.backend == "hpsv3" else VQAOptions, raw["options"]
    )
    result = ServiceConfig(semantics, options, decode(Limits, raw["limits"]))
    expected = {"mu", "log_sigma"} if semantics.backend == "hpsv3" else {"probability"}
    require(
        {c.name for c in semantics.outputs} == expected,
        "backend output contract differs",
    )
    return result
