"""Image/score envelopes and semantic versus transport identity."""

import base64
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..records import identity
from ..schema import digest, name, require


@dataclass(frozen=True)
class Channel:
    name: str
    direction: str

    def __post_init__(self):
        name(self.name)
        require(self.direction in {"higher", "lower"}, "invalid channel direction")


@dataclass(frozen=True)
class Capabilities:
    outputs: tuple[Channel, ...]
    max_batch_size: int
    transports: tuple[str, ...]

    def __post_init__(self):
        require(self.max_batch_size > 0, "invalid service batch size")
        require(
            bool(self.outputs)
            and len({c.name for c in self.outputs}) == len(self.outputs),
            "unique outputs required",
        )
        require(
            bool(self.transports)
            and set(self.transports) <= {"shared_file", "png_base64"},
            "unsupported transport",
        )


@dataclass(frozen=True)
class ImageInput:
    transport: str
    sha256: str
    payload: str

    def __post_init__(self):
        digest(self.sha256)
        require(
            self.transport in {"shared_file", "png_base64"},
            "unsupported image transport",
        )
        require(bool(self.payload), "image payload required")

    def read(self):
        data = (
            Path(self.payload).read_bytes()
            if self.transport == "shared_file"
            else base64.b64decode(self.payload, validate=True)
        )
        require(
            hashlib.sha256(data).hexdigest() == self.sha256, "score image bytes changed"
        )
        require(data.startswith(b"\x89PNG\r\n\x1a\n"), "score image must be PNG")
        return data


@dataclass(frozen=True)
class ScoreRequest:
    sample_id: str
    request_id: str
    cache_key: str
    prompt: str
    image: ImageInput
    seed: int | None

    def __post_init__(self):
        require(bool(self.sample_id) and bool(self.prompt), "missing score input")
        require(self.seed is None or 0 <= self.seed < 2**63, "invalid scorer seed")
        require(self.cache_key == self.semantic_key(), "score cache identity mismatch")
        require(
            self.request_id == identity([self.sample_id, self.cache_key]),
            "wire request identity mismatch",
        )

    def semantic_key(self):
        return identity(
            {
                "prompt": self.prompt,
                "image_sha256": self.image.sha256,
            }
        )

    @classmethod
    def create(cls, sample_id, prompt, image, seed=None):
        key = identity(
            {
                "prompt": prompt,
                "image_sha256": image.sha256,
            }
        )
        return cls(
            sample_id,
            identity([sample_id, key]),
            key,
            prompt,
            image,
            seed,
        )


@dataclass(frozen=True)
class ScoreError:
    code: str
    message: str
    retryable: bool

    def __post_init__(self):
        require(bool(self.code) and bool(self.message), "typed error required")


@dataclass(frozen=True)
class ScoreResult:
    sample_id: str
    request_id: str
    cache_key: str
    status: str
    values: dict[str, float]
    error: ScoreError | None
    metadata: dict[str, object]

    def __post_init__(self):
        require(self.status in {"ok", "failed"}, "unknown score status")
        require(
            self.request_id == identity([self.sample_id, self.cache_key]),
            "result wire identity mismatch",
        )
        if self.status == "ok":
            require(
                bool(self.values) and self.error is None,
                "successful result requires values and no error",
            )
            require(
                all(
                    type(x) in (int, float) and math.isfinite(x)
                    for x in self.values.values()
                ),
                "nonfinite score",
            )
        else:
            require(
                not self.values and self.error is not None,
                "failed result requires error and no values",
            )

    @classmethod
    def for_request(cls, request, *, values=None, error=None, metadata=None):
        return cls(
            request.sample_id,
            request.request_id,
            request.cache_key,
            "failed" if error else "ok",
            values or {},
            error,
            metadata or {},
        )


def validate_results(requests, results, outputs, *, complete):
    expected = {r.request_id: r for r in requests}
    require(len(expected) == len(requests), "duplicate wire IDs")
    channels = {c.name: c for c in outputs}
    joined = {}
    for result in results:
        require(
            result.request_id in expected and result.request_id not in joined,
            "unknown or duplicate score ID",
        )
        request = expected[result.request_id]
        require(
            (result.sample_id, result.cache_key)
            == (request.sample_id, request.cache_key),
            "score belongs to different inputs",
        )
        if result.status == "ok":
            require(
                result.values.keys() == channels.keys(),
                "missing or undeclared score channels",
            )
        joined[result.request_id] = result
    if complete:
        require(
            len(joined) == len(expected)
            and all(r.status == "ok" for r in joined.values()),
            "incomplete scoring; update forbidden",
        )
    return joined


def score_record(service_id, result, attempts):
    return {"scorer": service_id, "attempts": attempts, **asdict(result)}
