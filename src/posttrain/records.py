"""Serializable identities shared by rollout, score workers, and evaluation."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def identity(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class ScoreRequest:
    sample_id: str
    image_path: str
    image_sha256: str
    prompt: str

    def key(self, scorer_identity: str) -> str:
        return identity({**asdict(self), "image_path": None, "scorer": scorer_identity})


@dataclass(frozen=True)
class ScoreResult:
    sample_id: str
    request_id: str
    scorer: str
    scorer_identity: str
    status: str
    values: dict[str, float]
    attempts: int
    error: str | None

    def __post_init__(self):
        if self.status not in {"ok", "failed"}:
            raise ValueError("unknown score status")
        if self.status == "ok":
            if not self.values or self.error is not None:
                raise ValueError("successful score requires values and no error")
            if any(not math.isfinite(x) for x in self.values.values()):
                raise ValueError("nonfinite score")
        elif self.values or not self.error:
            raise ValueError("failed score requires an error and no values")


def join_scores(
    requests: list[ScoreRequest],
    results: list[ScoreResult],
    scorer_identities: dict[str, str],
    *,
    require_complete: bool,
) -> dict:
    expected = {r.sample_id: r for r in requests}
    if len(expected) != len(requests):
        raise ValueError("duplicate sample ID")
    joined = {sample_id: {} for sample_id in expected}
    for result in results:
        if result.sample_id not in expected or result.scorer not in scorer_identities:
            raise ValueError("unexpected score identity")
        scorer_id = scorer_identities[result.scorer]
        if result.scorer_identity != scorer_id or result.request_id != expected[
            result.sample_id
        ].key(scorer_id):
            raise ValueError("score belongs to different sample/scorer inputs")
        scores = joined[result.sample_id]
        if result.scorer in scores:
            raise ValueError("duplicate score result")
        scores[result.scorer] = result
    if require_complete and any(
        set(scores) != set(scorer_identities)
        or any(r.status != "ok" for r in scores.values())
        for scores in joined.values()
    ):
        raise ValueError("incomplete scoring round; generator update is forbidden")
    return joined
