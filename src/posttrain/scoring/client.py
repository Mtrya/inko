"""Backend-independent HTTP acquisition with durable per-request retry budgets."""

import base64
import json
import subprocess
import sys
import threading
import time
import urllib.error
from dataclasses import asdict, replace
from pathlib import Path

from ..records import atomic_json, identity
from ..schema import decode, require
from .contracts import (
    PROTOCOL,
    Capabilities,
    ImageInput,
    ScoreError,
    ScoreRequest,
    ScoreResult,
    score_record,
    validate_results,
)


def http_json(endpoint, route, timeout, payload=None):
    # The transport process has a hard elapsed deadline, including DNS, TLS,
    # headers and a trickling body. It never imports model dependencies.
    command = [sys.executable, "-m", "src.posttrain.scoring.transport"]
    try:
        result = subprocess.run(
            command,
            input=json.dumps(
                {
                    "endpoint": endpoint,
                    "route": route,
                    "timeout": timeout,
                    "payload": payload,
                },
                allow_nan=False,
            ),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("absolute HTTP request deadline exceeded") from exc
    if result.returncode:
        raise OSError(f"score transport failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def request_for_record(record, spec):
    payload = record["image_path"]
    if spec.transport == "png_base64":
        payload = base64.b64encode(Path(payload).read_bytes()).decode()
    return ScoreRequest.create(
        record["sample_id"],
        record["prompt"]["text"],
        ImageInput(spec.transport, record["image_sha256"], payload),
        spec.fingerprint,
        spec.input_schema,
        {
            field: record["prompt"].get("structured", {})[field]
            for field in spec.structured_fields
        },
        seed=None
        if spec.seed_policy == "none"
        else int(identity([spec.seed, record["sample_id"]])[:16], 16) % 2**63,
    )


class ScoreClient:
    def __init__(self, spec, endpoints, directory):
        require(
            bool(endpoints)
            and all(e.startswith(("http://", "https://")) for e in endpoints),
            "service endpoints required",
        )
        self.spec, self.endpoints = spec, tuple(endpoints)
        self.directory = Path(directory)
        self.slots = threading.BoundedSemaphore(spec.concurrency)
        self.lock, self.cursor = threading.Lock(), 0
        self.capabilities = []

    def handshake(self, deadline=None):
        deadline = (
            time.monotonic() + self.spec.timeout if deadline is None else deadline
        )
        self.capabilities = []
        for endpoint in self.endpoints:
            remaining = min(self.spec.timeout, deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("service handshake deadline exceeded")
            raw = http_json(endpoint, "/v1/capabilities", remaining)
            if time.monotonic() >= deadline:
                raise TimeoutError("service handshake deadline exceeded")
            cap = decode(Capabilities, raw)
            require(
                cap.fingerprint == self.spec.fingerprint
                and cap.input_schema == self.spec.input_schema
                and cap.structured_fields == self.spec.structured_fields,
                "service fingerprint/input schema mismatch",
            )
            require(cap.outputs == self.spec.outputs, "service output schema mismatch")
            require(
                cap.stable_replay
                and cap.deterministic == (self.spec.seed_policy == "none"),
                "service seed policy/stable replay mismatch",
            )
            require(
                self.spec.transport in cap.transports
                and self.spec.batch_size <= cap.max_batch_size,
                "service transport/batch mismatch",
            )
            self.capabilities.append(cap)
        return [asdict(c) for c in self.capabilities]

    def _path(self, request):
        return self.directory / "requests" / f"{request.request_id}.json"

    def saved(self, request):
        path = self._path(request)
        if not path.exists():
            return 0, None
        raw = json.loads(path.read_text())
        require(
            set(raw) == {"attempts", "result", "request_id"}
            and raw["request_id"] == request.request_id,
            "invalid acquisition journal",
        )
        attempts = raw["attempts"]
        require(
            type(attempts) is int and 0 <= attempts <= self.spec.attempts,
            "invalid persisted attempt budget",
        )
        result = None if raw["result"] is None else decode(ScoreResult, raw["result"])
        if result:
            validate_results([request], [result], self.spec.outputs, complete=False)
        return attempts, result

    def _persist(self, request, attempts, result):
        atomic_json(
            self._path(request),
            {
                "request_id": request.request_id,
                "attempts": attempts,
                "result": None if result is None else asdict(result),
            },
        )
        if result is not None and result.status == "ok":
            atomic_json(
                self.directory / "cache" / f"{request.cache_key}.json", asdict(result)
            )

    def acquire(self, requests):
        require(bool(self.capabilities), "handshake required before acquisition")
        validate_results(requests, [], self.spec.outputs, complete=False)
        for request in requests:
            require(
                request.fingerprint == self.spec.fingerprint
                and request.input_schema == self.spec.input_schema
                and set(request.structured) == set(self.spec.structured_fields)
                and ((request.seed is None) == (self.spec.seed_policy == "none")),
                "request differs from service contract",
            )
            request.image.read()
        final = {}
        attempts = {}
        pending = []
        for request in requests:
            count, result = self.saved(request)
            cache = self.directory / "cache" / f"{request.cache_key}.json"
            if (result is None or result.status != "ok") and cache.exists():
                hit = decode(ScoreResult, json.loads(cache.read_text()))
                result = replace(
                    hit, sample_id=request.sample_id, request_id=request.request_id
                )
                validate_results([request], [result], self.spec.outputs, complete=True)
                self._persist(request, count, result)
            attempts[request.request_id] = count
            if result and (result.status == "ok" or not result.error.retryable):
                final[request.request_id] = result
            elif count >= self.spec.attempts:
                result = result or ScoreResult.for_request(
                    request,
                    error=ScoreError(
                        "interrupted",
                        "request budget consumed before interruption",
                        False,
                    ),
                )
                final[request.request_id] = result
                self._persist(request, count, result)
            else:
                pending.append(request)
        with self.slots:
            while pending:
                batch, pending = (
                    pending[: self.spec.batch_size],
                    pending[self.spec.batch_size :],
                )
                with self.lock:
                    endpoint = self.endpoints[self.cursor % len(self.endpoints)]
                    self.cursor += 1
                for request in batch:
                    attempts[request.request_id] += 1
                    # Charge before transport; a crash cannot reset a consumed attempt.
                    self._persist(request, attempts[request.request_id], None)
                try:
                    raw = http_json(
                        endpoint,
                        "/v1/score",
                        self.spec.timeout,
                        {
                            "protocol": PROTOCOL,
                            "deadline": time.time() + self.spec.timeout,
                            "requests": [asdict(r) for r in batch],
                        },
                    )
                    require(
                        set(raw) == {"protocol", "results"}
                        and raw["protocol"] == PROTOCOL
                        and isinstance(raw["results"], list),
                        "invalid score response",
                    )
                    # Envelope corruption aborts instead of accepting possibly misjoined values.
                    expected = {r.request_id: r for r in batch}
                    ids = [r.get("request_id") for r in raw["results"]]
                    require(
                        len(ids) == len(set(ids)) and set(ids) <= expected.keys(),
                        "unknown or duplicate score ID",
                    )
                    joined = {}
                    for item in raw["results"]:
                        request = expected[item["request_id"]]
                        require(
                            all(
                                item.get(k) == getattr(request, k)
                                for k in ("sample_id", "cache_key", "fingerprint")
                            ),
                            "score belongs to different inputs",
                        )
                        try:
                            result = decode(ScoreResult, item)
                            validate_results(
                                [request], [result], self.spec.outputs, complete=False
                            )
                        except ValueError as exc:
                            result = ScoreResult.for_request(
                                request,
                                error=ScoreError("invalid_value", str(exc), True),
                            )
                        joined[request.request_id] = result
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.URLError,
                ) as exc:
                    joined = {
                        r.request_id: ScoreResult.for_request(
                            r, error=ScoreError("transport", str(exc), True)
                        )
                        for r in batch
                    }
                for request in batch:
                    result = joined.get(request.request_id) or ScoreResult.for_request(
                        request,
                        error=ScoreError("missing", "service omitted result", True),
                    )
                    self._persist(request, attempts[request.request_id], result)
                    if (
                        result.status == "failed"
                        and result.error.retryable
                        and attempts[request.request_id] < self.spec.attempts
                    ):
                        pending.append(request)
                    else:
                        final[request.request_id] = result
        return [
            score_record(self.spec.id, final[r.request_id], attempts[r.request_id])
            for r in requests
        ]
