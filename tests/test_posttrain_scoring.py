import base64
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.posttrain.records import file_hash, identity
from src.posttrain.schema import decode
from src.posttrain.scoring.client import ScoreClient
from src.posttrain.scoring.config import Limits
from src.posttrain.scoring.contracts import (
    Capabilities,
    ImageInput,
    ScoreRequest,
    ScoreResult,
    validate_results,
)
from src.posttrain.scoring.service import ScoringEngine, make_server


@pytest.fixture
def image_input(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (20, 100, 200)).save(path)
    return ImageInput("shared_file", file_hash(path), str(path))


def request(spec, image, sample="a", prompt="text", rubric="identify content"):
    return ScoreRequest.create(
        sample, prompt, image, spec.fingerprint, spec.input_schema, {"rubric": rubric}
    )


class Endpoint:
    def __init__(self, tmp_path, spec, backend, engine_type=ScoringEngine, port=0):
        cap = Capabilities(
            1,
            spec.fingerprint,
            spec.input_schema,
            spec.structured_fields,
            spec.outputs,
            2,
            ("shared_file", "png_base64"),
            True,
            True,
        )
        self.engine = engine_type(
            backend, cap, Limits(2, 4, 2.0, 1000000, 1), tmp_path / "service-cache"
        )
        self.server = make_server(("127.0.0.1", port), self.engine)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.engine.close()


class Backend:
    def __init__(self):
        self.calls = []

    def score(self, inputs):
        self.calls.append([i.prompt for i in inputs])
        return [
            {"fit": 0.75 if i.structured["rubric"] == "identify content" else 0.25}
            for i in inputs
        ]


def test_real_http_cache_fanout_relocation_and_service_restart(
    tmp_path, sampling_config, image_input
):
    spec = sampling_config.scorers[0]
    backend = Backend()
    endpoint = Endpoint(tmp_path, spec, backend)
    records = [request(spec, image_input, "a"), request(spec, image_input, "b")]
    try:
        client = ScoreClient(spec, [endpoint.url], tmp_path / "client")
        client.handshake()
        first = client.acquire(records)
        assert [r["values"] for r in first] == [{"fit": 0.75}, {"fit": 0.75}]
        assert first[0]["request_id"] != first[1]["request_id"]
        assert first[0]["cache_key"] == first[1]["cache_key"]
        assert backend.calls == [["text"]]
    finally:
        endpoint.close()
    # Fresh endpoint, client and replica placement preserve semantic result identity.
    endpoint = Endpoint(tmp_path, spec, backend)
    try:
        client = ScoreClient(spec, [endpoint.url], tmp_path / "client")
        client.handshake()
        assert client.acquire(records) == first
        renamed = tmp_path / "relocated.png"
        renamed.write_bytes(Path(image_input.payload).read_bytes())
        moved = request(spec, replace(image_input, payload=str(renamed)), "third")
        assert moved.cache_key == records[0].cache_key
        assert client.acquire([moved])[0]["attempts"] == 0
        assert backend.calls == [["text"]]
        changed = request(spec, image_input, "fourth", rubric="different rubric")
        assert changed.cache_key != records[0].cache_key
        assert client.acquire([changed])[0]["values"] == {"fit": 0.25}
    finally:
        endpoint.close()


def test_partial_out_of_order_retry_preserves_success(
    tmp_path, sampling_config, image_input
):
    class Partial(ScoringEngine):
        def __init__(self, *args):
            self.calls = []
            super().__init__(*args)

        def _score(self, requests, deadline, enqueued):
            self.calls.append([r.sample_id for r in requests])
            results = super()._score(requests, deadline, enqueued)
            return list(
                reversed(
                    [r for r in results if r.sample_id != "b" or len(self.calls) > 1]
                )
            )

    spec = sampling_config.scorers[0]
    endpoint = Endpoint(tmp_path, spec, Backend(), Partial)
    try:
        client = ScoreClient(spec, [endpoint.url], tmp_path / "client")
        client.handshake()
        rows = client.acquire(
            [request(spec, image_input, "a"), request(spec, image_input, "b")]
        )
        assert [r["attempts"] for r in rows] == [1, 2]
        assert endpoint.engine.calls == [["a", "b"], ["b"]]
    finally:
        endpoint.close()


def test_failure_budget_survives_restart_and_never_becomes_zero(
    tmp_path, sampling_config, image_input
):
    class Failing(Backend):
        def score(self, inputs):
            self.calls.append(inputs)
            raise RuntimeError("unavailable")

    spec, backend = sampling_config.scorers[0], Failing()
    endpoint = Endpoint(tmp_path, spec, backend)
    try:
        requests = [request(spec, image_input)]
        for _ in range(2):
            client = ScoreClient(spec, [endpoint.url], tmp_path / "client")
            client.handshake()
            row = client.acquire(requests)[0]
            assert (
                row["status"] == "failed"
                and row["values"] == {}
                and row["attempts"] == 2
            )
        assert len(backend.calls) == 2
        result = decode(
            ScoreResult,
            {k: v for k, v in row.items() if k not in {"attempts", "scorer"}},
        )
        with pytest.raises(ValueError, match="forbidden"):
            validate_results(requests, [result], spec.outputs, complete=True)
    finally:
        endpoint.close()


def test_encoded_png_schema_mismatch_and_content_hash(
    tmp_path, sampling_config, image_input
):
    spec = sampling_config.scorers[0]
    endpoint = Endpoint(tmp_path, spec, Backend())
    try:
        with pytest.raises(ValueError, match="schema mismatch"):
            ScoreClient(
                replace(spec, input_schema="another"), [endpoint.url], tmp_path
            ).handshake()
        client = ScoreClient(spec, [endpoint.url], tmp_path / "client")
        client.handshake()
        encoded = replace(
            image_input,
            transport="png_base64",
            payload=base64.b64encode(Path(image_input.payload).read_bytes()).decode(),
        )
        assert request(spec, encoded).cache_key == request(spec, image_input).cache_key
        assert client.acquire([request(spec, encoded)])[0]["values"] == {"fit": 0.75}
        with pytest.raises(ValueError, match="bytes changed"):
            client.acquire([request(spec, replace(image_input, sha256="0" * 64))])
    finally:
        endpoint.close()


@pytest.mark.parametrize(
    "values", [{"fit": 1.1}, {"wrong": 0.5}, {"fit": True}, {"fit": float("nan")}]
)
def test_channel_value_validation(sampling_config, image_input, values):
    spec = sampling_config.scorers[0]
    r = request(spec, image_input)
    with pytest.raises(ValueError):
        result = ScoreResult.for_request(r, values=values)
        validate_results([r], [result], spec.outputs, complete=True)


def test_duplicate_and_wrong_envelopes_rejected(sampling_config, image_input):
    spec = sampling_config.scorers[0]
    r = request(spec, image_input)
    result = ScoreResult.for_request(r, values={"fit": 0.3})
    with pytest.raises(ValueError, match="duplicate"):
        validate_results([r], [result, result], spec.outputs, complete=False)
    with pytest.raises(ValueError, match="duplicate"):
        validate_results([r, r], [], spec.outputs, complete=False)
    with pytest.raises(ValueError, match="wire"):
        replace(r, request_id=identity("wrong"))


def test_client_import_is_model_free():
    import sys

    code = 'import sys; import src.posttrain.scoring.client; assert not ({"torch", "transformers", "hpsv3", "t2v_metrics"} & sys.modules.keys())'
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hps_head_boundary_preserves_fp32_projection():
    import torch

    from src.posttrain.scoring.backends.hpsv3 import float32_head_inputs

    head = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[2.0, 3.0]]))
    hidden = torch.tensor([[4.0, 5.0]], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="same dtype"):
        head(hidden)
    head.register_forward_pre_hook(float32_head_inputs)
    assert head(hidden).item() == 23.0


def test_production_orchestration_config_only_substitution(
    tmp_path, sampling_config, image_input
):
    from src.posttrain.scoring.contracts import Channel
    from src.posttrain.workers import ResidentWorkers, run_pipeline

    class RubricEngine(ScoringEngine):
        def _score(self, requests, deadline, enqueued):
            return [
                replace(
                    r,
                    metadata={
                        **r.metadata,
                        "rubric": {"checks": [{"name": "content", "passed": True}]},
                    },
                )
                for r in super()._score(requests, deadline, enqueued)
            ]

    record = {
        "sample_id": "sample",
        "image_path": image_input.payload,
        "image_sha256": image_input.sha256,
        "prompt": {"text": "text", "structured": {"rubric": "identify content"}},
    }
    for identifier, metric in (("alien", "rubric_fit"), ("other", "agreement")):
        spec = replace(
            sampling_config.scorers[0],
            id=identifier,
            outputs=(Channel(metric, "higher", (0.0, 1.0), "Fixture channel"),),
        )

        class GenericBackend:
            def score(self, inputs, channel=metric):
                return [{channel: 0.625} for _ in inputs]

        endpoint = Endpoint(tmp_path / identifier, spec, GenericBackend(), RubricEngine)
        config = replace(
            sampling_config,
            scorers=(spec,),
            evaluation=replace(
                sampling_config.evaluation, channels=(f"{identifier}.{metric}",)
            ),
        )
        assets = {
            "services": {identifier: {"endpoints": [endpoint.url], "launches": []}}
        }
        scored = []
        try:
            with ResidentWorkers(
                config, assets, tmp_path / identifier / "caller", generate=False
            ) as workers:
                run_pipeline(
                    [{"task_id": "collection", "records": [record]}],
                    workers,
                    config,
                    lambda rows: pytest.fail("existing images must not regenerate"),
                    lambda role, rows, results=scored: results.extend(rows),
                )
            assert scored[0]["scorer"] == identifier
            assert scored[0]["values"] == {metric: 0.625}
            assert scored[0]["metadata"]["rubric"]["checks"][0]["passed"] is True
        finally:
            endpoint.close()


def test_absolute_deadline_stops_trickling_http_body():
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from src.posttrain.scoring.client import http_json

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "10000")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises((TimeoutError, OSError)):
            http_json(f"http://127.0.0.1:{server.server_port}", "/v1/health", 0.1)
        assert time.monotonic() - started < 1
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_service_stall_retires_engine(tmp_path, sampling_config, image_input):
    import time

    released, stalled = threading.Event(), threading.Event()

    class Hung:
        def score(self, inputs):
            released.wait(1)
            return [{"fit": 0.5}]

    spec = sampling_config.scorers[0]
    cap = Capabilities(
        1,
        spec.fingerprint,
        spec.input_schema,
        spec.structured_fields,
        spec.outputs,
        1,
        ("shared_file",),
        True,
        True,
    )
    engine = ScoringEngine(
        Hung(), cap, Limits(1, 1, 0.1, 10000, 1), tmp_path, on_stall=stalled.set
    )
    try:
        engine.submit([request(spec, image_input)], time.time() + 0.1)
        assert stalled.wait(1) and engine.poisoned
        with pytest.raises(ValueError, match="stalled"):
            engine.submit([request(spec, image_input)], time.time() + 1)
    finally:
        released.set()
        engine.close()


def test_stable_seeded_service_uses_same_generic_client(
    tmp_path, sampling_config, image_input
):
    from src.posttrain.scoring.client import request_for_record

    spec = replace(sampling_config.scorers[0], seed_policy="sample", seed=41)
    cap = Capabilities(
        1,
        spec.fingerprint,
        spec.input_schema,
        spec.structured_fields,
        spec.outputs,
        2,
        ("shared_file",),
        False,
        True,
    )

    class Seeded:
        def __init__(self):
            self.calls = []

        def score(self, inputs):
            self.calls.extend(i.seed for i in inputs)
            return [{"fit": (i.seed % 2) / 2} for i in inputs]

    backend = Seeded()
    engine = ScoringEngine(
        backend, cap, Limits(2, 2, 2, 100000, 1), tmp_path / "service"
    )
    server = make_server(("127.0.0.1", 0), engine)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        client = ScoreClient(
            spec, [f"http://127.0.0.1:{server.server_port}"], tmp_path / "client"
        )
        client.handshake()
        record = {
            "sample_id": "a",
            "image_path": image_input.payload,
            "image_sha256": image_input.sha256,
            "prompt": {"text": "text", "structured": {"rubric": "identify content"}},
        }
        first = request_for_record(record, spec)
        second = request_for_record(record, replace(spec, seed=42))
        assert (
            first.seed is not None
            and first.seed != second.seed
            and first.cache_key != second.cache_key
        )
        result = client.acquire([first])
        assert client.acquire([first]) == result and backend.calls == [first.seed]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        engine.close()


def test_each_service_receives_only_its_declared_structured_inputs(
    sampling_config, image_input
):
    from src.posttrain.scoring.client import request_for_record

    record = {
        "sample_id": "a",
        "image_path": image_input.payload,
        "image_sha256": image_input.sha256,
        "prompt": {
            "text": "text",
            "structured": {
                "rubric": {"criteria": ["content"]},
                "context": "other service",
            },
        },
    }
    spec = sampling_config.scorers[0]
    assert request_for_record(record, spec).structured == {
        "rubric": {"criteria": ["content"]}
    }
    assert (
        request_for_record(record, replace(spec, structured_fields=())).structured == {}
    )


def test_handshake_shares_remaining_startup_deadline(
    tmp_path, sampling_config, monkeypatch
):
    from dataclasses import asdict

    from src.posttrain.scoring import client as module

    spec = sampling_config.scorers[0]
    cap = Capabilities(
        1,
        spec.fingerprint,
        spec.input_schema,
        spec.structured_fields,
        spec.outputs,
        2,
        ("shared_file",),
        True,
        True,
    )
    clock, timeouts = [10.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def response(endpoint, route, timeout):
        timeouts.append(timeout)
        clock[0] += 0.75
        return asdict(cap)

    monkeypatch.setattr(module, "http_json", response)
    client = ScoreClient(spec, ["http://one", "http://two"], tmp_path)
    with pytest.raises(TimeoutError, match="handshake deadline"):
        client.handshake(deadline=11.0)
    assert timeouts == [1.0, 0.25]
    assert len(client.capabilities) == 1


@pytest.mark.parametrize("trickle", [False, True])
def test_service_releases_stalled_header_slots(tmp_path, sampling_config, trickle):
    import socket
    import time

    from src.posttrain.scoring.client import http_json

    endpoint = Endpoint(tmp_path, sampling_config.scorers[0], Backend())
    endpoint.engine.limits = replace(endpoint.engine.limits, request_timeout=0.15)
    sockets = []
    started = time.monotonic()
    try:
        for _ in range(5):
            sock = socket.create_connection(endpoint.server.server_address, timeout=1)
            sock.sendall(b"GET /v1/health HTTP/1.0\r\nX-Stalled: ")
            sockets.append(sock)
        if trickle:
            while time.monotonic() - started < 0.4:
                for sock in sockets:
                    try:
                        sock.sendall(b"x")
                    except OSError:
                        pass
                time.sleep(0.02)
        for sock in sockets:
            # Header parsing may emit a response on EOF; either way it must close.
            while sock.recv(65536):
                pass
        assert time.monotonic() - started < 1.0
        assert http_json(endpoint.url, "/v1/health", 1.0)["ready"] is True
    finally:
        for sock in sockets:
            sock.close()
        endpoint.close()
