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


def request(image, sample="a", prompt="text"):
    return ScoreRequest.create(sample, prompt, image)


class Endpoint:
    def __init__(self, tmp_path, spec, backend, engine_type=ScoringEngine, port=0):
        cap = Capabilities(
            spec.outputs,
            2,
            ("shared_file", "png_base64"),
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
        return [{"fit": 0.75 if i.prompt == "text" else 0.25} for i in inputs]


def test_real_http_cache_fanout_relocation_and_service_restart(
    tmp_path, sampling_config, image_input
):
    spec = sampling_config.scorers[0]
    backend = Backend()
    endpoint = Endpoint(tmp_path, spec, backend)
    records = [request(image_input, "a"), request(image_input, "b")]
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
        moved = request(replace(image_input, payload=str(renamed)), "third")
        assert moved.cache_key == records[0].cache_key
        assert client.acquire([moved])[0]["attempts"] == 0
        assert backend.calls == [["text"]]
        changed = request(image_input, "fourth", prompt="different text")
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
        rows = client.acquire([request(image_input, "a"), request(image_input, "b")])
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
        requests = [request(image_input)]
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


def test_encoded_png_transport_and_content_hash(tmp_path, sampling_config, image_input):
    spec = sampling_config.scorers[0]
    endpoint = Endpoint(tmp_path, spec, Backend())
    try:
        client = ScoreClient(spec, [endpoint.url], tmp_path / "client")
        client.handshake()
        encoded = replace(
            image_input,
            transport="png_base64",
            payload=base64.b64encode(Path(image_input.payload).read_bytes()).decode(),
        )
        assert request(encoded).cache_key == request(image_input).cache_key
        assert client.acquire([request(encoded)])[0]["values"] == {"fit": 0.75}
        with pytest.raises(ValueError, match="bytes changed"):
            client.acquire([request(replace(image_input, sha256="0" * 64))])
    finally:
        endpoint.close()


def test_handshake_rejects_mismatched_service_outputs(
    tmp_path, sampling_config, image_input
):
    from src.posttrain.scoring.contracts import Channel

    spec = sampling_config.scorers[0]
    endpoint = Endpoint(tmp_path, spec, Backend())
    try:
        alien = replace(spec, outputs=(Channel("probability", "higher"),))
        with pytest.raises(ValueError, match="output schema mismatch"):
            ScoreClient(alien, [endpoint.url], tmp_path).handshake()
    finally:
        endpoint.close()


@pytest.mark.parametrize(
    "values", [{"wrong": 0.5}, {"fit": True}, {"fit": float("nan")}]
)
def test_channel_value_validation(sampling_config, image_input, values):
    spec = sampling_config.scorers[0]
    r = request(image_input)
    with pytest.raises(ValueError):
        result = ScoreResult.for_request(r, values=values)
        validate_results([r], [result], spec.outputs, complete=True)


def test_duplicate_and_wrong_envelopes_rejected(sampling_config, image_input):
    spec = sampling_config.scorers[0]
    r = request(image_input)
    result = ScoreResult.for_request(r, values={"fit": 0.3})
    with pytest.raises(ValueError, match="duplicate"):
        validate_results([r], [result, result], spec.outputs, complete=False)
    with pytest.raises(ValueError, match="duplicate"):
        validate_results([r, r], [], spec.outputs, complete=False)
    with pytest.raises(ValueError, match="wire"):
        replace(r, request_id=identity("wrong"))


def test_round_acquisition_returns_combined_rewards(
    tmp_path, training_config, image_input
):
    from src.posttrain.round_scoring import acquire_round

    spec = training_config.scorers[0]
    endpoint = Endpoint(tmp_path, spec, Backend())
    sample_ids = [f"0:{slot}:{draw}" for slot in range(2) for draw in range(2)]
    records = [
        {
            "sample_id": key,
            "prompt": {"id": key.rsplit(":", 1)[0], "text": "text"},
            "image_path": image_input.payload,
            "image_sha256": image_input.sha256,
        }
        for key in sample_ids
    ]
    try:
        client = ScoreClient(spec, [endpoint.url], tmp_path / "client")
        client.handshake()
        result = acquire_round(training_config, {spec.id: client}, records)
        reward = result["reward"]
        assert reward["sample_ids"] == sorted(sample_ids)
        assert reward["combined"] == [0.75] * 4
        assert len(result["scores"]) == len(records)
    finally:
        endpoint.close()


def test_mixed_pairwise_and_continuous_reward_channels(training_config):
    import math

    import torch

    from src.posttrain.config import Scorer
    from src.posttrain.rewards import combine_round
    from src.posttrain.scoring.contracts import Channel
    from src.posttrain.training_config import RewardChannel, Rewards

    rubric = {
        "content": [1, 0.5, 0.5, 0],
        "form": [0.5, 0.5, 0.5, 0.5],
        "composition": [0, 1, 0.5, 0.5],
        "style": [1, 0, 1, 0],
        "defect_cost": [0, 0.5, 0.5, 1],
    }
    scorers = (
        Scorer(
            "judge",
            tuple(
                Channel(k, "lower" if k == "defect_cost" else "higher") for k in rubric
            ),
            1,
            1.0,
            1,
            4,
            "shared_file",
        ),
        Scorer("hps", (Channel("mu", "higher"),), 1, 1.0, 1, 4, "shared_file"),
        Scorer("vqa", (Channel("probability", "higher"),), 1, 1.0, 1, 4, "shared_file"),
    )
    channels = tuple(
        RewardChannel(f"judge.{k}", 0.4 if k == "content" else 0.1, "none")
        for k in rubric
    ) + (
        RewardChannel("hps.mu", 0.1, "round"),
        RewardChannel("vqa.probability", 0.1, "round"),
    )
    config = replace(training_config, scorers=scorers, rewards=Rewards(channels, 0.5))
    ids = [str(i) for i in range(4)]
    scores = []
    for i, key in enumerate(ids):
        for scorer, values in (
            ("judge", {k: v[i] for k, v in rubric.items()}),
            ("hps", {"mu": 2 + 2 * i}),
            ("vqa", {"probability": i / 4}),
        ):
            scores.append(
                {"sample_id": key, "scorer": scorer, "status": "ok", "values": values}
            )
    result = combine_round(ids, scores[::-1], config)
    expected = torch.tensor([0.55, 0.30, 0.35, 0])
    expected += 0.1 * torch.tensor([-3, -1, 1, 3]) / (math.sqrt(5) + 0.5)
    expected += (
        0.1 * torch.tensor([-0.375, -0.125, 0.125, 0.375]) / (math.sqrt(5) / 8 + 0.5)
    )
    torch.testing.assert_close(torch.tensor(result["combined"]), expected)
    assert result["statistics"]["judge.content"]["transformed"] == [1, 0.5, 0.5, 0]
    assert result["statistics"]["judge.defect_cost"]["direction"] == "lower"


def test_round_barrier_joins_the_published_reward_packet(tmp_path, training_config):
    from src.posttrain.records import atomic_json
    from src.posttrain.round_scoring import (
        collection_path,
        reward_path,
        wait_for_rewards,
    )

    output, attempt, step = tmp_path, "a", 3
    local = [{"sample_id": "3:0:0"}]
    collection = collection_path(output, attempt, step, 0)
    atomic_json(collection, {"step": step, "rank": 0, "records": local})
    packet = {
        "step": step,
        "policy_version": step,
        "collections": [file_hash(collection)],
        "reward": {"sample_ids": ["3:0:0", "3:0:1"], "combined": [1.0, -1.0]},
    }
    atomic_json(reward_path(output, attempt, step), packet)
    values, reward = wait_for_rewards(training_config, output, attempt, step, 0, local)
    assert values == {"3:0:0": 1.0, "3:0:1": -1.0} and reward == packet["reward"]
    with pytest.raises(ValueError, match="missing local reward"):
        wait_for_rewards(
            training_config, output, attempt, step, 0, [{"sample_id": "3:9:9"}]
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        collection.write_text("changed")
        wait_for_rewards(training_config, output, attempt, step, 0, local)


@pytest.mark.parametrize("resume", [False, True])
def test_coordinator_scores_additional_art_collections_and_resumes_after_skips(
    tmp_path, monkeypatch, training_config, resume
):
    import os
    import time

    from src.posttrain import round_scoring, train
    from src.posttrain.records import atomic_json
    from src.posttrain.round_scoring import collection_path, reward_path
    from src.posttrain.scoring import lifecycle

    output = tmp_path / "training"
    first_round, generator_step = (2, 2) if resume else (0, 0)
    checkpoint = None
    if resume:
        checkpoint = tmp_path / "checkpoint"
        atomic_json(
            checkpoint / "manifest.json",
            {"step": 2, "committed_rounds": 2, "world_size": 1, "contract": {}},
        )

    class Services:
        def __init__(self, *args):
            self.startup = {}
            self.clients = {}

        def __enter__(self):
            return self

        def check(self):
            pass

        def close(self):
            pass

    class Learner:
        returncode = None

        def __init__(self, argv, **kwargs):
            self.attempt = argv[argv.index("--attempt") + 1]
            self.calls = 0
            self.second_collection = False
            ready = output / "ready" / f"{self.attempt}-rank-000.json"
            atomic_json(ready, {})
            old = time.time() - 2 * training_config.execution.round_timeout
            os.utime(ready, (old, old))
            self.publish(first_round)

        def publish(self, r):
            atomic_json(
                output / "attempts" / f"{self.attempt}-round-{r:06d}.json",
                {"step": generator_step, "round": r},
            )
            atomic_json(
                collection_path(output, self.attempt, r, 0),
                {
                    "step": r,
                    "rank": 0,
                    "records": [
                        {
                            "sample_id": f"{r}:{slot}:{draw}",
                            "generator_step": generator_step,
                            "policy_version": r,
                        }
                        for slot in (0, 1)
                        for draw in (0, 1)
                    ],
                },
            )

        def poll(self):
            self.calls += 1
            assert self.calls < 30, (
                "coordinator stopped scoring before the update horizon"
            )
            if reward_path(output, self.attempt, first_round + 1).exists():
                self.returncode = 0
            elif (
                reward_path(output, self.attempt, first_round).exists()
                and not self.second_collection
            ):
                # A committed skip is progress even when startup and updates are old.
                atomic_json(
                    output
                    / "rounds"
                    / f"{self.attempt}-round-{first_round:06d}-rank-000.json",
                    {"skipped_zero_guidance": True},
                )
                self.publish(first_round + 1)
                self.second_collection = True
            return self.returncode

        def wait(self, timeout):
            return self.returncode

    scored = []

    def acquire(config, clients, rows, *, on_scored=None):
        scored.append(rows)
        return {"reward": {}, "scores": [], "acquisition_seconds": 0.0}

    monkeypatch.setattr(lifecycle, "Services", Services)
    monkeypatch.setattr(train.subprocess, "Popen", Learner)
    monkeypatch.setattr(train.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(round_scoring, "acquire_round", acquire)
    result = train.launch(
        replace(training_config, training=replace(training_config.training, rounds=4)),
        {"learner": {"python_paths": []}, "device_type": "cpu"},
        {},
        output,
        checkpoint,
        first_round + 2,
        tmp_path / "config.toml",
        tmp_path / "assets.json",
    )
    assert result["completed_rounds"] == first_round + 2
    assert [[r["sample_id"] for r in rows] for rows in scored] == [
        [f"{r}:{slot}:{draw}" for slot in (0, 1) for draw in (0, 1)]
        for r in (first_round, first_round + 1)
    ]


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


def test_vqa_backend_uses_fixed_yes_probability_contract():
    from types import SimpleNamespace

    import torch

    from src.posttrain.scoring.backends.vqascore import VQAScore

    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        return torch.tensor([0.25, 0.75])

    backend = VQAScore.__new__(VQAScore)
    backend.scorer = SimpleNamespace(forward=forward)
    rows = [
        SimpleNamespace(image_path="a.png", prompt="a cat"),
        SimpleNamespace(image_path="b.png", prompt="a dog"),
    ]
    assert backend.score(rows) == [{"probability": 0.25}, {"probability": 0.75}]
    assert calls == [
        {
            "images": ["a.png", "b.png"],
            "texts": ["a cat", "a dog"],
            "question_template": 'Does this figure show "{}"? Please answer Yes or No.',
            "answer_template": "Yes",
            "max_new_tokens": 1,
            "temperature": 1.0,
        }
    ]


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
        "prompt": {"text": "text"},
    }
    for identifier, metric in (("alien", "rubric_fit"), ("other", "agreement")):
        spec = replace(
            sampling_config.scorers[0],
            id=identifier,
            outputs=(Channel(metric, "higher"),),
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
        spec.outputs,
        1,
        ("shared_file",),
    )
    engine = ScoringEngine(
        Hung(), cap, Limits(1, 1, 0.1, 10000, 1), tmp_path, on_stall=stalled.set
    )
    try:
        engine.submit([request(image_input)], time.time() + 0.1)
        assert stalled.wait(1) and engine.poisoned
        with pytest.raises(ValueError, match="stalled"):
            engine.submit([request(image_input)], time.time() + 1)
    finally:
        released.set()
        engine.close()


def test_handshake_shares_remaining_startup_deadline(
    tmp_path, sampling_config, monkeypatch
):
    from dataclasses import asdict

    from src.posttrain.scoring import client as module

    spec = sampling_config.scorers[0]
    cap = Capabilities(
        spec.outputs,
        2,
        ("shared_file",),
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
