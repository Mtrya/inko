import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.posttrain.config import RewardChannel, Rewards
from src.posttrain.records import atomic_json
from src.posttrain.rewards import RoundScores, combine_round, reward_path
from src.posttrain.scoring.client import HTTPError, ScoreClient, Submissions
from src.posttrain.scoring.collections import Collections
from src.posttrain.scoring.contracts import image_payload
from src.posttrain.scoring.lifecycle import Replicas, isolated_environment
from src.posttrain.scoring.service import collection_routes, make_server


def test_worker_environment_preserves_native_paths_unless_overridden(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/native/cann:/native/tbe")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "missing-trainer-interface")
    native = isolated_environment()
    assert native["PYTHONPATH"].split(os.pathsep)[:2] == ["/native/cann", "/native/tbe"]
    explicit = isolated_environment(["/overlay/torch29"])
    assert explicit["PYTHONPATH"].split(os.pathsep)[0] == "/overlay/torch29"
    assert "/native" not in explicit["PYTHONPATH"]
    assert "RANK" not in native and "RANK" not in explicit
    assert "GLOO_SOCKET_IFNAME" not in native and "GLOO_SOCKET_IFNAME" not in explicit


def test_replica_dispatch_uses_free_capacity():
    entered, release = threading.Event(), threading.Event()

    class Worker:
        def __init__(self, name):
            self.name = name

        def score(self, requests):
            if self.name == "slow":
                entered.set()
                assert release.wait(5)
            return self.name

    replicas = Replicas([Worker("fast"), Worker("slow")], concurrency=1)
    with ThreadPoolExecutor(2) as threads:
        assert replicas.score([]) == "fast"
        pending = threads.submit(replicas.score, [])
        try:
            assert entered.wait(1)
            assert replicas.score([]) == "fast"
            assert threads.submit(replicas.score, []).result(timeout=1) == "fast"
        finally:
            release.set()
        assert pending.result(timeout=1) == "slow"


def test_service_restart_cleans_inputs_and_sigterm_reaps_workers(tmp_path):
    stale = tmp_path / "inputs" / "collection-interrupted" / "sample-0"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"interrupted image upload")
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            {
                "device_type": "cpu",
                "rubric": {"weights": "fixture"},
                "hpsv3": {},
                "vqascore": {},
            }
        )
    )
    code = """
import json, sys
from src.posttrain.scoring.lifecycle import Runtime
from src.posttrain.scoring.service import main
from src.posttrain.records import atomic_json
spawn = Runtime.spawn
def start(self, name, arguments, environment):
    spawn(self, name, [sys.executable, '-c', 'import time; time.sleep(60)'], None)
    atomic_json(self.output / 'workers.json', [p.pid for p in self.processes])
Runtime.spawn = start
sys.argv = ['service', '--config', 'configs/scoring_server.toml',
            '--assets', sys.argv[1], '--output', sys.argv[2]]
main()
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", code, str(assets), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    workers = []
    try:
        deadline = time.monotonic() + 5
        path = tmp_path / "workers.json"
        while len(workers) < 3:
            assert parent.poll() is None, parent.communicate()
            assert time.monotonic() < deadline
            if path.exists():
                workers = json.loads(path.read_text())
            time.sleep(0.05)
        assert not stale.exists()
        parent.terminate()
        parent.wait(timeout=10)
        assert all(
            not (stat := Path(f"/proc/{pid}/stat")).exists()
            or stat.read_text().split()[2] == "Z"
            for pid in workers
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for pid in workers:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class Endpoint:
    def __init__(self, tmp_path, backend, *, token=None):
        backends = backend if isinstance(backend, dict) else {"hpsv3": backend}
        self.engine = Collections(
            backends,
            concurrency={key: 1 for key in backends},
            batch_sizes={key: 2 for key in backends},
            directory=tmp_path / "inputs",
            max_collections=2,
            max_samples=32,
            max_image_bytes=1000000,
        )
        self.server = make_server(
            ("127.0.0.1", 0),
            collection_routes(self.engine, {"semantics": {"model": "fixture"}}),
            max_body_bytes=1000000,
            token=token,
        )
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
        self.calls.extend(i.prompt for i in inputs)
        return [{"mu": 0.75 if i.prompt == "text" else 0.25} for i in inputs]


def picture():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (20, 100, 200)).save(stream, format="PNG")
    return image_payload(stream.getvalue())


def test_http_streaming_retry_identity_release_and_auth(tmp_path, sampling_config):
    backend = Backend()
    endpoint = Endpoint(tmp_path, backend, token="fixture-token")
    client = ScoreClient(
        sampling_config.scoring, {"endpoint": endpoint.url, "token": "fixture-token"}
    )
    try:
        assert "hpsv3.mu" in client.info()["channels"]
        with pytest.raises(HTTPError, match="401"):
            ScoreClient(sampling_config.scoring, {"endpoint": endpoint.url}).info()
        groups = [{"id": "p", "prompt": "text", "draws": 2}]
        client.open("run", ["hpsv3.mu"], 7, groups)
        client.open("run", ["hpsv3.mu"], 7, groups)
        with Submissions(client, "run") as sender:
            sender.submit("a", "p", 0, picture())
        deadline = time.monotonic() + 5
        while not (first := client.read("run"))["results"]:
            assert time.monotonic() < deadline
        assert first["progress"]["accepted"] == 1
        assert first["results"][0]["values"] == {"hpsv3.mu": 0.75}
        client.submit("run", "a", "p", 0, picture())
        client.submit("run", "b", "p", 1, picture())
        while (last := client.read("run", first["next"]))["progress"]["completed"] < 2:
            assert time.monotonic() < deadline
        assert [r["sample_id"] for r in last["results"]] == ["b"]
        assert backend.calls == ["text", "text"]
        client.release("run")
        client.release("run")
        assert not list((tmp_path / "inputs").iterdir())
    finally:
        endpoint.close()


def test_mixed_pairwise_and_continuous_reward_channels(training_config):
    config = replace(
        training_config,
        rewards=Rewards(
            (
                RewardChannel("rubric.content_alignment", 0.5, "none"),
                RewardChannel("hpsv3.mu", 0.25, "round"),
                RewardChannel("vqascore.probability", 0.25, "round"),
            ),
            epsilon=0.5,
        ),
    )
    rows = [
        {"sample_id": key, "values": values}
        for key, values in (
            (
                "b",
                {
                    "rubric.content_alignment": 1.0,
                    "hpsv3.mu": 4.0,
                    "vqascore.probability": 0.9,
                },
            ),
            (
                "a",
                {
                    "rubric.content_alignment": 0.0,
                    "hpsv3.mu": 2.0,
                    "vqascore.probability": 0.9,
                },
            ),
        )
    ]
    result = combine_round(["b", "a"], rows, config)
    assert result["sample_ids"] == ["a", "b"]
    assert result["combined"] == pytest.approx([-1 / 6, 2 / 3])
    assert result["statistics"]["vqascore.probability"]["transformed"] == [0.0, 0.0]
    for bad in (
        rows[:1],
        rows + rows[:1],
        [rows[0], {"sample_id": "unknown", "values": rows[1]["values"]}],
    ):
        with pytest.raises(ValueError):
            combine_round(["a", "b"], bad, config)


def test_round_coordinator_waits_for_all_scores(tmp_path, training_config):
    endpoint = Endpoint(tmp_path, Backend())
    client = ScoreClient(training_config.scoring, {"endpoint": endpoint.url})
    collection = "attempt-round-000000"
    try:
        client.open(
            collection,
            ["hpsv3.mu"],
            2,
            [{"id": str(i), "prompt": "text", "draws": 2} for i in (0, 1)],
        )
        atomic_json(
            tmp_path / "attempts" / f"{collection}.json", {"collection": collection}
        )
        atomic_json(tmp_path / "active_collection.json", {"id": collection})
        coordinator = RoundScores(training_config, client, tmp_path, "attempt", 0)
        client.submit(collection, "0:0:0", "0", 0, picture())
        assert not coordinator.poll()
        assert not reward_path(tmp_path, "attempt", 0).exists()
        for slot, draw in ((0, 1), (1, 0), (1, 1)):
            client.submit(collection, f"0:{slot}:{draw}", str(slot), draw, picture())
        deadline = time.monotonic() + 5
        while not coordinator.poll():
            assert time.monotonic() < deadline
        assert coordinator.step == 1
        assert reward_path(tmp_path, "attempt", 0).exists()
        assert not (tmp_path / "active_collection.json").exists()
    finally:
        endpoint.close()


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
