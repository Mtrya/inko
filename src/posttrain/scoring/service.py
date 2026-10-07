"""Run one resident scoring replica independently of any trainer."""

import argparse
import concurrent.futures
import json
import os
import queue
import socket
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..records import atomic_json
from ..schema import decode, require
from .config import load_service_config
from .contracts import PROTOCOL, ScoreError, ScoreRequest, ScoreResult, validate_results


@dataclass(frozen=True)
class BackendInput:
    prompt: str
    image_path: str
    structured: dict[str, object]
    seed: int | None


class ScoringEngine:
    """One bounded inference queue, immutable inputs, no adapter-level retries."""

    def __init__(
        self, backend, capabilities, limits, cache_dir, telemetry=dict, on_stall=None
    ):
        self.backend, self.capabilities, self.limits = backend, capabilities, limits
        self.cache_dir, self.telemetry = Path(cache_dir), telemetry
        self.queue = queue.Queue(limits.queue_capacity)
        self.poisoned = False
        self.on_stall = on_stall
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def submit(self, requests, deadline):
        require(not self.poisoned, "service inference stalled; restart required")
        require(
            0 < len(requests) <= self.capabilities.max_batch_size, "invalid batch size"
        )
        validate_results(requests, [], self.capabilities.outputs, complete=False)
        for r in requests:
            require(
                r.fingerprint == self.capabilities.fingerprint
                and r.input_schema == self.capabilities.input_schema
                and set(r.structured) == set(self.capabilities.structured_fields),
                "request contract mismatch",
            )
            require(
                (r.seed is None) == self.capabilities.deterministic,
                "service seed policy mismatch",
            )
        future = concurrent.futures.Future()
        self.queue.put_nowait(
            (
                requests,
                min(deadline, time.time() + self.limits.request_timeout),
                time.perf_counter(),
                future,
            )
        )
        return future

    def _work(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            requests, deadline, enqueued, future = item
            if time.time() >= deadline:
                future.set_result(
                    [
                        ScoreResult.for_request(
                            r,
                            error=ScoreError(
                                "expired", "request expired in service queue", True
                            ),
                        )
                        for r in requests
                    ]
                )
                continue
            timer = threading.Timer(max(0, deadline - time.time()), self._expire)
            timer.daemon = True
            timer.start()
            try:
                results = self._score(requests, deadline, enqueued)
                future.set_result(results)
            except Exception as exc:  # noqa: BLE001 - explicit service failure envelope
                future.set_exception(exc)
            finally:
                timer.cancel()

    def _expire(self):
        self.poisoned = True
        if self.on_stall:
            self.on_stall()

    def _score(self, requests, deadline, enqueued):
        started = time.perf_counter()
        results, unique = {}, {}
        with tempfile.TemporaryDirectory(prefix="inko-score-input-") as temporary:
            inputs = []
            for request in requests:
                try:
                    data = request.image.read()
                    if time.time() >= deadline:
                        raise TimeoutError("request lifetime exceeded before inference")
                    path = self.cache_dir / f"{request.cache_key}.json"
                    if path.exists():
                        hit = decode(ScoreResult, json.loads(path.read_text()))
                        result = ScoreResult.for_request(
                            request, values=hit.values, metadata=hit.metadata
                        )
                        require(
                            hit.cache_key == request.cache_key
                            and hit.fingerprint == request.fingerprint,
                            "service cache identity mismatch",
                        )
                        validate_results(
                            [request],
                            [result],
                            self.capabilities.outputs,
                            complete=True,
                        )
                        results[request.request_id] = result
                    elif request.cache_key not in unique:
                        unique[request.cache_key] = request
                        image_path = Path(temporary) / f"{request.cache_key}.png"
                        image_path.write_bytes(data)
                        inputs.append(
                            BackendInput(
                                request.prompt,
                                str(image_path),
                                request.structured,
                                request.seed,
                            )
                        )
                except (ValueError, OSError, TimeoutError) as exc:
                    results[request.request_id] = ScoreResult.for_request(
                        request, error=ScoreError("input", str(exc), False)
                    )
            values, error = [], None
            if inputs:
                try:
                    values = self.backend.score(inputs)
                    require(len(values) == len(inputs), "backend result count mismatch")
                except Exception as exc:  # noqa: BLE001 - no silent backend retry
                    error = ScoreError("backend", f"{type(exc).__name__}: {exc}", True)
            cache_results = {}
            for index, request in enumerate(unique.values()):
                try:
                    result = ScoreResult.for_request(
                        request, values=None if error else values[index], error=error
                    )
                    validate_results(
                        [request], [result], self.capabilities.outputs, complete=False
                    )
                except ValueError as exc:
                    result = ScoreResult.for_request(
                        request, error=ScoreError("invalid_value", str(exc), True)
                    )
                cache_results[request.cache_key] = result
                if result.status == "ok":
                    atomic_json(
                        self.cache_dir / f"{request.cache_key}.json", asdict(result)
                    )
            metadata = {
                "queue_seconds": str(started - enqueued),
                "service_seconds": str(time.perf_counter() - started),
                **{k: str(v) for k, v in self.telemetry().items()},
            }
            for request in requests:
                if request.request_id not in results:
                    result = cache_results[request.cache_key]
                    results[request.request_id] = ScoreResult.for_request(
                        request,
                        values=result.values,
                        error=result.error,
                        metadata=metadata,
                    )
        return list(results.values())

    def close(self):
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            self.poisoned = True
        self.thread.join(timeout=self.limits.request_timeout)


def make_server(bind, engine):
    class Handler(BaseHTTPRequestHandler):
        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/health":
                self.reply(
                    200,
                    {
                        "protocol": PROTOCOL,
                        "ready": engine.thread.is_alive() and not engine.poisoned,
                        "fingerprint": engine.capabilities.fingerprint,
                    },
                )
            elif self.path == "/v1/capabilities":
                self.reply(200, asdict(engine.capabilities))
            else:
                self.reply(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/score":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                self.connection.settimeout(engine.limits.request_timeout)
                length = int(self.headers.get("Content-Length", 0))
                require(0 < length <= engine.limits.max_body_bytes, "invalid body size")
                payload = json.loads(self.rfile.read(length))
                require(
                    set(payload) == {"protocol", "deadline", "requests"}
                    and payload["protocol"] == PROTOCOL,
                    "invalid protocol envelope",
                )
                deadline = decode(float, payload["deadline"])
                require(deadline > time.time(), "expired request")
                requests = [decode(ScoreRequest, r) for r in payload["requests"]]
                future = engine.submit(requests, deadline)
                results = future.result(
                    timeout=min(engine.limits.request_timeout, deadline - time.time())
                )
                self.reply(
                    200, {"protocol": PROTOCOL, "results": [asdict(r) for r in results]}
                )
            except queue.Full:
                self.reply(503, {"error": "queue full"})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except TimeoutError:
                self.reply(504, {"error": "request lifetime exceeded"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(engine.limits.queue_capacity + 1)
            super().__init__(*args)

        def process_request(self, request, address):
            request.settimeout(engine.limits.request_timeout)
            if not self.slots.acquire(blocking=False):
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
                )
                self.shutdown_request(request)
                return
            super().process_request(request, address)

        def process_request_thread(self, request, address):
            def expire():
                try:
                    request.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            timer = threading.Timer(engine.limits.request_timeout, expire)
            timer.daemon = True
            timer.start()
            try:
                super().process_request_thread(request, address)
            finally:
                timer.cancel()
                self.slots.release()

    return Server(bind, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--bind", default="127.0.0.1:8100")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_service_config(args.config)
    if args.validate_only:
        print(json.dumps(asdict(config.capabilities), indent=2))
        return
    if args.assets is None or args.cache is None:
        parser.error("--assets and --cache required")
    host, port = args.bind.rsplit(":", 1)
    require(
        host in {"127.0.0.1", "localhost", "::1"},
        "initial owned services must bind loopback",
    )
    from .backends.runtime import load_backend

    started = time.perf_counter()
    backend, telemetry, provenance = load_backend(
        config, json.loads(args.assets.read_text()), args.device
    )
    engine = ScoringEngine(
        backend,
        config.capabilities,
        config.limits,
        args.cache,
        telemetry,
        on_stall=lambda: os._exit(70),
    )
    server = make_server((host, int(port)), engine)
    print(
        json.dumps(
            {
                "event": "ready",
                "fingerprint": config.fingerprint,
                "load_seconds": time.perf_counter() - started,
                **telemetry(),
                "provenance": provenance,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
