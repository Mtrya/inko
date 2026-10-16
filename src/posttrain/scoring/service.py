"""Launch the resident three-model scoring service on its own allocation."""

import argparse
import fcntl
import json
import shutil
import signal
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..schema import require
from .collections import Collections
from .config import load_server_config
from .contracts import CHANNELS


def make_server(bind, dispatch, *, max_body_bytes, timeout=60, token=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def reply(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            self.respond(True)

        def respond(self, body):
            self.connection.settimeout(timeout)
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self.reply(401, {"error": "invalid scoring token"})
                return
            try:
                if body:
                    size = int(self.headers.get("Content-Length", 0))
                    require(0 < size <= max_body_bytes, "invalid request body size")
                    body = json.loads(self.rfile.read(size))
                self.reply(200, dispatch(self.path, body))
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - model failure crosses HTTP
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    return ThreadingHTTPServer(bind, Handler)


def collection_routes(engine, info, check=lambda: None):
    def dispatch(route, body):
        check()
        if route == "/info":
            return {"channels": CHANNELS, **info}
        operations = {
            "/open": engine.open,
            "/submit": engine.submit,
            "/read": engine.read,
            "/release": engine.release,
        }
        require(route in operations, "unknown scoring operation")
        result = operations[route](**body)
        return result if result is not None else {"accepted": True}

    return dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--bind", default="0.0.0.0:8100")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_server_config(args.config)
    if args.validate_only:
        print(json.dumps(asdict(config), indent=2))
        return
    if args.assets is None or args.output is None:
        parser.error("--assets and --output are required")
    from .lifecycle import Runtime

    assets = json.loads(args.assets.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    host, port = args.bind.rsplit(":", 1)

    def stop_loading(*_):
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_loading)
    with (args.output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = args.output / "inputs"
        if inputs.exists():
            shutil.rmtree(inputs)
        with Runtime(config, assets, args.output) as runtime:
            engine = Collections(
                runtime.backends,
                concurrency=runtime.concurrency,
                batch_sizes={
                    "rubric": 1,
                    "hpsv3": config.hpsv3.batch_size,
                    "vqascore": config.vqascore.batch_size,
                },
                directory=inputs,
                max_collections=config.server.max_collections,
                max_samples=config.server.max_samples,
                max_image_bytes=config.server.max_image_bytes,
            )
            server = make_server(
                (host, int(port)),
                collection_routes(engine, runtime.info, runtime.check),
                max_body_bytes=config.server.max_body_bytes,
                token=assets.get("token"),
            )
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(
                    sig, lambda *_: threading.Thread(target=server.shutdown).start()
                )
            print(
                json.dumps({"event": "ready", "bind": args.bind, **runtime.info}),
                flush=True,
            )
            try:
                server.serve_forever()
            finally:
                server.server_close()
                runtime.close()
                engine.close()


if __name__ == "__main__":
    main()
