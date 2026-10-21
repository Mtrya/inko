"""Private HPS/VQA worker; each process owns one isolated eager runtime."""

import argparse
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def load_backend(kind, assets, index, cpu_threads):
    import torch

    from src.utils.device import peak_memory, set_device, synchronize

    from ...artifacts import package_versions

    started = time.monotonic()
    device = set_device(assets["device_type"], index, cpu_threads)
    os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)
    if kind == "hpsv3":
        from .hpsv3 import HPSv3

        model = HPSv3(assets, device)
    else:
        from .vqascore import VQAScore

        model = VQAScore(assets, device)
    synchronize(device)
    info = {
        "device": str(device),
        "load_seconds": time.monotonic() - started,
        "packages": package_versions(),
    }
    lock = threading.Lock()

    def dispatch(route, body):
        if route == "/info":
            return {**info, "peak_memory_bytes": peak_memory(device)}
        if route != "/score":
            raise ValueError("unknown worker operation")
        with lock, torch.inference_mode():
            if device.type != "cpu":
                getattr(torch, device.type).set_device(device)
            values = model.score([SimpleNamespace(**row) for row in body["inputs"]])
            synchronize(device)
            return {"values": values, "peak_memory_bytes": peak_memory(device)}

    return dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hpsv3", "vqascore"), required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    from ..service import make_server

    dispatch = load_backend(
        args.backend, json.loads(args.assets.read_text()), args.device, args.cpu_threads
    )
    server = make_server(
        ("127.0.0.1", args.port),
        dispatch,
        max_body_bytes=1024 * 1024,
        timeout=args.timeout,
    )
    print(json.dumps({"event": "ready", **dispatch("/info", None)}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
