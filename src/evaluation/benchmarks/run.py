"""Prepare released catalogs and score saved images in isolated interpreters."""

import argparse
import fcntl
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.evaluation.records import (
    append_jsonl,
    load_catalog,
    path_component,
    prompt_key,
    read_jsonl,
)

from .common import join_images, write_json

BENCHMARKS = ("geneval", "dpg", "oneig", "tiif", "qwen_image")


def adapter(name):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {name}")
    return importlib.import_module(f"src.evaluation.benchmarks.{name}")


def dimensions(name, row):
    if name == "oneig":
        return adapter(name).dimensions(row)
    if name == "tiif":
        return ["vlm", "ocr"] if row["metadata"]["type"] == "text" else ["vlm"]
    return [{"geneval": "detector", "dpg": "vqa", "qwen_image": "judge"}[name]]


def result_key(row):
    return (*prompt_key(row["prompt"]), row["dimension"])


def prepare(name, settings, assets, output):
    rows = adapter(name).prepare(assets, settings)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("")
    append_jsonl(output, rows)
    load_catalog(output)
    return {
        "prompts": len(rows),
        "images": sum(len(r["seeds"]) for r in rows),
        "catalog": str(output),
    }


def score(name, settings, assets, model, images, output):
    images, output = Path(images).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalogs, samples = [], []
        for path in sorted(
            (images / model / path_component(name)).glob("*/prompts.jsonl")
        ):
            catalogs.extend(load_catalog(path))
            samples.extend(read_jsonl(path.with_name("samples.jsonl")))
        if not catalogs:
            raise ValueError(f"no {name} catalog for {model} in {images}")
        groups = join_images(catalogs, samples, model, images)
        request = {
            "benchmark": name,
            "settings": settings,
            "assets": assets,
            "groups": groups,
            "model": model,
        }
        path = output / "request.json"
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError(
                "scoring inputs or judge settings changed; use a new score directory"
            )
        write_json(path, request)
        native = output / "native"
        native.mkdir(exist_ok=True)
        adapter(name).layout(groups, native)
        records = read_jsonl(output / "scores.jsonl", repair_tail=True)
        done = {result_key(row) for row in records}
        expected = {
            (*prompt_key(g["prompt"]), dim)
            for g in groups
            for dim in dimensions(name, g["prompt"])
        }
        if len(done) != len(records) or done - expected:
            raise ValueError("duplicate or unexpected benchmark result")
        env = {**os.environ, **assets.get("env", {}), "PYTHONUTF8": "1"}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        for dimension in sorted({key[-1] for key in expected - done}):
            python = (
                assets.get("ocr_python", assets["python"])
                if dimension == "ocr"
                else assets["python"]
            )
            subprocess.run(
                [
                    python,
                    "-m",
                    "src.evaluation.benchmarks.run",
                    str(path),
                    "--dimension",
                    dimension,
                ],
                env=env,
                check=True,
            )
        records = read_jsonl(output / "scores.jsonl")
        if {result_key(row) for row in records} != expected:
            raise ValueError("benchmark evaluator omitted requested results")
        subprocess.run(
            [
                assets["python"],
                "-m",
                "src.evaluation.benchmarks.run",
                str(path),
                "--summarize",
            ],
            env=env,
            check=True,
        )
        return json.loads((output / "summary.json").read_text())


class Results:
    def __init__(self, output):
        self.output = output
        self.started = self.previous = time.perf_counter()
        self.load_seconds = None
        self.samples = 0

    def loaded(self):
        self.load_seconds = time.perf_counter() - self.started
        self.previous = time.perf_counter()

    def __call__(self, group, dimension, native):
        elapsed = time.perf_counter() - self.previous
        append_jsonl(
            self.output / "scores.jsonl",
            [{**group, "dimension": dimension, "native": native, "seconds": elapsed}],
        )
        self.samples += len(group["samples"])
        self.previous = time.perf_counter()

    def finish(self, dimension):
        telemetry = {
            "dimension": dimension,
            "load_seconds": self.load_seconds,
            "wall_seconds": time.perf_counter() - self.started,
            "images": self.samples,
        }
        if "torch" in sys.modules:
            import torch

            telemetry["peak_allocated_bytes"] = [
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ]
        append_jsonl(self.output / "execution.jsonl", [telemetry])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--dimension")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = args.request.parent.resolve()
    name = request["benchmark"]
    module = adapter(name)
    records = read_jsonl(output / "scores.jsonl", repair_tail=True)
    if args.summarize:
        summary = module.summarize(records, request["assets"])
        expected = sum(len(dimensions(name, g["prompt"])) for g in request["groups"])
        write_json(
            output / "summary.json",
            {
                "model": request["model"],
                "benchmark": name,
                "expected": expected,
                "evaluated": len(records),
                "scores": summary,
            },
        )
        return
    completed = {result_key(row) for row in records}
    pending = [
        g
        for g in request["groups"]
        if args.dimension in dimensions(name, g["prompt"])
        and (*prompt_key(g["prompt"]), args.dimension) not in completed
    ]
    emit = Results(output)
    module.evaluate(
        pending,
        request["assets"],
        {**request["settings"], "dimension": args.dimension},
        output / "native",
        emit,
    )
    emit.finish(args.dimension)


if __name__ == "__main__":
    main()
