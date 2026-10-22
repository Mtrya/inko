"""Qwen-Image-Bench's Q-Judger and hierarchical native aggregation."""

import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.records import append_jsonl, read_jsonl

from .common import load_source, prompt


def prepare(assets, settings):
    rows = read_jsonl(assets["data"])
    metadata = json.loads(
        (Path(assets["source"]) / "metadata/bench_metadata.json").read_text()
    )
    dims = {row["ID"]: row["dims_en"] for row in metadata}
    result = []
    for row in rows:
        if row["dims_en"] != dims[row["ID"]]:
            raise ValueError(f"Qwen metadata differs for ID {row['ID']}")
        result.append(
            prompt(
                "qwen_image",
                "released",
                row["ID"],
                row["prompt_en"],
                {"ID": row["ID"], "dims_en": row["dims_en"]},
                [settings["seed"] + row["ID"]],
                "en",
            )
        )
    return result


def layout(groups, root):
    path = root / "input.jsonl"
    path.write_text("")
    append_jsonl(
        path,
        [
            {
                "ID": g["prompt"]["metadata"]["ID"],
                "prompt": g["prompt"]["text"],
                "image_path": g["samples"][0]["image_path"],
            }
            for g in groups
        ],
    )


def upstream(assets):
    sys.path.insert(0, assets["source"])
    return load_source(Path(assets["source"]) / "judge.py")


def judge_batch(native, judge, batch, batch_size):
    import pandas as pd

    inputs = pd.DataFrame(
        [
            {
                "ID": group["prompt"]["metadata"]["ID"],
                "prompt": group["prompt"]["text"],
                "image_path": group["samples"][0]["image_path"],
            }
            for group in batch
        ]
    )
    meta = pd.DataFrame([group["prompt"]["metadata"] for group in batch])
    # The released harness does not retry parse failures; bound the retries.
    for _ in range(8):
        results, failures, _, image_failures = native._run_batch_inference(
            judge, SimpleNamespace(batch_size=batch_size), inputs, meta, "Q-Judger"
        )
        if not failures and not image_failures:
            return results
        if image_failures:
            # Broken images do not heal by resampling.
            break
    # Native aggregation omits missing dimensions. Keep partial scores after
    # bounded retries, but reject rows with no parsed dimensions.
    if not image_failures and all(result["judge_model_output"] for result in results):
        return results
    raise ValueError(
        f"Q-Judger failed to evaluate IDs {inputs['ID'].tolist()}: {results}"
    )


def judge_worker(
    rank, model, template, request_config, assets, batch_size, incoming, outgoing
):
    native = upstream(assets)
    from backends.ms_swift_backend import MsSwiftJudge
    from swift import TransformersEngine

    # Processor serialization omits metadata attached by the native model loader.
    template.processor.model_info = template.model_info
    template.processor.model_meta = template.model_meta
    judge = MsSwiftJudge.__new__(MsSwiftJudge)
    judge.engine = TransformersEngine(
        model, template=template, max_batch_size=batch_size
    )
    judge.request_config = request_config
    outgoing.put(("ready", rank, None))
    while (task := incoming.get()) is not None:
        index, batch = task
        outgoing.put(("result", index, judge_batch(native, judge, batch, batch_size)))


def evaluate(groups, assets, settings, root, emit):
    native = upstream(assets)
    from backends.ms_swift_backend import MsSwiftJudge

    batch_size, processes = settings["batch_size"], settings["processes"]
    if min(batch_size, processes) < 1:
        raise ValueError("Q-Judger batch_size and processes must be positive")
    judge = MsSwiftJudge(
        assets["model"], max_batch_size=batch_size, max_new_tokens=4096
    )
    batches = [groups[i : i + batch_size] for i in range(0, len(groups), batch_size)]
    if processes == 1:
        emit.loaded()
        for batch in batches:
            results = judge_batch(native, judge, batch, batch_size)
            for group, result in zip(batch, results, strict=True):
                emit(group, "judge", result)
        return

    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    incoming, outgoing = context.Queue(processes), context.Queue(processes)
    workers = mp.spawn(
        judge_worker,
        args=(
            judge.engine.model,
            judge.engine.template,
            judge.request_config,
            assets,
            batch_size,
            incoming,
            outgoing,
        ),
        nprocs=processes,
        join=False,
    )

    def receive():
        while True:
            try:
                return outgoing.get(timeout=1)
            except queue.Empty:
                if workers.join(timeout=0):
                    raise RuntimeError(
                        "Q-Judger workers exited before completing results"
                    )

    try:
        for _ in range(processes):
            receive()
        emit.loaded()
        remaining = iter(enumerate(batches))
        pending = {}

        def submit():
            task = next(remaining, None)
            if task is not None:
                index, batch = task
                pending[index] = batch
                incoming.put(task)

        for _ in range(processes):
            submit()
        while pending:
            _, index, results = receive()
            for group, result in zip(pending.pop(index), results, strict=True):
                emit(group, "judge", result)
            submit()
        for _ in range(processes):
            incoming.put(None)
        while not workers.join(timeout=1):
            pass
    finally:
        # The CUDA IPC owner must outlive every worker using its weights.
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for channel in (incoming, outgoing):
            channel.cancel_join_thread()
            channel.close()


def summarize(rows, assets):
    return upstream(assets).compute_bench_scores(
        [json.loads(row["native"]["judge_model_output"]) for row in rows]
    )
