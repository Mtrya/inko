"""Streaming collections, seeded pairs and independent model work queues."""

import math
import queue
import random
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from ..schema import require
from .backends.rubric import merge_orders
from .contracts import CHANNELS, data_url, image_bytes, rubric_criteria


class Collections:
    """Spool bounded encoded inputs; only active model batches occupy image RAM.

    Reserving the whole collection's sample slots lets a late pair partner enter
    even when every early arrival is waiting for its partner.
    """

    def __init__(
        self,
        backends,
        *,
        concurrency,
        batch_sizes,
        directory,
        max_collections,
        max_samples,
        max_image_bytes,
    ):
        self.backends, self.batch_sizes = backends, batch_sizes
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_collections, self.max_samples = max_collections, max_samples
        self.max_image_bytes = max_image_bytes
        self.lock = threading.RLock()
        self.collections = {}
        self.closed = threading.Event()
        self.queues, self.threads = {}, []
        for backend in backends:
            self.queues[backend] = queue.Queue(
                maxsize=max_collections * max_samples * 2
            )
            for _ in range(concurrency[backend]):
                thread = threading.Thread(
                    target=self._work, args=(backend,), daemon=True
                )
                thread.start()
                self.threads.append(thread)

    def open(self, collection_id, channels, seed, groups):
        with self.lock:
            require(
                bool(collection_id) and bool(groups),
                "collection ID and groups are required",
            )
            require(
                bool(channels)
                and len(channels) == len(set(channels))
                and set(channels) <= set(CHANNELS),
                "unknown or duplicate score channel",
            )
            models = {channel.split(".")[0] for channel in channels}
            require(models <= self.backends.keys(), "requested model is unavailable")
            existing = self.collections.get(collection_id)
            request = {"channels": channels, "seed": seed, "groups": groups}
            if existing is not None:
                require(not existing.released, "collection is being released")
                require(existing.request == request, "collection inputs changed")
                return
            require(
                len(self.collections) < self.max_collections,
                "scoring collection capacity reached",
            )
            require(len({g["id"] for g in groups}) == len(groups), "duplicate group ID")
            require(
                all(
                    type(g["draws"]) is int and g["draws"] > 0 and bool(g["prompt"])
                    for g in groups
                ),
                "group requires prompt and positive draw count",
            )
            expected = sum(g["draws"] for g in groups)
            require(expected <= self.max_samples, "collection exceeds sample capacity")
            if "rubric" in models:
                for group in groups:
                    require(
                        group["draws"] % 2 == 0,
                        "rubric groups require an even draw count",
                    )
                    rubric_criteria(group["rubric"])
            temporary = tempfile.TemporaryDirectory(
                dir=self.directory, prefix="collection-"
            )
            collection = SimpleNamespace(
                id=collection_id,
                request=request,
                channels=set(channels),
                models=models,
                groups={},
                expected=expected,
                temporary=temporary,
                samples={},
                positions={},
                partners={},
                orders={},
                pairs=set(),
                results=[],
                errors=[],
                pending=0,
                released=False,
                started=time.monotonic(),
                first_sample=None,
                first_score=None,
                first_pair=None,
                metrics={
                    model: {"calls": 0, "seconds": 0.0, "failures": 0}
                    for model in models
                },
            )
            try:
                rng = random.Random(seed)
                for index, group in enumerate(groups):
                    entry = dict(group)
                    reference = group.get("reference")
                    if reference is not None:
                        data = image_bytes(reference)
                        require(
                            len(data) <= self.max_image_bytes,
                            "reference image too large",
                        )
                        path = Path(temporary.name) / f"reference-{index}"
                        path.write_bytes(data)
                        entry["reference"] = (str(path), reference["mime_type"])
                    collection.groups[group["id"]] = entry
                    if "rubric" in models:
                        indices = list(range(group["draws"]))
                        rng.shuffle(indices)
                        for a, b in zip(indices[::2], indices[1::2], strict=True):
                            pair = (group["id"], min(a, b), max(a, b))
                            collection.partners[(group["id"], a)] = pair
                            collection.partners[(group["id"], b)] = pair
                self.collections[collection_id] = collection
            except BaseException:
                temporary.cleanup()
                raise

    def submit(self, collection_id, sample_id, group_id, draw_index, image):
        data = image_bytes(image)
        require(len(data) <= self.max_image_bytes, "sample image too large")
        with self.lock:
            c = self.collections[collection_id]
            require(not c.released, "collection is being released")
            require(not c.errors, "collection scoring failed")
            require(
                group_id in c.groups
                and type(draw_index) is int
                and 0 <= draw_index < c.groups[group_id]["draws"],
                "unknown group or draw",
            )
            position = (group_id, draw_index)
            if sample_id in c.samples:
                require(
                    c.samples[sample_id].position == position, "sample identity changed"
                )
                return
            require(position not in c.positions, "draw already has a sample")
            path = Path(c.temporary.name) / f"sample-{len(c.samples)}"
            path.write_bytes(data)
            sample = SimpleNamespace(
                id=sample_id,
                position=position,
                image_path=str(path),
                mime=image["mime_type"],
                values={},
                finished=set(),
                judgment=None,
            )
            c.samples[sample_id], c.positions[position] = sample, sample_id
            if c.first_sample is None:
                c.first_sample = time.monotonic() - c.started
            for model in c.models - {"rubric"}:
                self._enqueue(model, c, sample_id)
            if "rubric" in c.models:
                pair = c.partners[position]
                if pair not in c.pairs and all(
                    (group_id, draw) in c.positions for draw in pair[1:]
                ):
                    c.pairs.add(pair)
                    self._enqueue("rubric", c, (pair, 0))
                    self._enqueue("rubric", c, (pair, 1))

    def _enqueue(self, model, collection, work):
        collection.pending += 1
        self.queues[model].put_nowait((collection, work))

    def read(self, collection_id, after=0):
        with self.lock:
            c = self.collections[collection_id]
            require(not c.released, "collection is being released")
            require(
                type(after) is int and 0 <= after <= len(c.results),
                "invalid result offset",
            )
            return {
                "results": c.results[after:],
                "next": len(c.results),
                "errors": list(c.errors),
                "progress": {
                    "expected": c.expected,
                    "accepted": len(c.samples),
                    "completed": len(c.results),
                    "pairs_completed": sum(
                        len(orders) == 2 for orders in c.orders.values()
                    ),
                    "pending_work": c.pending,
                    "first_sample_seconds": c.first_sample,
                    "first_score_seconds": c.first_score,
                    "first_pair_seconds": c.first_pair,
                    "elapsed_seconds": time.monotonic() - c.started,
                    "backends": {key: dict(value) for key, value in c.metrics.items()},
                },
            }

    def release(self, collection_id):
        with self.lock:
            c = self.collections.get(collection_id)
            if c is not None:
                c.released = True
                if c.pending == 0:
                    c.temporary.cleanup()
                    del self.collections[c.id]

    def _work(self, model):
        jobs = self.queues[model]
        while not self.closed.is_set() or not jobs.empty():
            try:
                batch = [jobs.get(timeout=0.2)]
            except queue.Empty:
                continue
            for _ in range(self.batch_sizes[model] - 1 if model != "rubric" else 0):
                try:
                    batch.append(jobs.get_nowait())
                except queue.Empty:
                    break
            with self.lock:
                live = [
                    (c, work) for c, work in batch if not c.released and not c.errors
                ]
            started = time.monotonic()
            try:
                if model == "rubric" and live:
                    c, (pair, order) = live[0]
                    group = c.groups[pair[0]]
                    ids = [c.positions[(pair[0], draw)] for draw in pair[1:]]
                    samples = [
                        c.samples[key] for key in (ids if order == 0 else ids[::-1])
                    ]
                    images = [data_url(s.image_path, s.mime) for s in samples]
                    if group.get("reference") is not None:
                        images.append(data_url(*group["reference"]))
                    result = self.backends[model].judge(
                        group["prompt"], group["rubric"], images
                    )
                    with self.lock:
                        if not c.released:
                            orders = c.orders.setdefault(pair, {})
                            orders[order] = result
                            if len(orders) == 2:
                                values, normalized = merge_orders(
                                    orders[0]["decisions"], orders[1]["decisions"], ids
                                )
                                judgment = {
                                    "sample_ids": ids,
                                    "orders": normalized,
                                    "usage": [
                                        orders[i].get("usage", {}) for i in (0, 1)
                                    ],
                                }
                                for key, value in zip(ids, values, strict=True):
                                    self._finish(c, key, model, value, judgment)
                                if c.first_pair is None:
                                    c.first_pair = time.monotonic() - c.started
                elif live:
                    requests = [
                        SimpleNamespace(
                            prompt=c.groups[c.samples[key].position[0]]["prompt"],
                            image_path=c.samples[key].image_path,
                        )
                        for c, key in live
                    ]
                    results = self.backends[model].score(requests)
                    require(len(results) == len(live), "backend omitted sample scores")
                    with self.lock:
                        for (c, key), value in zip(live, results, strict=True):
                            if not c.released:
                                self._finish(
                                    c,
                                    key,
                                    model,
                                    {f"{model}.{k}": v for k, v in value.items()},
                                )
            except Exception as exc:  # noqa: BLE001 - report backend failures
                with self.lock:
                    for c, work in live:
                        if not c.released:
                            c.errors.append(
                                {
                                    "backend": model,
                                    "work": work,
                                    "message": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            c.metrics[model]["failures"] += 1
            finally:
                with self.lock:
                    for c in {id(c): c for c, _ in live}.values():
                        c.metrics[model]["calls"] += 1
                        c.metrics[model]["seconds"] += time.monotonic() - started
                    for c, _ in batch:
                        c.pending -= 1
                        if c.released and c.pending == 0:
                            c.temporary.cleanup()
                            del self.collections[c.id]
                for _ in batch:
                    jobs.task_done()

    def _finish(self, c, key, model, values, judgment=None):
        expected = {channel for channel in c.channels if channel.split(".")[0] == model}
        require(
            expected <= values.keys()
            and all(
                type(v) in (float, int) and math.isfinite(v) for v in values.values()
            ),
            "backend returned missing or nonfinite scores",
        )
        s = c.samples[key]
        s.values.update({channel: values[channel] for channel in expected})
        s.finished.add(model)
        if judgment is not None:
            s.judgment = judgment
        if c.first_score is None:
            c.first_score = time.monotonic() - c.started
        if s.finished == c.models:
            c.results.append(
                {"sample_id": key, "values": dict(s.values), "judgment": s.judgment}
            )
            Path(s.image_path).unlink()

    def close(self):
        with self.lock:
            for key in list(self.collections):
                self.release(key)
            self.closed.set()
        for thread in self.threads:
            thread.join(timeout=1)
