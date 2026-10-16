"""Admission, streaming and pair identity independent of model quality."""

import io
import threading
import time

import pytest
from PIL import Image

from src.posttrain.scoring.collections import Collections
from src.posttrain.scoring.contracts import image_payload


def picture(color, format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (5, 7), color).save(stream, format=format)
    return image_payload(stream.getvalue())


def group(key, draws=2):
    return {
        "id": key,
        "prompt": "A blue tree",
        "draws": draws,
        "rubric": {
            key: [{"text": "Visible criterion", "weight": "high"}]
            for key in (
                "content_alignment",
                "form_structure",
                "aesthetics",
                "style_medium",
                "artifacts",
            )
        },
        "reference": picture("green", "JPEG"),
    }


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("scoring work did not complete")


class Auxiliary:
    def __init__(self, metric, value):
        self.metric, self.value = metric, value
        self.started = threading.Event()

    def score(self, requests):
        self.started.set()
        return [{self.metric: self.value} for _ in requests]


class Judge:
    def __init__(self):
        self.images = []

    def judge(self, prompt, rubric, images):
        self.images.append(images)
        return {
            "decisions": {
                category: {"verdict": "tie", "basis": "No visible preference."}
                for category in rubric
            }
        }


def make_engine(tmp_path, backends):
    return Collections(
        backends,
        concurrency={key: 1 for key in backends},
        batch_sizes={key: 2 for key in backends},
        directory=tmp_path,
        max_collections=1,
        max_samples=4,
        max_image_bytes=10000,
    )


def test_first_arrivals_score_and_all_late_pair_partners_are_admitted(tmp_path):
    hps, vqa, judge = Auxiliary("mu", 3.25), Auxiliary("probability", 0.8), Judge()
    engine = make_engine(tmp_path, {"hpsv3": hps, "vqascore": vqa, "rubric": judge})
    groups = [group("one"), group("two")]
    channels = ["hpsv3.mu", "vqascore.probability", "rubric.style_medium"]
    try:
        engine.open("round", channels, 19, groups)
        engine.open("round", channels, 19, groups)
        engine.submit("round", "a", "one", 0, picture("red"))
        assert hps.started.wait(timeout=2) and vqa.started.wait(timeout=2)
        engine.submit("round", "b", "two", 0, picture("blue"))
        assert not judge.images
        assert engine.read("round")["progress"]["completed"] == 0
        # Every admitted image is waiting for its pair partner. Both later
        # partners must still enter at the collection's capacity limit.
        engine.submit("round", "d", "two", 1, picture("white"))
        engine.submit("round", "c", "one", 1, picture("black"))
        packet = wait_until(
            lambda: (
                engine.read("round")
                if engine.read("round")["progress"]["completed"] == 4
                else None
            )
        )
        assert not packet["errors"]
        assert packet["progress"]["pairs_completed"] == 2
        assert {r["sample_id"] for r in packet["results"]} == {"a", "b", "c", "d"}
        assert all(
            r["values"]
            == {
                "hpsv3.mu": 3.25,
                "vqascore.probability": 0.8,
                "rubric.style_medium": 0.5,
            }
            for r in packet["results"]
        )
        assert len(judge.images) == 4
        for offset in (0, 2):
            forward, reverse = judge.images[offset : offset + 2]
            assert forward[:2] == reverse[:2][::-1]
            assert forward[2] == reverse[2]
        engine.submit("round", "a", "one", 0, picture("red"))
        assert engine.read("round", after=4)["results"] == []
        assert len(judge.images) == 4
        engine.release("round")
        wait_until(lambda: not list(tmp_path.iterdir()))
    finally:
        engine.close()


def test_auxiliary_only_odd_group_and_failure_are_explicit(tmp_path):
    class Failed:
        def score(self, requests):
            raise RuntimeError("model failed")

    engine = make_engine(tmp_path, {"hpsv3": Failed()})
    try:
        engine.open(
            "failed", ["hpsv3.mu"], 7, [{"id": "p", "prompt": "A tree", "draws": 1}]
        )
        engine.submit("failed", "x", "p", 0, picture("red"))
        errors = wait_until(lambda: engine.read("failed")["errors"])
        assert errors[0]["backend"] == "hpsv3"
        assert "model failed" in errors[0]["message"]
        assert engine.read("failed")["results"] == []
    finally:
        engine.close()


def test_submission_retry_cannot_change_sample_identity(tmp_path):
    engine = make_engine(tmp_path, {"hpsv3": Auxiliary("mu", 1.0)})
    try:
        engine.open(
            "round", ["hpsv3.mu"], 7, [{"id": "p", "prompt": "A tree", "draws": 2}]
        )
        engine.submit("round", "x", "p", 0, picture("red"))
        with pytest.raises(ValueError, match="identity changed"):
            engine.submit("round", "x", "p", 1, picture("red"))
        with pytest.raises(ValueError, match="draw already"):
            engine.submit("round", "y", "p", 0, picture("red"))
        with pytest.raises(ValueError, match="capacity"):
            engine.open(
                "second", ["hpsv3.mu"], 7, [{"id": "p", "prompt": "A tree", "draws": 1}]
            )
    finally:
        engine.close()


def test_release_holds_capacity_until_inflight_inputs_are_unused(tmp_path):
    class Held(Auxiliary):
        def __init__(self):
            super().__init__("mu", 1.0)
            self.proceed = threading.Event()

        def score(self, requests):
            self.started.set()
            assert self.proceed.wait(timeout=3)
            return super().score(requests)

    backend = Held()
    engine = make_engine(tmp_path, {"hpsv3": backend})
    groups = [{"id": "p", "prompt": "A tree", "draws": 4}]
    try:
        engine.open("old", ["hpsv3.mu"], 7, groups)
        engine.submit("old", "a", "p", 0, picture("red"))
        assert backend.started.wait(timeout=2)
        for draw in range(1, 4):
            engine.submit("old", str(draw), "p", draw, picture("red"))
        engine.release("old")
        with pytest.raises(ValueError, match="capacity"):
            engine.open("new", ["hpsv3.mu"], 7, groups)
        backend.proceed.set()
        wait_until(lambda: not list(tmp_path.iterdir()))
        engine.open("new", ["hpsv3.mu"], 7, groups)
        assert engine.read("new")["progress"]["accepted"] == 0
    finally:
        backend.proceed.set()
        engine.close()
