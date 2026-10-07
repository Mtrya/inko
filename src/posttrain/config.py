"""Strict configuration for the sampling/scoring increment."""

import math
import re
import tomllib
from pathlib import Path


def fields(table: dict, names: str, label: str) -> None:
    expected = set(names.split())
    if set(table) != expected:
        raise ValueError(
            f"{label}: missing={sorted(expected - table.keys())}, unknown={sorted(table.keys() - expected)}"
        )


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        c = tomllib.load(stream)
    fields(c, "initialization sampling execution evaluation arms scorers", "config")
    fields(c["initialization"], "step weights sha256 text_exit_layer", "initialization")
    if c["initialization"]["weights"] not in {"ema", "live"}:
        raise ValueError("weights must be ema or live")
    if not re.fullmatch(r"[0-9a-f]{64}", c["initialization"]["sha256"]):
        raise ValueError("initialization requires a SHA256 digest")
    for key in ("step", "text_exit_layer"):
        if type(c["initialization"][key]) is not int or c["initialization"][key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    fields(c["sampling"], "precision batch_size retain_selected shapes", "sampling")
    if c["sampling"]["precision"] not in {"bfloat16", "float32"}:
        raise ValueError("unsupported sampling precision")
    if (
        type(c["sampling"]["retain_selected"]) is not bool
        or not c["sampling"]["shapes"]
    ):
        raise ValueError("retain_selected must be boolean and shapes must be nonempty")
    for aspect, shape in c["sampling"]["shapes"].items():
        if len(shape) != 2 or any(
            type(x) is not int or x <= 0 or x % 16 for x in shape
        ):
            raise ValueError(
                f"{aspect}: image height/width must be positive multiples of 16"
            )
    fields(
        c["execution"],
        "learner_devices hpsv3_devices vqascore_devices max_inflight_batches score_attempts startup_timeout task_timeout cpu_threads",
        "execution",
    )
    devices = []
    for role in ("learner", "hpsv3", "vqascore"):
        allocation = c["execution"][role + "_devices"]
        if not allocation or any(type(x) is not int or x < 0 for x in allocation):
            raise ValueError(f"{role}: nonempty device allocation required")
        devices.extend(allocation)
    if len(devices) != len(set(devices)):
        raise ValueError("role device allocations must be disjoint")
    for key in (
        "max_inflight_batches",
        "score_attempts",
        "startup_timeout",
        "task_timeout",
        "cpu_threads",
    ):
        if type(c["execution"][key]) is not int or c["execution"][key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(c["sampling"]["batch_size"]) is not int or c["sampling"]["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    fields(
        c["evaluation"],
        "suite bringup_suite monitor_suite bootstrap_seed bootstrap_repeats confidence",
        "evaluation",
    )
    if (
        not 0 < c["evaluation"]["confidence"] < 1
        or c["evaluation"]["bootstrap_repeats"] < 1
    ):
        raise ValueError("invalid uncertainty settings")
    if (
        type(c["evaluation"]["bootstrap_repeats"]) is not int
        or type(c["evaluation"]["bootstrap_seed"]) is not int
    ):
        raise ValueError("bootstrap count/seed must be integers")
    names = []
    for arm in c["arms"]:
        fields(arm, "name rule steps guidance", "arm")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", arm["name"]):
            raise ValueError("arm name must be a safe filename component")
        if (
            arm["rule"] not in {"euler", "renoise"}
            or type(arm["steps"]) is not int
            or arm["steps"] < 1
        ):
            raise ValueError("invalid sampler arm")
        if not math.isfinite(arm["guidance"]) or arm["guidance"] != 1:
            raise ValueError("this increment implements conditional CFG=1 sampling")
        names.append(arm["name"])
    if not names or len(set(names)) != len(names):
        raise ValueError("arm names must be nonempty and unique")
    fields(c["scorers"], "hpsv3 vqascore", "scorers")
    fields(
        c["scorers"]["hpsv3"],
        "source_repository source_revision weights_repository weights_revision weights_file backbone_repository backbone_revision primary_output",
        "hpsv3",
    )
    fields(
        c["scorers"]["vqascore"],
        "source_repository source_revision release model weights_repository weights_revision question_template answer_template max_new_tokens temperature primary_output",
        "vqascore",
    )
    for scorer in c["scorers"].values():
        for key in ("source_revision", "weights_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", scorer[key]):
                raise ValueError("scorer source/weights must be pinned to commits")
    vqa = c["scorers"]["vqascore"]
    if (
        vqa["max_new_tokens"] != 1
        or vqa["answer_template"] != "Yes"
        or vqa["model"] != "qwen3-vl-8b"
    ):
        raise ValueError(
            "initial adapter supports one-token Yes scoring with qwen3-vl-8b"
        )
    if (
        vqa["temperature"] <= 0
        or not math.isfinite(vqa["temperature"])
        or vqa["question_template"].count("{}") != 1
    ):
        raise ValueError("invalid VQA question/temperature")
    if (
        c["scorers"]["hpsv3"]["primary_output"] != "mu"
        or vqa["primary_output"] != "probability"
    ):
        raise ValueError("unknown primary scorer output")
    return c
