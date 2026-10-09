from dataclasses import replace

import pytest

from src.posttrain.config import (
    Arm,
    Checkpoint,
    Config,
    Diversity,
    Evaluation,
    Execution,
    Sampling,
    Scorer,
)
from src.posttrain.scoring.contracts import Channel


@pytest.fixture
def sampling_config():
    """Explicit test inputs; no expectations are sourced from production configs."""
    channel = Channel("fit", "higher", (0.0, 1.0), "Fixture alignment")
    scorer = Scorer(
        "unfamiliar",
        "a" * 64,
        "rubric-v1",
        ("rubric",),
        (channel,),
        2,
        2.0,
        2,
        2,
        "shared_file",
        "none",
        0,
    )
    checkpoint = Checkpoint("first", "ema", "1" * 64, "2" * 64)
    arm = Arm("reference", checkpoint, "euler", 2, 1.0, "none")
    return Config(
        Sampling("float32", 1, True, 2, {"1:1": (32, 32)}),
        Execution((0,), 2, 10, 10, 1),
        Evaluation(
            "unused",
            "unused",
            "unused",
            "unused",
            ("unfamiliar.fit",),
            0,
            100,
            0.95,
            Diversity(False, "", "", 1, 1),
        ),
        (arm, replace(arm, name="candidate", rule="renoise")),
        (scorer,),
    )
