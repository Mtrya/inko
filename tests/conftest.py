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
    channel = Channel("fit", "higher")
    scorer = Scorer(
        "unfamiliar",
        (channel,),
        2,
        2.0,
        2,
        2,
        "shared_file",
    )
    checkpoint = Checkpoint("first", "ema")
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
            Diversity(False, "", 1, 1),
        ),
        (arm, replace(arm, name="candidate", rule="renoise")),
        (scorer,),
    )
