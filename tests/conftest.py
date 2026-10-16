import copy
from dataclasses import replace

import pytest
import torch

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
from src.posttrain.evaluation_config import IntervalEvaluation
from src.posttrain.scoring.contracts import Channel
from src.posttrain.training_config import (
    ART,
    Collection,
    Initialization,
    Optimizer,
    RewardChannel,
    Rewards,
    Student,
    Training,
    TrainingConfig,
    TrainingExecution,
)


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
    arm = Arm("reference", checkpoint, 2, 1.0, "none")
    return Config(
        Sampling("float32", 1, 2, {"1:1": (32, 32)}),
        Execution((0,), 2, 10, 10, 1),
        Evaluation(
            "unused",
            "unused",
            ("unfamiliar.fit",),
            0,
            100,
            0.95,
            Diversity((), "", 1, 1),
        ),
        (arm, replace(arm, name="candidate", guidance=2.0)),
        (scorer,),
    )


@pytest.fixture
def small_model():
    """Tiny active Inko builder; callers deepcopy it when they need two roles."""
    from src.models.inko import Inko

    def make():
        model = Inko(
            hidden_size=16,
            num_heads=2,
            double_stream_depth=1,
            single_stream_depth=1,
            mlp_ratio=2,
        )
        torch.nn.init.normal_(model.final_layer[1].weight, std=0.1)
        # Exercise active transformer branches, not Inko's deliberately closed
        # fresh-init gates.
        for name, parameter in model.named_parameters():
            if "modulation" in name and name.endswith("bias"):
                torch.nn.init.constant_(parameter, 0.2)
        return model

    return make


@pytest.fixture
def training_config(sampling_config):
    """Explicit single-stage ART inputs; recipes never source these contracts."""
    return TrainingConfig(
        initialization=Initialization("initial"),
        student=Student("full", 2, 2.0, "attention_mlp"),
        collection=Collection(
            prompts="fixture.jsonl",
            prompts_per_round=2,
            draws_per_prompt=2,
            height=16,
            width=16,
            steps=2,
            shift="shape",
            text_exit_layer=1,
        ),
        art=ART(reference_coefficient=1e-4),
        student_optimizer=Optimizer(
            kind="adamw",
            lr=1e-4,
            betas=(0.9, 0.999),
            epsilon=1e-8,
            weight_decay=0,
            conditioning_weight_decay=0,
            max_grad_norm=1,
            muon_lr=1e-3,
            muon_weight_decay=0,
            muon_momentum=0.95,
        ),
        execution=TrainingExecution(
            learner_devices=(0,),
            microbatch=1,
            cpu_threads=1,
            collective_timeout=30,
            startup_timeout=30,
            round_timeout=30,
            precision="float32",
            activation_checkpointing=False,
            intra_host_roce=False,
            deterministic_algorithms=True,
        ),
        training=Training(7, 3, 1, 0, 0.9),
        scorers=sampling_config.scorers,
        rewards=Rewards((RewardChannel("unfamiliar.fit", 1.0, "none"),), 1e-4),
        evaluation=IntervalEvaluation(
            every=1,
            suite="unused",
            monitor_suite="unused",
            prompt_limit=0,
            draws_per_prompt=2,
            steps=2,
            guidance=1.0,
            shift="none",
            shapes={"1:1": (32, 32)},
            channels=("unfamiliar.fit",),
            diversity_asset="lpips",
        ),
    )


@pytest.fixture
def new_roles():
    from src.posttrain.roles import Roles

    return lambda config, base: Roles(copy.deepcopy(base), config)
