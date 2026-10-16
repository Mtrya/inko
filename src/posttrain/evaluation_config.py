"""Training-time evaluation controls and their standalone benchmark recipe."""

from dataclasses import dataclass

from .config import Arm, Checkpoint, Config, Diversity, Evaluation, Execution, Sampling
from .schema import require


@dataclass(frozen=True)
class IntervalEvaluation:
    every: int
    suite: str
    monitor_suite: str
    prompt_limit: int
    draws_per_prompt: int
    steps: int
    guidance: float
    shift: str
    shapes: dict[str, tuple[int, int]]
    channels: tuple[str, ...]
    diversity_asset: str
    suites: tuple[str, ...] = ("development", "monitor")
    weights: tuple[str, ...] = ("ema",)
    bootstrap_seed: int = 7
    bootstrap_repeats: int = 2000
    confidence: float = 0.95

    def __post_init__(self):
        require(
            self.every > 0 and self.prompt_limit >= 0 and self.draws_per_prompt > 0,
            "invalid evaluation cadence/prompt selection",
        )
        require(
            bool(self.suites)
            and len(set(self.suites)) == len(self.suites)
            and set(self.suites) <= {"development", "monitor"},
            "invalid evaluation suites",
        )
        require(
            bool(self.weights)
            and len(set(self.weights)) == len(self.weights)
            and set(self.weights) <= {"live", "ema"},
            "evaluation weights must be live or ema",
        )
        require(0 <= self.bootstrap_seed < 2**63, "invalid evaluation seed")


def benchmark_config(training):
    c, e, v = training.collection, training.execution, training.evaluation
    services = {channel.split(".")[0] for channel in v.channels}
    return Config(
        Sampling(e.precision, e.microbatch, c.text_exit_layer, v.shapes),
        Execution(
            e.learner_devices,
            2 * len(e.learner_devices),
            e.startup_timeout,
            e.round_timeout,
            e.cpu_threads,
        ),
        Evaluation(
            v.suite,
            v.monitor_suite,
            v.channels,
            v.bootstrap_seed,
            v.bootstrap_repeats,
            v.confidence,
            Diversity(("development",), v.diversity_asset, 2, e.cpu_threads),
            v.prompt_limit,
            v.draws_per_prompt,
        ),
        (
            Arm(
                "base",
                Checkpoint(training.initialization.asset, "initial"),
                v.steps,
                v.guidance,
                v.shift,
            ),
        ),
        tuple(s for s in training.scorers if s.id in services),
    )


def boundaries(config):
    every = config.evaluation.every
    rounds = set(range(every, config.training.rounds + 1, every))
    rounds.add(config.training.rounds)
    yield from sorted(rounds)


def evaluation_due(config, round_number):
    return round_number > 0 and (
        round_number % config.evaluation.every == 0
        or round_number == config.training.rounds
    )
