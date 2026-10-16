"""Training state, round accounting and exact checkpoint recovery."""

from dataclasses import asdict, dataclass

from . import checkpoint
from .optim import RoleOptimizer
from .schema import require


@dataclass
class Clock:
    step: int = 0
    committed_rounds: int = 0
    skipped_rounds: int = 0
    samples_used: int = 0
    samples_scored: int = 0

    def next_action(self, config):
        return "done" if self.committed_rounds == config.training.rounds else "collect"

    def update(self, config):
        require(
            self.next_action(config) == "collect"
            and self.step
            < (self.committed_rounds - self.skipped_rounds + 1)
            * config.updates_per_round,
            "no student update beyond the training horizon",
        )
        self.step += 1
        self.samples_used += (
            config.collection.prompts_per_round * config.collection.draws_per_prompt
        )

    def commit_round(self, config):
        self.check_round(config)
        self.committed_rounds += 1
        self.samples_scored += (
            config.collection.prompts_per_round * config.collection.draws_per_prompt
        )

    def check_round(self, config):
        require(
            self.step
            == (self.committed_rounds - self.skipped_rounds + 1)
            * config.updates_per_round,
            "commit requires exactly one complete uncommitted collection round",
        )

    def skip_round(self, config):
        require(
            self.next_action(config) == "collect",
            "only an unfinished ART run can skip a collection",
        )
        self.validate(config)
        self.committed_rounds += 1
        self.skipped_rounds += 1
        self.samples_scored += (
            config.collection.prompts_per_round * config.collection.draws_per_prompt
        )

    def validate(self, config):
        require(
            min(
                self.step,
                self.committed_rounds,
                self.skipped_rounds,
                self.samples_used,
                self.samples_scored,
            )
            >= 0,
            "invalid committed training clock",
        )
        total = config.collection.prompts_per_round * config.collection.draws_per_prompt
        require(
            self.committed_rounds <= config.training.rounds
            and self.step
            == (self.committed_rounds - self.skipped_rounds) * config.updates_per_round
            and self.samples_used == total * self.step
            and self.samples_scored == total * self.committed_rounds,
            "committed step/round/sample clocks differ",
        )

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, saved):
        return cls(**{name: saved[name] for name in cls.__dataclass_fields__})


class TrainingState:
    def __init__(self, config, roles, student, clock):
        self.config, self.roles, self.student, self.clock = (
            config,
            roles,
            student,
            clock,
        )
        self.last_reward = None
        self.student_optimizer = RoleOptimizer(
            self.roles.student,
            config.student_optimizer,
        )

    def commit_round(self):
        self.clock.commit_round(self.config)

    def skip_inactive_art_round(self, guidance):
        """Consume a globally inactive collection without an optimizer/EMA step.

        An enabled reference penalty remains an independent training objective.
        """
        if self.config.reference_coefficient or bool(guidance.active.any()):
            return False
        self.clock.skip_round(self.config)
        return True

    def state_dict(self):
        clock, c = self.clock, self.config
        clock.validate(c)
        require(
            self.student_optimizer.steps == clock.step,
            "student optimizer/step clock differs",
        )
        return {
            **self.roles.state_dict(),
            **clock.state_dict(),
            "student_optimizer": self.student_optimizer.state_dict(),
            "last_reward": self.last_reward,
            "next_action": clock.next_action(c),
            "architecture": self.roles.student.get_config(),
            "ownership": self.roles.ownership(),
        }

    def load_state_dict(self, saved):
        clock = Clock.from_state(saved)
        clock.validate(self.config)
        self.roles.load_state_dict(saved)
        self.student_optimizer.load_state_dict(saved["student_optimizer"])
        self.clock = clock
        self.last_reward = saved["last_reward"]
        checkpoint.assert_exact(saved, self.state_dict())
