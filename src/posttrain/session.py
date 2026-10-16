"""Recoverable role/optimizer state at committed collection-round boundaries."""

from . import checkpoint
from .clock import Clock
from .optim import RoleOptimizer
from .schema import require


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
