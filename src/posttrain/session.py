"""Recoverable role/optimizer state at committed collection-round boundaries."""

from . import checkpoint
from .clock import Clock
from .distributed import unanimous_finite
from .optim import RoleOptimizer, finite
from .schedules import refresh_decay
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
            self.roles.student, config.student_optimizer, config.training.steps
        )

    def commit_round(self):
        self.clock.check_round(self.config)
        decay = refresh_decay(self.config.old_policy, self.clock.committed_rounds)
        self.roles.refresh_old(decay)
        unanimous_finite(
            finite(list(self.roles.old.parameters())),
            next(self.roles.student.parameters()).device,
        )
        self.clock.commit_round(self.config, decay)
        return decay

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
            "old_schedule": {
                "clock": "committed_rounds",
                "next_decay": refresh_decay(c.old_policy, clock.committed_rounds),
            },
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
