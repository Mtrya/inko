"""Recoverable role/optimizer state at cold and complete joint-round boundaries."""

from . import checkpoint
from .distributed import unanimous_finite
from .optim import RoleOptimizer, finite
from .schedules import cold_point
from .schema import require
from .stages import Clock, refresh_decay


class TrainingState:
    def __init__(
        self,
        config,
        roles,
        student,
        fake,
        clock,
        *,
        initial_generator_steps=0,
        initial_fake_steps=0,
    ):
        self.config, self.roles, self.student, self.fake, self.clock = (
            config,
            roles,
            student,
            fake,
            clock,
        )
        self.initial_generator_steps, self.initial_fake_steps = (
            initial_generator_steps,
            initial_fake_steps,
        )
        self.last_reward = None
        self._optimizers()

    def _optimizers(self):
        c = self.config
        horizon = (
            c.stages.cold_steps if self.clock.stage == "cold" else c.stages.joint_steps
        )
        self.student_optimizer = RoleOptimizer(
            self.roles.student, c.student_optimizer, horizon
        )
        self.fake_optimizer = (
            RoleOptimizer(self.roles.fake, c.fake_optimizer, horizon * c.fake.updates)
            if self.roles.fake is not None
            else None
        )

    def enter_joint(self):
        require(
            self.clock.next_action(self.config) == "transition",
            "not a cold-stage boundary",
        )
        if self.config.joint.dmd_coefficient == 0:
            self.fake = None
            self.fake_optimizer = None
        self.roles.enter_joint()
        self.clock.transition(self.config)
        self._optimizers()

    def commit_round(self):
        self.clock.check_round(self.config)
        decay = None
        if self.clock.stage == "joint" and self.roles.old is not None:
            decay = refresh_decay(self.config.old_policy, self.clock.joint_rounds)
            self.roles.refresh_old(decay)
            unanimous_finite(
                finite(list(self.roles.old.parameters())),
                next(self.roles.student.parameters()).device,
            )
        self.clock.commit_round(self.config, decay)
        return decay

    def state_dict(self):
        clock, c = self.clock, self.config
        clock.validate(c, self.initial_generator_steps, self.initial_fake_steps)
        require(
            self.student_optimizer.steps == clock.stage_generator_steps,
            "student optimizer/stage clock differs",
        )
        if self.fake_optimizer is not None:
            require(
                self.fake_optimizer.steps
                == clock.stage_generator_steps * c.fake.updates,
                "fake optimizer/stage clock differs",
            )
        return {
            **self.roles.state_dict(),
            **clock.state_dict(),
            "student_optimizer": self.student_optimizer.state_dict(),
            "fake_optimizer": None
            if self.fake_optimizer is None
            else self.fake_optimizer.state_dict(),
            "initial_generator_steps": self.initial_generator_steps,
            "initial_fake_steps": self.initial_fake_steps,
            "last_reward": self.last_reward,
            "next_action": clock.next_action(c),
            "cold_schedule": cold_point(c, clock.stage_generator_steps)
            if clock.stage == "cold"
            else None,
            "old_schedule": {
                "clock": "committed_joint_rounds",
                "next_decay": refresh_decay(c.old_policy, clock.joint_rounds),
            }
            if self.roles.old is not None
            else None,
            "architecture": self.roles.student.get_config(),
            "ownership": self.roles.ownership(),
        }

    def load_state_dict(self, saved):
        require(self.clock.stage == saved["stage"], "resume role stage differs")
        clock = Clock.from_state(saved)
        clock.validate(
            self.config, self.initial_generator_steps, self.initial_fake_steps
        )
        require(
            (saved["initial_generator_steps"], saved["initial_fake_steps"])
            == (self.initial_generator_steps, self.initial_fake_steps),
            "resume initialization clocks differ",
        )
        self.roles.load_state_dict(saved)
        self.student_optimizer.load_state_dict(saved["student_optimizer"])
        require(
            (self.fake_optimizer is None) == (saved["fake_optimizer"] is None),
            "fake optimizer presence differs",
        )
        if self.fake_optimizer is not None:
            self.fake_optimizer.load_state_dict(saved["fake_optimizer"])
        self.clock = clock
        self.initial_generator_steps = saved["initial_generator_steps"]
        self.initial_fake_steps = saved["initial_fake_steps"]
        self.last_reward = saved["last_reward"]
        checkpoint.assert_exact(saved, self.state_dict())
