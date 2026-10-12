"""Committed step/round clock and old-policy refresh bookkeeping."""

from dataclasses import asdict, dataclass

from .schedules import refresh_decay
from .schema import require


def rounds_before(config, step):
    """Committed collection rounds at a stop boundary; stops split no round."""
    updates = config.nft.updates_per_round
    require(
        0 <= step <= config.training.steps, "generator step outside the training plan"
    )
    require(step % updates == 0, "cannot stop inside a collection round")
    return step // updates


@dataclass
class Clock:
    step: int = 0
    committed_rounds: int = 0
    old_policy_version: int = 0
    samples_used: int = 0
    samples_scored: int = 0
    old_refresh_decay: float | None = None

    def next_action(self, config):
        return "done" if self.step == config.training.steps else "collect"

    def update(self, config):
        require(
            self.next_action(config) == "collect",
            "no student update beyond the training horizon",
        )
        self.step += 1
        self.samples_used += (
            config.collection.prompts_per_round * config.collection.draws_per_prompt
        )

    def commit_round(self, config, decay):
        self.check_round(config)
        require(decay is not None, "a committed round requires its old-policy refresh")
        self.committed_rounds += 1
        self.old_policy_version += 1
        self.old_refresh_decay = decay
        self.samples_scored += (
            config.collection.prompts_per_round * config.collection.draws_per_prompt
        )

    def check_round(self, config):
        require(
            rounds_before(config, self.step) == self.committed_rounds + 1,
            "commit requires exactly one complete uncommitted collection round",
        )

    def validate(self, config):
        require(
            min(
                self.step,
                self.committed_rounds,
                self.old_policy_version,
                self.samples_used,
                self.samples_scored,
            )
            >= 0,
            "invalid committed training clock",
        )
        total = config.collection.prompts_per_round * config.collection.draws_per_prompt
        require(
            self.committed_rounds == rounds_before(config, self.step)
            and self.samples_used == total * self.step
            and self.samples_scored == total * self.committed_rounds
            and self.old_policy_version == self.committed_rounds,
            "committed step/round/sample clocks differ",
        )
        expected = (
            refresh_decay(config.old_policy, self.committed_rounds - 1)
            if self.committed_rounds
            else None
        )
        require(
            self.old_refresh_decay == expected, "committed old-policy schedule differs"
        )

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, saved):
        return cls(**{name: saved[name] for name in cls.__dataclass_fields__})
