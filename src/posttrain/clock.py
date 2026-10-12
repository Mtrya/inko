"""Committed step/round clock and old-policy refresh bookkeeping."""

from dataclasses import asdict, dataclass

from .schedules import refresh_decay
from .schema import require


@dataclass
class Clock:
    step: int = 0
    committed_rounds: int = 0
    skipped_rounds: int = 0
    old_policy_version: int = 0
    samples_used: int = 0
    samples_scored: int = 0
    old_refresh_decay: float | None = None

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

    def commit_round(self, config, decay):
        self.check_round(config)
        self.committed_rounds += 1
        self.samples_scored += (
            config.collection.prompts_per_round * config.collection.draws_per_prompt
        )
        if config.objective == "nft":
            require(
                decay is not None, "a committed round requires its old-policy refresh"
            )
            self.old_policy_version += 1
            self.old_refresh_decay = decay
        else:
            require(decay is None, "the ART objective has no old-policy refresh")

    def check_round(self, config):
        require(
            self.step
            == (self.committed_rounds - self.skipped_rounds + 1)
            * config.updates_per_round,
            "commit requires exactly one complete uncommitted collection round",
        )

    def skip_round(self, config):
        require(
            config.objective == "art" and self.next_action(config) == "collect",
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
                self.old_policy_version,
                self.samples_used,
                self.samples_scored,
            )
            >= 0,
            "invalid committed training clock",
        )
        total = config.collection.prompts_per_round * config.collection.draws_per_prompt
        refreshes = self.committed_rounds if config.objective == "nft" else 0
        require(
            self.committed_rounds <= config.training.rounds
            and self.step
            == (self.committed_rounds - self.skipped_rounds) * config.updates_per_round
            and (config.objective == "art" or self.skipped_rounds == 0)
            and self.samples_used == total * self.step
            and self.samples_scored == total * self.committed_rounds
            and self.old_policy_version == refreshes,
            "committed step/round/sample clocks differ",
        )
        if config.objective == "nft":
            expected = (
                refresh_decay(config.old_policy, self.committed_rounds - 1)
                if self.committed_rounds
                else None
            )
            require(
                self.old_refresh_decay == expected,
                "committed old-policy schedule differs",
            )
        else:
            require(
                self.old_refresh_decay is None,
                "the ART objective has no old-policy schedule",
            )

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, saved):
        return cls(**{name: saved[name] for name in cls.__dataclass_fields__})
