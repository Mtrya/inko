"""Committed optimizer-update, collection-round and sample clocks."""

from dataclasses import asdict, dataclass

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
