"""Explicit stage boundaries, generator/round clocks and old refresh schedule."""

from dataclasses import asdict, dataclass

from .schema import require


def round_position(config, round_index):
    require(round_index >= 0, "negative collection round")
    cold = config.stages.cold_steps
    if round_index < cold:
        return "cold", round_index, 1
    local = (round_index - cold) * config.joint_updates
    require(
        0 <= local < config.stages.joint_steps, "collection round outside stage plan"
    )
    return "joint", local, config.joint_updates


def rounds_before(config, step):
    cold = config.stages.cold_steps
    require(0 <= step <= config.training.steps, "generator step outside stage plan")
    if step <= cold:
        return step
    require(
        (step - cold) % config.joint_updates == 0,
        "cannot stop inside a joint collection round",
    )
    return cold + (step - cold) // config.joint_updates


def refresh_decay(config, joint_rounds):
    require(joint_rounds >= 0, "negative old-policy round clock")
    if config.strategy == "copy":
        return 0.0
    if config.strategy == "constant":
        return config.decay
    return config.decay_start + (config.decay_end - config.decay_start) * min(
        joint_rounds / config.horizon, 1
    )


@dataclass
class Clock:
    stage: str
    step: int = 0
    stage_generator_steps: int = 0
    global_generator_steps: int = 0
    fake_steps: int = 0
    committed_rounds: int = 0
    joint_rounds: int = 0
    old_policy_version: int | None = None
    samples_used: int = 0
    samples_scored: int = 0
    old_refresh_decay: float | None = None

    @classmethod
    def initial(cls, config, generator_steps=0, fake_steps=0):
        stage = "cold" if config.stages.cold_steps else "joint"
        return cls(
            stage,
            global_generator_steps=generator_steps,
            fake_steps=fake_steps,
            old_policy_version=0
            if stage == "joint" and config.joint.nft_coefficient > 0
            else None,
        )

    def next_action(self, config):
        if self.step == config.training.steps:
            return "done"
        if (
            self.stage == "cold"
            and self.stage_generator_steps == config.stages.cold_steps
        ):
            return "transition"
        return "collect"

    def transition(self, config):
        require(
            self.next_action(config) == "transition",
            "transition requires completed cold boundary",
        )
        self.stage = "joint"
        self.stage_generator_steps = 0
        self.old_policy_version = 0 if config.joint.nft_coefficient > 0 else None

    def update(self, config):
        require(
            self.next_action(config) == "collect", "no update beyond a stage boundary"
        )
        self.step += 1
        self.stage_generator_steps += 1
        self.global_generator_steps += 1
        dmd = config.dmd_enabled(self.stage)
        nft = self.stage == "joint" and config.joint.nft_coefficient > 0
        self.fake_steps += config.fake.updates if dmd else 0
        self.samples_used += (
            config.collection.prompts_per_round
            * config.collection.draws_per_prompt
            * (int(dmd) + int(nft))
        )

    def commit_round(self, config, decay=None):
        self.check_round(config)
        self.committed_rounds += 1
        scored = (self.stage == "cold" and config.objective.mode == "rtdm") or (
            self.stage == "joint" and config.joint.nft_coefficient > 0
        )
        if scored:
            self.samples_scored += (
                config.collection.prompts_per_round * config.collection.draws_per_prompt
            )
        if self.stage == "joint":
            if config.joint.nft_coefficient > 0:
                require(decay is not None, "NFT round requires old refresh")
                self.old_policy_version += 1
                self.old_refresh_decay = decay
            self.joint_rounds += 1

    def check_round(self, config):
        require(
            rounds_before(config, self.step) == self.committed_rounds + 1,
            "commit requires exactly one complete uncommitted collection round",
        )

    def validate(self, config, initial_generator_steps, initial_fake_steps):
        require(
            self.stage in {"cold", "joint"}
            and min(
                self.step,
                self.stage_generator_steps,
                self.global_generator_steps,
                self.fake_steps,
                self.committed_rounds,
                self.joint_rounds,
                self.samples_used,
                self.samples_scored,
                initial_generator_steps,
                initial_fake_steps,
            )
            >= 0,
            "invalid committed training clock",
        )
        cold = config.stages.cold_steps
        if self.stage == "cold":
            require(
                cold > 0 and self.step == self.stage_generator_steps <= cold,
                "cold stage clock differs",
            )
            cold_done, joint_done = self.step, 0
        else:
            require(
                config.stages.joint_steps > 0
                and self.step == cold + self.stage_generator_steps
                and self.stage_generator_steps <= config.stages.joint_steps,
                "joint stage clock differs",
            )
            cold_done, joint_done = cold, self.stage_generator_steps
        nft = config.joint.nft_coefficient > 0
        dmd = config.joint.dmd_coefficient > 0
        joint_rounds = joint_done // config.joint_updates
        require(
            self.committed_rounds == rounds_before(config, self.step)
            and self.joint_rounds == joint_rounds
            and self.global_generator_steps == initial_generator_steps + self.step
            and self.fake_steps
            == initial_fake_steps
            + (cold_done + int(dmd) * joint_done) * config.fake.updates,
            "committed generator/fake/round clocks differ",
        )
        total = config.collection.prompts_per_round * config.collection.draws_per_prompt
        require(
            self.samples_used
            == total * (cold_done + (int(dmd) + int(nft)) * joint_done)
            and self.samples_scored
            == total
            * (
                int(config.objective.mode == "rtdm") * cold_done
                + int(nft) * joint_rounds
            ),
            "committed sample clocks differ",
        )
        old = self.stage == "joint" and nft
        expected_decay = (
            refresh_decay(config.old_policy, joint_rounds - 1)
            if old and joint_rounds
            else None
        )
        require(
            self.old_policy_version == (joint_rounds if old else None)
            and self.old_refresh_decay == expected_decay,
            "committed old-policy clock/schedule differs",
        )

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, saved):
        return cls(**{name: saved[name] for name in cls.__dataclass_fields__})
