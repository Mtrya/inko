"""Old-policy refresh and training-input time schedules; each names its clock."""

import torch

from .schema import require


def refresh_decay(config, committed_rounds):
    require(committed_rounds >= 0, "negative old-policy round clock")
    if config.strategy == "copy":
        return 0.0
    if config.strategy == "constant":
        return config.decay
    return config.decay_start + (config.decay_end - config.decay_start) * min(
        committed_rounds / config.horizon, 1
    )


def time_draws(seeds, minimum, maximum):
    """Uniform physical time on the configured open interval, per sample."""
    draws = torch.stack(
        [
            torch.rand((), generator=torch.Generator().manual_seed(seed))
            for seed in seeds
        ]
    )
    return minimum + (maximum - minimum) * draws
