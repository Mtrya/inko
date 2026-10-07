"""Cold generator-clock schedules; fake updates never advance annealing."""

import math
import random

import torch


def cosine_remaining(step, horizon):
    if step < 0 or horizon <= 0:
        raise ValueError("invalid generator clock/horizon")
    return 0.5 * (1 + math.cos(math.pi * min(step / horizon, 1)))


def cold_point(config, step):
    dg, rs = config.dyna_dg, config.dyna_rs
    remaining = cosine_remaining(step, rs.horizon)
    return {
        "generator_step": step,
        "real_scale": dg.start_scale * cosine_remaining(step, dg.horizon)
        if dg.enabled
        else 0.0,
        "distribution": "beta_noise_coordinate" if rs.enabled else "uniform_physical",
        "alpha": 1 + (rs.alpha_start - 1) * remaining if rs.enabled else 1.0,
        "beta": 1 + (rs.beta_start - 1) * remaining if rs.enabled else 1.0,
        "dyna_dg": {
            "enabled": dg.enabled,
            "horizon": dg.horizon,
            "start_scale": dg.start_scale,
        },
        "dyna_rs": {
            "enabled": rs.enabled,
            "horizon": rs.horizon,
            "alpha_start": rs.alpha_start,
            "beta_start": rs.beta_start,
        },
    }


def physical_times(draws, minimum, maximum):
    """Noise coordinate is reversed relative to Inko's data-increasing time."""
    if not 0 < minimum < maximum < 1 or not bool(((draws >= 0) & (draws <= 1)).all()):
        raise ValueError("invalid bounded noise coordinate")
    return minimum + (maximum - minimum) * (1 - draws.float())


def time_draws(seeds, point, minimum, maximum):
    # Host generators are sample-local and do not alter Python/Torch global RNG.
    if point["distribution"] == "beta_noise_coordinate":
        draws = torch.tensor(
            [
                random.Random(seed).betavariate(point["alpha"], point["beta"])
                for seed in seeds
            ],
            dtype=torch.float32,
        )
        return physical_times(draws, minimum, maximum)
    draws = torch.stack(
        [
            torch.rand((), generator=torch.Generator().manual_seed(seed))
            for seed in seeds
        ]
    )
    return minimum + (maximum - minimum) * draws
