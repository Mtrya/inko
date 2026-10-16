"""Frozen-target ART regression with directional corrective masking.

The reference implementation runs in decreasing noise time sigma with velocity
`eps - x0`; Inko's data-increasing time `t = 1 - sigma` and velocity `x0 - eps`
carry the same update after `dt -> -dt`, so the frozen target becomes
`cached + G` and the timestep weight becomes `1 - t`. The trust-region proxy
grades a move against the reference transition variance, whose factor `-dt_ref`
is `dt_ours > 0` here; the ODE coefficient enters only squared, so the proxy
value is unchanged by the mapping.

Constants are the paper's; they are not campaign factors.
"""

import torch

NOISE_LEVEL = 0.8
KL_THRESHOLD = 1e-5
EPS = 1e-5


def mean_elements(value):
    return value.float().flatten(1).mean(1)


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("nonfinite ART input/loss")


def reference_losses(new, reference):
    if new.shape != reference.shape:
        raise ValueError("reference prediction shapes differ")
    result = mean_elements((new.float() - reference.detach().float()).square())
    finite(result)
    return result


def losses(
    new,
    cached,
    times,
    deltas,
    direction,
    magnitude,
    *,
    eps=EPS,
    noise=NOISE_LEVEL,
    threshold=KL_THRESHOLD,
):
    """Per-item ART loss and keep mask; every input except `new` is frozen.

    `cached` is the rollout policy's prediction at the trained state, `direction`
    its guidance direction (zero for an inactive group) and `magnitude` its
    per-item guidance magnitude. `deltas` are the physical-time steps of the
    collection grid. An item trains only when its group is active and its move is
    not both above the proxy threshold and outward.
    """
    new = new.float()
    cached = cached.detach().float()
    direction = direction.detach().float()
    times = times.detach().float().reshape(-1)
    deltas = deltas.detach().float().reshape(-1)
    magnitude = magnitude.detach().float().reshape(-1)
    if not (
        new.shape == cached.shape == direction.shape
        and new.shape[0] == times.numel() == deltas.numel() == magnitude.numel()
    ):
        raise ValueError("ART item shapes differ")
    finite(new, cached, direction, times, deltas, magnitude)
    if not bool(((times >= 0) & (times < 1)).all()):
        raise ValueError("ART physical times must exclude the data endpoint")
    if not bool((deltas > 0).all()):
        raise ValueError("ART states must move toward data")
    if not bool((magnitude >= 0).all()):
        raise ValueError("invalid ART guidance magnitude")
    shape = (-1,) + (1,) * (new.ndim - 1)
    displacement = magnitude.view(shape) * direction
    target = (cached + displacement).detach()
    residual = target - new
    # Per-item scalars stay flat: a [N] weight broadcast against a [N, C, H, W]
    # mean would otherwise multiply along the latent axis instead of the item axis.
    unmasked = (1 - times) * mean_elements(residual.square())
    movement = (new - cached).detach()
    alignment = mean_elements(movement * residual.detach())
    initial = torch.isclose(times, torch.zeros_like(times), atol=eps, rtol=0)
    variance = (noise**2 * (1 - times) / times.clamp_min(eps) * deltas).clamp_min(0)
    std = torch.where(initial, torch.full_like(times, noise), variance.sqrt())
    coefficient = torch.where(initial, deltas, deltas * (1 + 0.5 * noise**2))
    proxy = (
        coefficient.square()
        * mean_elements(movement.square())
        / (2 * std.square().clamp_min(eps**2))
    )
    raw_rms = direction.square().flatten(1).mean(1).sqrt()
    active = (magnitude > 0) & (raw_rms > 0) & (times < 1)
    masked = active & (proxy >= threshold) & (alignment > 0)
    keep = active & ~masked
    result = torch.where(keep, unmasked, torch.zeros_like(unmasked))
    finite(result, proxy, alignment)
    return result, keep, masked
