"""Detached terminal reward weighting and exact selected-state replay admission."""

import torch


def verify_replay(prediction, collected):
    if prediction.shape != collected.shape or not torch.equal(
        prediction.detach(), collected
    ):
        difference = (
            float((prediction.detach() - collected).abs().max())
            if prediction.shape == collected.shape
            else None
        )
        raise RuntimeError(f"selected prediction replay changed (max_abs={difference})")


def weighted_losses(losses, weights):
    weights = weights.detach().to(losses.device, torch.float32)
    if (
        weights.shape != losses.shape
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
    ):
        raise ValueError("invalid detached RT-DM weights")
    return losses * weights
