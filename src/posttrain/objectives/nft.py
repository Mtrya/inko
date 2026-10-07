"""Detached implicit NFT branches in Inko's data-increasing flow coordinates."""

import math

import torch

from .dmd import clean_prediction


def mean_elements(value):
    return value.float().flatten(1).mean(1)


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("nonfinite NFT input/loss")


def losses(
    new,
    old,
    terminal,
    noise,
    times,
    probability,
    *,
    parameterization,
    beta,
    clip,
    epsilon,
):
    if any(not math.isfinite(v) or v <= 0 for v in (beta, clip, epsilon)):
        raise ValueError("NFT beta/clip/epsilon must be positive")
    old, terminal, noise = (
        old.detach().float(),
        terminal.detach().float(),
        noise.detach().float(),
    )
    times = times.detach().float()
    if not (new.shape == old.shape == terminal.shape == noise.shape) or times.shape != (
        new.shape[0],
    ):
        raise ValueError("NFT latent/time shapes differ")
    probability = probability.detach().float()
    if probability.shape != (new.shape[0],) or not bool(
        ((probability >= 0) & (probability <= 1)).all()
    ):
        raise ValueError("invalid NFT probability")
    finite(new, old, terminal, noise, times, probability)
    if not bool(((times >= 0) & (times < 1)).all()):
        raise ValueError("NFT physical times must exclude the terminal endpoint")
    plus = (1 - beta) * old + beta * new.float()
    minus = (1 + beta) * old - beta * new.float()
    if parameterization == "clean_adaptive":
        shape = (-1,) + (1,) * (new.ndim - 1)
        z = (
            times.float().view(shape) * terminal
            + (1 - times.float().view(shape)) * noise
        )
        positive = clean_prediction(z, times, plus) - terminal
        negative = clean_prediction(z, times, minus) - terminal
        hp = mean_elements(positive.detach().abs()).clamp_min(epsilon)
        hm = mean_elements(negative.detach().abs()).clamp_min(epsilon)
        lp, lm = (
            mean_elements(positive.square()) / hp,
            mean_elements(negative.square()) / hm,
        )
    elif parameterization == "velocity_mse":
        target = terminal - noise
        lp, lm = (
            mean_elements((plus - target).square()),
            mean_elements((minus - target).square()),
        )
    else:
        raise ValueError("unknown NFT parameterization")
    result = clip / beta * (probability * lp + (1 - probability) * lm)
    finite(result)
    return result


def reference_losses(new, reference):
    if new.shape != reference.shape:
        raise ValueError("reference prediction shapes differ")
    result = mean_elements((new.float() - reference.detach().float()).square())
    finite(result)
    return result
