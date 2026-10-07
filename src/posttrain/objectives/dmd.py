"""Inko flow-coordinate DMD estimators and detached fake-score targets."""

import torch


def expand_time(t, x):
    return t.float().reshape((-1,) + (1,) * (x.ndim - 1))


def clean_prediction(x, t, velocity):
    return x.float() + (1 - expand_time(t, x)) * velocity.float()


def forward_noise(clean, t, noise):
    t = expand_time(t, clean)
    return t * clean.detach().float() + (1 - t) * noise.float()


def dmd_gradient(clean, real, fake, t, *, estimator, epsilon):
    with torch.no_grad():
        delta = fake.float() - real.float()
        if estimator == "teacher_residual":
            residual = (clean.detach().float() - real.float()).abs().flatten(1).mean(1)
            delta = delta / expand_time(residual + epsilon, delta)
        elif estimator == "score_kl":
            time = expand_time(t, delta)
            delta = delta * time.square() / (1 - time).square()
        elif estimator != "clean_difference":
            raise ValueError("unknown DMD estimator")
        if not torch.isfinite(delta).all():
            raise FloatingPointError("nonfinite DMD gradient")
        return delta.detach()


def surrogate(clean, gradient):
    target = (clean.float() - gradient.detach()).detach()
    return 0.5 * (clean.float() - target).square().flatten(1).mean(1)


def fake_loss(velocity, clean, noise):
    return (
        (velocity.float() - (clean.detach().float() - noise.float()))
        .square()
        .flatten(1)
        .mean(1)
    )
