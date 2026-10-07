"""Per-role optimizers, finite transaction admission and explicit step clocks."""

import math

import torch

from src.optim.muon import CONDITIONING_WEIGHTS, build_param_groups

from .distributed import unanimous_finite


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from tensors(v)


def finite(values):
    # Group scalar predicates before one host synchronization per device.
    flags = [torch.isfinite(t).all() for t in values]
    by_device = {}
    for flag in flags:
        by_device.setdefault(flag.device, []).append(flag)
    return all(torch.stack(group).all().item() for group in by_device.values())


class RoleOptimizer:
    def __init__(self, model, config, total_steps):
        self.config, self.total_steps, self.steps = config, total_steps, 0
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if config.kind == "adamw":
            ordinary = [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n not in CONDITIONING_WEIGHTS
            ]
            conditioning = [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and n in CONDITIONING_WEIGHTS
            ]
            groups = []
            for parameters, decay in (
                (ordinary, config.weight_decay),
                (conditioning, config.conditioning_weight_decay),
            ):
                if parameters:
                    groups.append({"params": parameters, "weight_decay": decay})
            self.optimizers = [
                torch.optim.AdamW(
                    groups,
                    lr=config.lr,
                    betas=config.betas,
                    eps=config.epsilon,
                    weight_decay=config.weight_decay,
                )
            ]
        else:
            self.optimizers = build_param_groups(
                model,
                config.muon_lr,
                config.muon_weight_decay,
                config.lr,
                config.weight_decay,
                config.conditioning_weight_decay,
                config.epsilon,
                config.betas,
                config.muon_momentum,
            )
        self.base_rates = [
            [g["lr"] for g in opt.param_groups] for opt in self.optimizers
        ]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def rate_factor(self):
        c, step = self.config, self.steps
        if step < c.warmup_steps:
            return (step + 1) / c.warmup_steps
        if c.schedule == "constant":
            return 1.0
        progress = (step - c.warmup_steps) / max(
            1, self.total_steps - c.warmup_steps - 1
        )
        return c.minimum_ratio + (1 - c.minimum_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1))
        )

    def step(self, loss):
        device = self.parameters[0].device
        gradients = [p.grad for p in self.parameters if p.grad is not None]
        valid = bool(gradients) and finite(
            [
                loss,
                *self.parameters,
                *gradients,
                *tensors([o.state for o in self.optimizers]),
            ]
        )
        unanimous_finite(valid, device)
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.max_grad_norm, foreach=False
        )
        unanimous_finite(bool(torch.isfinite(norm)), device)
        factor = self.rate_factor()
        # Every optimizer in this role has passed admission before any mutates.
        for optimizer, rates in zip(self.optimizers, self.base_rates, strict=True):
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate * factor
            optimizer.step()
        unanimous_finite(
            finite([*self.parameters, *tensors([o.state for o in self.optimizers])]),
            device,
        )
        self.steps += 1
        return float(norm)

    def state_dict(self):
        return {
            "steps": self.steps,
            "base_rates": self.base_rates,
            "optimizers": [o.state_dict() for o in self.optimizers],
        }

    def load_state_dict(self, state):
        if (
            len(state["optimizers"]) != len(self.optimizers)
            or state["base_rates"] != self.base_rates
        ):
            raise ValueError("optimizer grouping changed")
        for optimizer, saved in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(saved)
        self.steps = state["steps"]
