"""Resident student, frozen reference and EMA views."""

import copy

import torch
from torch import nn

from .adapters import install_adapters


class Velocity(nn.Module):
    def __init__(self, model, precision):
        super().__init__()
        self.model, self.precision = model, precision

    def forward(self, x, t, conditioning):
        dtype = getattr(torch, self.precision)
        with torch.autocast(x.device.type, dtype=dtype, enabled=dtype != torch.float32):
            output = self.model(x, t.float(), *conditioning).float()
        return output


class Roles:
    def __init__(self, student, config):
        self.config = config
        self.student = student.float().eval().requires_grad_(True)
        self.student_targets = ()
        if config.student.mode == "lora":
            self.student_targets = install_adapters(
                self.student,
                rank=config.student.rank,
                scale=config.student.alpha,
                targets=config.student.targets,
            )
        self.student.activation_checkpointing = (
            config.execution.activation_checkpointing
        )
        self.student_field = Velocity(self.student, config.execution.precision)
        self.reference = self.reference_field = None
        if config.reference_coefficient > 0:
            self.reference = self.frozen_policy()
            self.reference_field = Velocity(self.reference, config.execution.precision)
        self.reset_ema()

    def frozen_policy(self):
        # Frozen base parameters/buffers are shared only for LoRA. All trainable
        # adapter tensors and module objects are actual independent copies.
        memo = {}
        if self.config.student.mode == "lora":
            memo = {id(p): p for p in self.student.parameters() if not p.requires_grad}
            memo.update({id(b): b for b in self.student.buffers()})
        return copy.deepcopy(self.student, memo).eval().requires_grad_(False)

    def reset_ema(self):
        self.ema = {
            view.name: {
                name: p.detach().to("cpu" if view.device == "cpu" else p.device).clone()
                for name, p in self.student.named_parameters()
                if p.requires_grad
            }
            for view in self.config.training.ema
        }

    @torch.no_grad()
    def update_ema(self):
        actual = {n: p for n, p in self.student.named_parameters() if p.requires_grad}
        for view in self.config.training.ema:
            parameters = self.ema[view.name]
            if actual.keys() != parameters.keys():
                raise ValueError("EMA parameter names changed")
            for name, value in parameters.items():
                value.mul_(view.decay).add_(
                    actual[name].to(value.device), alpha=1 - view.decay
                )

    def state_dict(self):
        return {
            "student": self.student.state_dict(),
            "reference": (
                None if self.reference is None else self.reference.state_dict()
            ),
            "ema": self.ema,
        }

    def load_state_dict(self, saved):
        frozen = {n for n, p in self.student.named_parameters() if not p.requires_grad}
        frozen.update(n for n, _ in self.student.named_buffers())
        for name, model in (
            ("student", self.student),
            ("reference", self.reference),
        ):
            if (model is None) != (saved[name] is None):
                raise ValueError("saved role presence differs")
            if model is None:
                continue
            if name == "reference":
                for key in frozen:
                    if not torch.equal(saved["student"][key], saved[name][key]):
                        raise ValueError("frozen reference base differs from student")
            model.load_state_dict(saved[name])
        if self.ema.keys() != saved["ema"].keys():
            raise ValueError("evaluation EMA names differ")
        for view, parameters in self.ema.items():
            if parameters.keys() != saved["ema"][view].keys():
                raise ValueError("EMA parameter names differ")
            for name, value in parameters.items():
                value.copy_(saved["ema"][view][name].to(value.device))

    def ownership(self):
        return {
            role: [name for name, p in model.named_parameters() if p.requires_grad]
            for role, model in (
                ("student", self.student),
                ("reference", self.reference),
            )
            if model is not None
        }
