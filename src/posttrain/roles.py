"""Resident student, frozen teacher and independently updated fake-score views."""

import copy
from contextlib import nullcontext

import torch
from torch import nn

from .adapters import install_adapters, score_view


class Velocity(nn.Module):
    def __init__(self, model, precision):
        super().__init__()
        self.model, self.precision = model, precision

    def forward(self, x, t, conditioning, unconditional=None, guidance=1.0):
        dtype = getattr(torch, self.precision)
        with torch.autocast(x.device.type, dtype=dtype, enabled=dtype != torch.float32):
            output = self.model(x, t.float(), *conditioning).float()
            if guidance != 1:
                if unconditional is None:
                    raise ValueError("CFG requires empty-prompt conditioning")
                empty = self.model(x, t.float(), *unconditional).float()
                output = empty + guidance * (output - empty)
        return output


class Roles:
    def __init__(self, student, teacher, config):
        self.student = student.float().eval().requires_grad_(True)
        self.teacher = teacher.float().eval().requires_grad_(False)
        self.student_targets = ()
        if config.student.mode == "lora":
            self.student_targets = install_adapters(
                self.student,
                kind="weight",
                rank=config.student.rank,
                scale=config.student.alpha,
                targets=config.student.targets,
            )
        self.fake_targets = ()
        self.shared_fake = config.fake.strategy == "teacher_qv"
        if self.shared_fake:
            self.fake = self.teacher
            self.fake_targets = install_adapters(
                self.fake,
                kind="qv",
                rank=config.fake.rank,
                scale=config.fake.multiplier,
            )
        else:
            self.fake = copy.deepcopy(self.teacher).requires_grad_(True)
        for model in (self.student, self.fake):
            model.activation_checkpointing = config.execution.activation_checkpointing
        self.student_field = Velocity(self.student, config.execution.precision)
        self.fake_field = Velocity(self.fake, config.execution.precision)
        self.teacher_field = Velocity(self.teacher, config.execution.precision)
        self.ema = (
            {
                name: p.detach().clone()
                for name, p in self.student.named_parameters()
                if p.requires_grad
            }
            if config.training.ema_enabled
            else None
        )

    def real_view(self):
        return score_view(self.teacher, 0) if self.shared_fake else nullcontext()

    @torch.no_grad()
    def update_ema(self, decay):
        if self.ema is not None:
            actual = {
                n: p for n, p in self.student.named_parameters() if p.requires_grad
            }
            if actual.keys() != self.ema.keys():
                raise ValueError("EMA parameter names changed")
            for name, value in self.ema.items():
                value.mul_(decay).add_(actual[name], alpha=1 - decay)

    def ownership(self):
        return {
            role: [name for name, p in model.named_parameters() if p.requires_grad]
            for role, model in (("student", self.student), ("fake", self.fake))
        }
