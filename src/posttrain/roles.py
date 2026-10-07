"""Resident student, frozen teacher and independently updated fake-score views."""

import copy
from contextlib import nullcontext

import torch
from torch import nn

from .adapters import install_adapters, prepare_score_views, score_forward, score_view


class Velocity(nn.Module):
    def __init__(self, model, precision, *, real=False):
        super().__init__()
        self.model, self.precision = model, precision
        self.real = real
        prepare_score_views(model)

    def forward(self, x, t, conditioning, unconditional=None, guidance=1.0):
        dtype = getattr(torch, self.precision)
        with (
            score_forward(self.model, real=self.real),
            torch.autocast(x.device.type, dtype=dtype, enabled=dtype != torch.float32),
        ):
            output = self.model(x, t.float(), *conditioning).float()
            if guidance != 1:
                if unconditional is None:
                    raise ValueError("CFG requires empty-prompt conditioning")
                empty = self.model(x, t.float(), *unconditional).float()
                output = empty + guidance * (output - empty)
        return output


class Roles:
    def __init__(self, student, teacher, config, *, stage="cold"):
        self.config = config
        self.student = student.float().eval().requires_grad_(True)
        self.teacher = None
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
        self.shared_fake = False
        self.fake = None
        if config.dmd_enabled(stage):
            if teacher is None:
                raise ValueError("DMD requires its frozen teacher")
            self.teacher = teacher.float().eval().requires_grad_(False)
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
            if model is None:
                continue
            model.activation_checkpointing = config.execution.activation_checkpointing
        self.student_field = Velocity(self.student, config.execution.precision)
        self.fake_field = (
            Velocity(self.fake, config.execution.precision)
            if self.fake is not None
            else None
        )
        self.teacher_field = (
            Velocity(self.teacher, config.execution.precision, real=True)
            if self.teacher is not None
            else None
        )
        self.old = self.reference = self.old_field = self.reference_field = None
        self.reset_ema()

    def reset_ema(self):
        self.ema = (
            {
                name: p.detach().clone()
                for name, p in self.student.named_parameters()
                if p.requires_grad
            }
            if self.config.training.ema_enabled
            else None
        )

    def frozen_policy(self):
        # Frozen base parameters/buffers are shared only for LoRA. All trainable
        # adapter tensors and module objects are actual independent copies.
        memo = {}
        if self.config.student.mode == "lora":
            memo = {id(p): p for p in self.student.parameters() if not p.requires_grad}
            memo.update({id(b): b for b in self.student.buffers()})
        return copy.deepcopy(self.student, memo).eval().requires_grad_(False)

    def enter_joint(self):
        c = self.config
        if c.joint.dmd_coefficient == 0:
            self.fake = self.teacher = self.fake_field = self.teacher_field = None
            self.fake_targets = ()
            self.shared_fake = False
        if c.joint.nft_coefficient > 0:
            self.old = self.frozen_policy()
            self.old_field = Velocity(self.old, c.execution.precision)
        if c.joint.reference_coefficient > 0:
            self.reference = self.frozen_policy()
            self.reference_field = Velocity(self.reference, c.execution.precision)
        self.reset_ema()

    @torch.no_grad()
    def refresh_old(self, decay):
        if self.old is None or not 0 <= decay < 1:
            raise ValueError("old refresh requires an active policy and valid decay")
        current, old = (
            dict(self.student.named_parameters()),
            dict(self.old.named_parameters()),
        )
        if current.keys() != old.keys() or any(
            current[n].shape != old[n].shape for n in current
        ):
            raise ValueError("old/student parameter inventory differs")
        for name, parameter in current.items():
            if self.config.student.mode == "full" or parameter.requires_grad:
                if parameter is old[name]:
                    raise ValueError("old policy aliases a trainable student parameter")
                old[name].mul_(decay).add_(parameter, alpha=1 - decay)

    def state_dict(self):
        return {
            "student": self.student.state_dict(),
            "fake": None if self.fake is None else self.fake.state_dict(),
            "old": None if self.old is None else self.old.state_dict(),
            "reference": None
            if self.reference is None
            else self.reference.state_dict(),
            "ema": self.ema,
        }

    def load_state_dict(self, saved):
        frozen = {n for n, p in self.student.named_parameters() if not p.requires_grad}
        frozen.update(n for n, _ in self.student.named_buffers())
        for name, model in (
            ("student", self.student),
            ("fake", self.fake),
            ("old", self.old),
            ("reference", self.reference),
        ):
            if (model is None) != (saved[name] is None):
                raise ValueError("saved role presence differs")
            if model is None:
                continue
            if name in {"old", "reference"}:
                for key in frozen:
                    if not torch.equal(saved["student"][key], saved[name][key]):
                        raise ValueError(
                            "frozen old/reference base differs from student"
                        )
            model.load_state_dict(saved[name])
        if (self.ema is None) != (saved["ema"] is None):
            raise ValueError("evaluation EMA presence differs")
        if self.ema is not None:
            if self.ema.keys() != saved["ema"].keys():
                raise ValueError("EMA parameter names differ")
            for name, value in self.ema.items():
                value.copy_(saved["ema"][name].to(value.device))

    def real_view(self, scale=0):
        if self.teacher is None:
            raise ValueError("disabled DMD has no teacher view")
        if not self.shared_fake and scale != 0:
            raise ValueError("independent full fake cannot supply DynaDG")
        return score_view(self.teacher, scale) if self.shared_fake else nullcontext()

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
            for role, model in (
                ("student", self.student),
                ("fake", self.fake),
                ("old", self.old),
                ("reference", self.reference),
            )
            if model is not None
        }
