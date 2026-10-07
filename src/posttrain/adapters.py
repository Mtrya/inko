"""Separate weight LoRA and post-projection Q/V score parameterizations."""

import math
import threading
from contextlib import contextmanager

import torch
from torch import nn


class WeightLoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.b = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.b(self.a(x))

    @torch.no_grad()
    def merged(self):
        layer = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
        ).to(self.base.weight)
        layer.weight.copy_(
            self.base.weight + self.scale * (self.b.weight @ self.a.weight)
        )
        if layer.bias is not None:
            layer.bias.copy_(self.base.bias)
        return layer


class OutputQV(nn.Module):
    def __init__(self, base, rank, multiplier):
        super().__init__()
        self.base = base.requires_grad_(False)
        width = base.out_features // 3
        if base.out_features != 3 * base.in_features:
            raise ValueError("score adapters require fused full-width QKV")
        self.qa = nn.Linear(width, rank, bias=False).to(base.weight)
        self.qb = nn.Linear(rank, width, bias=False).to(base.weight)
        self.va = nn.Linear(width, rank, bias=False).to(base.weight)
        self.vb = nn.Linear(rank, width, bias=False).to(base.weight)
        for layer in (self.qa, self.va):
            nn.init.normal_(layer.weight, mean=0, std=1)
        for layer in (self.qb, self.vb):
            nn.init.zeros_(layer.weight)
        self.multiplier, self.view_scale = multiplier, 1.0
        self.view_owner = None

    def forward(self, x):
        if self.view_owner is not None and self.view_owner != threading.get_ident():
            raise RuntimeError(
                "score forward overlaps a different worker thread's view"
            )
        q, k, v = self.base(x).chunk(3, dim=-1)
        if self.view_scale != 0:
            scale = self.multiplier * self.view_scale
            q = q + scale * self.qb(self.qa(q))
            v = v + scale * self.vb(self.va(v))
        return torch.cat((q, k, v), dim=-1)


def install_adapters(model, *, kind, rank, scale, targets="attention"):
    if kind not in {"weight", "qv"} or targets not in {"attention", "attention_mlp"}:
        raise ValueError("unsupported adapter kind/targets")
    if rank <= 0 or scale <= 0 or (kind == "qv" and targets != "attention"):
        raise ValueError("invalid adapter dimensions/targets")
    attention = {"qkv_img", "qkv_txt", "proj_img", "proj_txt", "qkv", "proj"}
    qkv = {"qkv_img", "qkv_txt", "qkv"}
    selected = []
    model.requires_grad_(False)
    for name, layer in list(model.named_modules()):
        pieces = name.split(".")
        if (
            len(pieces) != 4
            or pieces[0] != "blocks"
            or not isinstance(layer, nn.Linear)
        ):
            continue
        matched = pieces[2] == "attn" and pieces[3] in (
            attention if kind == "weight" else qkv
        )
        matched |= (
            kind == "weight"
            and targets == "attention_mlp"
            and pieces[2] in {"mlp", "mlp_img", "mlp_txt"}
            and pieces[3] in {"up_proj", "down_proj"}
        )
        if matched:
            parent = model.get_submodule(".".join(pieces[:-1]))
            setattr(
                parent,
                pieces[-1],
                (WeightLoRA if kind == "weight" else OutputQV)(layer, rank, scale),
            )
            selected.append(name)
    if not selected:
        raise ValueError("no supported Inko adapter targets resolved")
    return tuple(selected)


def prepare_score_views(model):
    """Only shared Q/V models need a lock; student exports remain copyable."""
    if any(isinstance(m, OutputQV) for m in model.modules()):
        model.__dict__.setdefault("_score_forward_lock", threading.RLock())


@contextmanager
def score_forward(model, *, real=False):
    """Hold view admission through the complete conditional/CFG forward."""
    lock = model.__dict__.get("_score_forward_lock")
    if lock is None:
        yield
        return
    if not lock.acquire(blocking=False):
        raise RuntimeError("overlapping score forwards/views are forbidden")
    try:
        active = model.__dict__.get("_score_view_owner") is not None
        if real != active:
            raise RuntimeError("score forward role does not match the active view")
        yield
    finally:
        lock.release()


@contextmanager
def score_view(model, scale):
    adapters = [m for m in model.modules() if isinstance(m, OutputQV)]
    if not adapters:
        raise ValueError("score views require shared Q/V adapters")
    prepare_score_views(model)
    lock = model.__dict__["_score_forward_lock"]
    if not lock.acquire(blocking=False):
        raise RuntimeError("overlapping score views are forbidden")
    if model.__dict__.get("_score_view_owner") is not None:
        lock.release()
        raise RuntimeError("overlapping score views are forbidden")
    previous = [(m.view_scale, m.view_owner) for m in adapters]
    try:
        model.__dict__["_score_view_owner"] = threading.get_ident()
        for m in adapters:
            m.view_scale = scale
            m.view_owner = threading.get_ident()
        yield model
    finally:
        for m, (old, owner) in zip(adapters, previous, strict=True):
            m.view_scale, m.view_owner = old, owner
        model.__dict__["_score_view_owner"] = None
        lock.release()


def merge_student(model):
    """Materialize a detached student copy; score adapters can never be exported."""
    import copy

    if any(isinstance(m, OutputQV) for m in model.modules()):
        raise ValueError("fake-score adapters are not student export weights")
    result = copy.deepcopy(model).cpu().float()
    for name, module in list(result.named_modules()):
        if isinstance(module, WeightLoRA):
            parent, leaf = name.rsplit(".", 1)
            setattr(result.get_submodule(parent), leaf, module.merged())
    return result.requires_grad_(False)
