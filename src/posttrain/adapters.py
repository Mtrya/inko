"""Weight LoRA adapters on resolved Inko attention and MLP projections."""

import math

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


def install_adapters(model, *, rank, scale, targets="attention"):
    if targets not in {"attention", "attention_mlp"} or rank <= 0 or scale <= 0:
        raise ValueError("invalid adapter dimensions/targets")
    attention = {"qkv_img", "qkv_txt", "proj_img", "proj_txt", "qkv", "proj"}
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
        matched = pieces[2] == "attn" and pieces[3] in attention
        matched |= (
            targets == "attention_mlp"
            and pieces[2] in {"mlp", "mlp_img", "mlp_txt"}
            and pieces[3] in {"up_proj", "down_proj"}
        )
        if matched:
            parent = model.get_submodule(".".join(pieces[:-1]))
            setattr(parent, pieces[-1], WeightLoRA(layer, rank, scale))
            selected.append(name)
    if not selected:
        raise ValueError("no supported Inko adapter targets resolved")
    return tuple(selected)


def merge_student(model):
    """Materialize a detached student copy of the merged weight-LoRA weights."""
    import copy

    result = copy.deepcopy(model).cpu().float()
    for name, module in list(result.named_modules()):
        if isinstance(module, WeightLoRA):
            parent, leaf = name.rsplit(".", 1)
            setattr(result.get_submodule(parent), leaf, module.merged())
    return result.requires_grad_(False)
