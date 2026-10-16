"""Frozen-reference ownership and weight-LoRA installation/merge contracts."""

import copy
from dataclasses import replace

import pytest
import torch


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_role_separation_and_lora_base_sharing(
    new_roles, training_config, small_model, mode
):
    config = replace(
        training_config, student=replace(training_config.student, mode=mode)
    )
    roles = new_roles(config, small_model())
    trainable = {n for n, p in roles.student.named_parameters() if p.requires_grad}
    assert roles.reference is not None
    for name, parameter in roles.student.named_parameters():
        reference = dict(roles.reference.named_parameters())[name]
        assert not reference.requires_grad
        if name in trainable:
            assert parameter is not reference
        elif mode == "lora":
            assert parameter is reference
    assert set(roles.ownership()) == {"student", "reference"}
    without_anchor = replace(config, art=replace(config.art, reference_coefficient=0.0))
    assert new_roles(without_anchor, small_model()).reference is None


@pytest.mark.parametrize("targets,count", [("attention", 6), ("attention_mlp", 12)])
def test_student_lora_targets_are_zero_initialized_and_merge(
    targets, count, small_model
):
    from src.posttrain.adapters import WeightLoRA, install_adapters, merge_student

    torch.manual_seed(4)
    model = small_model()
    original = copy.deepcopy(model)
    names = install_adapters(model, rank=2, scale=3, targets=targets)
    assert len(names) == count
    args = (
        torch.randn(1, 16, 2, 2),
        torch.tensor([0.4]),
        torch.randn(1, 3, 1024),
        torch.randn(1, 1024),
        torch.ones(1, 3),
    )
    torch.testing.assert_close(model(*args), original(*args), rtol=0, atol=0)
    for module in model.modules():
        if isinstance(module, WeightLoRA):
            torch.nn.init.normal_(module.b.weight, std=0.03)
    torch.testing.assert_close(
        model(*args), merge_student(model)(*args), rtol=2e-5, atol=2e-6
    )
    assert all(
        ".a." in n or ".b." in n for n, p in model.named_parameters() if p.requires_grad
    )


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_one_ema_tracks_only_trainable_parameters(
    new_roles, training_config, small_model, mode
):
    config = replace(
        training_config,
        student=replace(training_config.student, mode=mode),
        training=replace(training_config.training, ema_decay=0.25),
    )
    roles = new_roles(config, small_model())
    initial = {name: value.clone() for name, value in roles.ema.items()}
    trainable = {
        name for name, p in roles.student.named_parameters() if p.requires_grad
    }
    assert roles.ema.keys() == trainable
    with torch.no_grad():
        for name, parameter in roles.student.named_parameters():
            if name in trainable:
                parameter.copy_(initial[name] + 4)
    roles.update_ema()
    for name in trainable:
        torch.testing.assert_close(roles.ema[name], initial[name] + 3)
    roles.update_ema()
    for name in trainable:
        torch.testing.assert_close(roles.ema[name], initial[name] + 3.75)
