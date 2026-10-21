"""Rotary positions checked against geometric symmetry and known plane rotations."""

import math

import pytest
import torch
import torch.nn.functional as F

from src.models.dit_blocks import (
    MSRoPE,
    apply_rotary_emb,
    apply_rotary_emb_real,
    apply_rotary_emb_realfreq,
)


def test_centered_grid_conjugate_symmetry():
    rope = MSRoPE(theta=10000, axes_dim=[4, 4], centered=True)
    frequencies, _ = rope((3, 5), 4, torch.device("cpu"))
    grid = frequencies.reshape(3, 5, -1)
    torch.testing.assert_close(grid, grid.flip((0, 1)).conj())
    torch.testing.assert_close(grid[1, 2], torch.ones_like(grid[1, 2]))


def test_rotary_embedding_matches_quarter_and_half_turns():
    vector = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    frequencies = torch.tensor([[1j, -1 + 0j]])
    expected = torch.tensor([[[[-2.0, 1.0, -3.0, -4.0]]]])
    torch.testing.assert_close(apply_rotary_emb(vector, frequencies), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("restore_fp32", [False, True])
def test_factor_one_uses_original_fp32_rotations_after_dtype_cast(dtype, restore_fp32):
    rope = MSRoPE(axes_dim=[4, 4], reference_hw=(3, 5), centered=True).eval().to(dtype)
    if restore_fp32:
        rope.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        image, text = rope((3, 5), 4, "cpu")
    assert image.dtype == text.dtype == torch.complex64
    torch.testing.assert_close(
        image[0],
        torch.exp(1j * torch.tensor([-1.0, -0.01, -2.0, -0.02])),
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        text[0], torch.exp(1j * torch.tensor([5.0, 0.05, 5.0, 0.05])), rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("centered", [False, True])
def test_yarn_scales_each_axis_and_keeps_text_diagonal(centered):
    rope = MSRoPE(axes_dim=[4, 4], reference_hw=(2, 3), centered=centered).eval()
    image, text = rope((4, 9), 2, "cpu", time=(0.0,))
    tau = 1 + 0.1 * math.log(3)
    grid = image.reshape(4, 9, 4)
    # Both frequency pairs fall below the interpolation ramp at time zero.
    height_angles = torch.tensor([0.5, 0.005])
    width_angles = torch.tensor([1 / 3, 0.01 / 3])
    torch.testing.assert_close(
        (grid[1, 0] * grid[0, 0].conj())[:2] / tau,
        torch.exp(1j * height_angles),
    )
    torch.testing.assert_close(
        (grid[0, 1] * grid[0, 0].conj())[2:] / tau,
        torch.exp(1j * width_angles),
    )
    torch.testing.assert_close(
        text,
        math.sqrt(tau)
        * torch.exp(
            1j
            * torch.tensor([[9.0], [10.0]])
            * torch.cat([height_angles, width_angles])
        ),
    )


def test_training_and_smaller_requests_cannot_reuse_scaled_tables():
    rope = MSRoPE(axes_dim=[4, 4], reference_hw=(2, 3), centered=True).eval()
    scaled = rope((4, 9), 2, "cpu", time=(0.0,))
    rope.train()
    image, text = rope((4, 9), 2, "cpu")
    torch.testing.assert_close(
        image[0], torch.exp(1j * torch.tensor([-1.5, -0.015, -4.0, -0.04]))
    )
    torch.testing.assert_close(
        text[0], torch.exp(1j * torch.tensor([9.0, 0.09, 9.0, 0.09]))
    )
    rope.eval()
    for actual, expected in zip(
        rope((4, 9), 2, "cpu", time=(0.0,)), scaled, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    image, text = rope((2, 3), 1, "cpu")
    torch.testing.assert_close(
        image[0], torch.exp(1j * torch.tensor([-0.5, -0.005, -1.0, -0.01]))
    )
    torch.testing.assert_close(
        text[0], torch.exp(1j * torch.tensor([3.0, 0.03, 3.0, 0.03]))
    )


def test_dynamic_yarn_moves_frequency_bands_and_retains_temperature():
    rope = MSRoPE(
        axes_dim=[8, 8],
        reference_hw=(64, 1),
        decay_exponent=2,
        yarn_alpha=0.2,
        yarn_beta=2,
    ).eval()
    tau = 1 + 0.1 * math.log(2)
    mid_early = (3.2 / math.pi - 0.2) / 1.8
    mid_late = (0.32 / math.pi - 0.05) / 0.45
    for time, angles in (
        (0.0, [1.0, 0.05 * (1 + mid_early), 0.005, 0.0005]),
        (0.5, [1.0, 0.1, 0.005 * (1 + mid_late), 0.0005]),
        (1.0, [1.0, 0.1, 0.01, 0.001]),
        (0.0, [1.0, 0.05 * (1 + mid_early), 0.005, 0.0005]),
    ):
        image, text = rope((128, 1), 1, "cpu", time=(time,))
        torch.testing.assert_close(
            (image[1] * image[0].conj())[:4] / tau,
            torch.exp(1j * torch.tensor(angles)),
        )
        # The short width axis keeps its frequencies; global temperature
        # multiplies both axes, image and text, even when phases return to base.
        torch.testing.assert_close(image.abs().square(), torch.full((128, 8), tau))
        torch.testing.assert_close(
            text[0],
            math.sqrt(tau)
            * torch.exp(1j * 128 * torch.tensor(angles + [1.0, 0.1, 0.01, 0.001])),
            rtol=1e-5,
            atol=1e-5,
        )


def test_yarn_leaves_training_and_reference_shapes_unscaled():
    rope = MSRoPE(axes_dim=[4, 4], reference_hw=(2, 3), centered=True)
    # Training ignores dynamic inference settings and needs no timestep.
    image, text = rope((4, 9), 1, "cpu")
    torch.testing.assert_close(
        image[0], torch.exp(1j * torch.tensor([-1.5, -0.015, -4.0, -0.04]))
    )
    torch.testing.assert_close(
        text[0], torch.exp(1j * torch.tensor([9.0, 0.09, 9.0, 0.09]))
    )
    rope.eval()
    image, text = rope((2, 3), 1, "cpu")
    torch.testing.assert_close(
        image[0], torch.exp(1j * torch.tensor([-0.5, -0.005, -1.0, -0.01]))
    )
    torch.testing.assert_close(
        text[0], torch.exp(1j * torch.tensor([3.0, 0.03, 3.0, 0.03]))
    )
    with pytest.raises(ValueError, match="current flow time"):
        rope((4, 9), 1, "cpu")


@pytest.mark.parametrize(
    "rotate", [apply_rotary_emb, apply_rotary_emb_real, apply_rotary_emb_realfreq]
)
def test_rotary_operators_support_per_image_tables(rotate):
    vector = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]], [[[1.0, 2.0, 3.0, 4.0]]]])
    frequencies = torch.tensor([[[1j, -1 + 0j]], [[-1 + 0j, -1j]]])
    expected = torch.tensor([[[[-2.0, 1.0, -3.0, -4.0]]], [[[-1.0, -2.0, 4.0, -3.0]]]])
    if rotate is apply_rotary_emb_realfreq:
        frequencies = torch.view_as_real(frequencies)
    torch.testing.assert_close(rotate(vector, frequencies), expected)


@pytest.mark.parametrize(
    "rotate", [apply_rotary_emb, apply_rotary_emb_real, apply_rotary_emb_realfreq]
)
def test_yarn_amplitude_applies_logit_temperature_once(rotate):
    rope = MSRoPE(axes_dim=[2, 2], reference_hw=(1, 1)).eval()
    freqs = rope.prepare_freqs((1, 2), 0, "cpu", time=(1.0,))
    query = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]])
    value = torch.tensor([[[[2.0, -1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0]]]])
    if rotate is apply_rotary_emb_realfreq:
        freqs = torch.view_as_real(freqs)
    q = rotate(query, freqs).transpose(1, 2)
    k = rotate(key, freqs).transpose(1, 2)
    probability = 1 / (1 + math.exp(-(1 + 0.1 * math.log(2)) / 2))
    expected = probability * torch.tensor(
        [[[[2.0, -1.0, 0.0, 3.0], [2.0, -1.0, 0.0, 3.0]]]]
    )
    torch.testing.assert_close(F.scaled_dot_product_attention(q, k, value), expected)


def test_yarn_ramp_and_schedule_changes_invalidate_cache():
    rope = MSRoPE(axes_dim=[8, 8], reference_hw=(64, 1), yarn_beta=2).eval()
    for alpha, decay, middle_angle in (
        (0.2, 1, 0.0821700606),
        (0.4, 1, 0.0799413182),
        (0.4, 2, 0.0940884242),
        (0.2, 1, 0.0821700606),
    ):
        rope.yarn_alpha, rope.decay_exponent = alpha, decay
        image, _ = rope((128, 1), 1, "cpu", time=(0.25,))
        # The 0.1-frequency pair moves within the ramp at these input settings.
        torch.testing.assert_close(
            image[1, :4] / math.sqrt(1 + 0.1 * math.log(2)),
            torch.exp(1j * torch.tensor([1.0, middle_angle, 0.005, 0.0005])),
        )
