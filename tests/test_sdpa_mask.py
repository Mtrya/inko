"""Masked attention and its gradients checked against unpadded PyTorch attention."""

import pytest
import torch
import torch.nn.functional as F

from src.models.dit_blocks import sdpa_with_pad_mask


@pytest.mark.parametrize(
    "keep",
    [
        [True] * 7,
        [True, False, True, False, False, True, False],
        [True, False, False, False, False, False, False],
    ],
)
def test_masked_attention_matches_removing_padded_keys(keep):
    torch.manual_seed(0)
    q, k, v = [torch.randn(2, 3, 7, 8, requires_grad=True) for _ in range(3)]
    mask = torch.tensor(keep)
    actual = sdpa_with_pad_mask(q, k, v, mask[None, None, None])
    expected = F.scaled_dot_product_attention(q, k[:, :, mask], v[:, :, mask])
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    for output, reference in zip((actual, *actual_grads), (expected, *expected_grads)):
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA attention check")
def test_inference_attention_preserves_distinct_padding_and_backend_state():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    generator = torch.Generator(device="cuda").manual_seed(7)
    q, k, v = [
        torch.randn(
            2, 4, 129, 72, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for _ in range(3)
    ]
    keep = (
        torch.arange(129, device="cuda")[None, :]
        < torch.tensor([97, 117], device="cuda")[:, None]
    )
    priority = torch._C._get_sdp_priority_order()
    enabled = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )
    with torch.inference_mode():
        actual = sdpa_with_pad_mask(q, k, v, keep[:, None, None, :])
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[i : i + 1].float(),
                        k[i : i + 1, :, keep[i]].float(),
                        v[i : i + 1, :, keep[i]].float(),
                    )
                    for i in range(2)
                ]
            )
    torch.testing.assert_close(actual.float(), expected, atol=3e-3, rtol=3e-2)
    assert torch._C._get_sdp_priority_order() == priority
    assert (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    ) == enabled
