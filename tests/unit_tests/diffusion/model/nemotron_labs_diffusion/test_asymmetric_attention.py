"""Asymmetric replay must match serial block attention, including logical RoPE."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from megatron.bridge.diffusion.common.dllm import asymmetric_semi_ar_mask_mod
from megatron.bridge.diffusion.models.common import nemotron_labs_diffusion_attention as attention_module

from .test_nemotron_labs_diffusion_attention import _make_attention


pytestmark = pytest.mark.unit


def metadata(layer, device="cpu", offset=0):
    return layer.build_asymmetric_ar_metadata(
        noisy_length=12 + offset,
        clean_length=24,
        noisy_response_offset=offset,
        prompt_lengths=torch.tensor([1, 5, 9], device=device),
        response_lengths=torch.tensor([5, 8, 0], device=device),
        noisy_valid_lengths=torch.tensor([8, 8, 0], device=device),
        clean_lengths=torch.tensor([6, 13, 9], device=device),
    )


def oracle_mask(meta):
    n, c = meta.noisy_length, meta.clean_length
    mask = torch.eye(n + c, dtype=torch.bool, device=meta.prompt_lengths.device).repeat(3, 1, 1)
    for row in range(3):
        prompt = int(meta.prompt_lengths[row])
        for start in range(0, int(meta.noisy_valid_lengths[row]), 4):
            lo = meta.noisy_response_offset + start
            mask[row, lo : lo + 4, lo : lo + 4] = True
            mask[row, lo : lo + 4, n : n + prompt + start] = True
        length = int(meta.clean_lengths[row])
        mask[row, n : n + length, n : n + length] = torch.ones(
            length, length, device=mask.device, dtype=torch.bool
        ).tril()
    return mask[:, None]


@pytest.mark.parametrize("offset", [0, 2])
def test_mask_blocks_leakage_padding_and_cross_sample_attention(offset):
    layer = _make_attention(block_size=4)
    meta = metadata(layer, offset=offset)
    predicate = asymmetric_semi_ar_mask_mod(
        block_size=4,
        noisy_length=meta.noisy_length,
        noisy_response_offset=offset,
        prompt_lengths=meta.prompt_lengths,
        noisy_valid_lengths=meta.noisy_valid_lengths,
        clean_lengths=meta.clean_lengths,
    )
    actual = create_mask(predicate, 3, 1, meta.noisy_length + 24, meta.noisy_length + 24, device="cpu")
    torch.testing.assert_close(actual, oracle_mask(meta))
    assert actual.any(-1).all()  # Padded queries have a harmless self edge.


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("offset", [0, 2])
def test_forward_and_backward_match_dense_logical_position_oracle(device, offset):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise compiled flex attention")
    layer = _make_attention(num_heads=4, num_kv_heads=2, head_dim=16, block_size=4, apply_llama4=True)
    layer.config.sequence_parallel = True  # No distributed RNG needed with dropout=0.
    layer.to(device)
    meta = metadata(layer, device, offset)
    layer.set_asymmetric_ar_metadata(meta)
    torch.manual_seed(123)
    seq = meta.noisy_length + meta.clean_length
    q = torch.randn(seq, 3, 4, 16, device=device, requires_grad=True)
    k = torch.randn(seq, 3, 2, 16, device=device, requires_grad=True)
    v = torch.randn(seq, 3, 2, 16, device=device, requires_grad=True)
    dense_mask = oracle_mask(meta)

    # CPU uses SDPA for the kernel only; production RoPE/layout/mask construction
    # still execute. CUDA exercises the real compiled flex forward and backward.
    def dense_kernel(q, k, v, *, block_mask):
        assert block_mask is not None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)

    if device == "cpu":
        with patch.object(attention_module, "fused_flex_attention", dense_kernel):
            actual = layer(q, k, v)
    else:
        actual = layer(q, k, v)
    assert meta.block_mask is not None

    positions = torch.zeros(3, seq, dtype=torch.long, device=device)
    for row in range(3):
        start = int(meta.prompt_lengths[row])
        valid = int(meta.noisy_valid_lengths[row])
        positions[row, offset : offset + valid] = torch.arange(start, start + valid, device=device)
        positions[row, meta.noisy_length :] = torch.arange(meta.clean_length, device=device)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    query, key, value = [t.permute(1, 2, 0, 3) for t in (qr, kr, vr)]
    cos, sin = layer.rope_embedding_module(query, positions)
    query, key = attention_module.apply_rotary_pos_emb(query, key, cos, sin)
    scale = 1 + layer.beta * torch.log(1 + torch.floor(positions / layer.max_position_embeddings))
    query = query * scale[:, None, :, None]
    expected = (
        F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(2, dim=1),
            value.repeat_interleave(2, dim=1),
            attn_mask=dense_mask,
        )
        .permute(2, 0, 1, 3)
        .reshape_as(actual)
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    for tensor, reference in zip((q, k, v), (qr, kr, vr)):
        torch.testing.assert_close(tensor.grad, reference.grad, atol=5e-5, rtol=5e-5)
    layer.clear_asymmetric_ar_metadata()
    assert layer._asymmetric_ar_metadata is None


def test_metadata_validation_and_per_microbatch_cache():
    layer = _make_attention(block_size=4)
    first, second = metadata(layer), metadata(layer)
    layer.set_asymmetric_ar_metadata(first)
    layer.set_asymmetric_ar_metadata(second)
    assert layer._asymmetric_ar_metadata is second
    assert second.block_mask is None
    kwargs = dict(
        noisy_length=8,
        clean_length=16,
        noisy_response_offset=0,
        prompt_lengths=torch.tensor([3]),
        response_lengths=torch.tensor([5]),
        noisy_valid_lengths=torch.tensor([8]),
        clean_lengths=torch.tensor([8]),
    )
    for overrides in (
        {"noisy_valid_lengths": torch.tensor([4])},
        {"clean_lengths": torch.tensor([7])},
        {"prompt_lengths": torch.tensor([-1])},
        {"response_lengths": torch.tensor([[5]])},
        {"noisy_length": 4},
    ):
        with pytest.raises(ValueError):
            layer.build_asymmetric_ar_metadata(**(kwargs | overrides))
    other_block_size = _make_attention(block_size=8)
    with pytest.raises(ValueError, match="block size"):
        other_block_size.set_asymmetric_ar_metadata(first)
