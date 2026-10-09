# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal packing and CP position tests with the native Nemotron RoPE config."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import (
    NemotronLabsDiffusionAttention,
)
from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
    NemotronLabsDiffusionModelProvider,
)


pytestmark = [pytest.mark.unit]


def make_attention(cp_size: int = 1) -> NemotronLabsDiffusionAttention:
    config = NemotronLabsDiffusionModelProvider(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=8,
        seq_length=32,
        context_parallel_size=cp_size,
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        apply_llama4_style_query_key_layer_scaling=True,
        llama4_scaling_beta=0.1,
        yarn_original_max_position_embeddings=8,
        attention_dropout=0.0,
        rotary_base=10000.0,
    )
    pg = MagicMock()
    pg.tp.size.return_value = 1
    attention = NemotronLabsDiffusionAttention(config, 1, AttnMaskType.causal, "self", pg_collection=pg)
    attention.set_inference_mode(True)
    attention.set_inference_params(causal=True, cache_enabled=False)
    return attention


def packed_params(lengths: list[int], padded: list[int] | None = None) -> PackedSeqParams:
    cu = torch.tensor([0] + lengths).cumsum(0).to(torch.int32)
    padded_cu = None if padded is None else torch.tensor([0] + padded).cumsum(0).to(torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=padded_cu,
        cu_seqlens_kv_padded=padded_cu,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_padded_positions_reset_and_match_zigzag(cp_size: int) -> None:
    params = packed_params([5, 11], padded=[8, 16])
    full = torch.cat([torch.arange(8), torch.arange(16)])
    for rank in range(cp_size):
        group = MagicMock()
        group.size.return_value = cp_size
        group.rank.return_value = rank
        expected = []
        offset = 0
        for length in [8, 16]:
            chunks = full[offset : offset + length].chunk(2 * cp_size)
            expected += [chunks[rank], chunks[2 * cp_size - rank - 1]]
            offset += length
        actual = NemotronLabsDiffusionAttention._packed_position_ids(params, 24 // cp_size, torch.device("cpu"), group)
        torch.testing.assert_close(actual.squeeze(0), torch.cat(expected))


def test_invalid_cp_padding_is_rejected() -> None:
    group = MagicMock()
    group.size.return_value = 2
    group.rank.return_value = 0
    with pytest.raises(ValueError, match="divisible"):
        NemotronLabsDiffusionAttention._packed_position_ids(packed_params([6]), 3, torch.device("cpu"), group)


class ReferencePackedCore(torch.nn.Module):
    """Evaluate independent causal sequences as a CPU reference."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        attn_mask_type: AttnMaskType,
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        """Apply causal SDPA separately within each pack."""
        assert attention_mask is None
        assert attn_mask_type == AttnMaskType.causal
        outputs = []
        cu = packed_seq_params.cu_seqlens_q.tolist()
        for start, end in zip(cu[:-1], cu[1:]):
            q_part = q[start:end].transpose(0, 1).unsqueeze(0)
            k_part = k[start:end].transpose(0, 1).unsqueeze(0).repeat_interleave(2, dim=1)
            v_part = v[start:end].transpose(0, 1).unsqueeze(0).repeat_interleave(2, dim=1)
            out = F.scaled_dot_product_attention(q_part, k_part, v_part, is_causal=True)
            outputs.append(out.squeeze(0).transpose(0, 1).flatten(1))
        return torch.cat(outputs)


def test_packed_outputs_and_gradients_match_separate_ar_sequences() -> None:
    torch.manual_seed(42)
    attention = make_attention()
    params = packed_params([5, 11])
    q = torch.randn(16, 1, 4, 8, requires_grad=True)
    k = torch.randn(16, 1, 2, 8, requires_grad=True)
    v = torch.randn(16, 1, 2, 8, requires_grad=True)
    with patch(
        "megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention.TEDotProductAttention",
        return_value=ReferencePackedCore(),
    ) as ctor:
        packed = attention(q, k, v, packed_seq_params=params)
        assert ctor.call_args.kwargs["config"] is attention.config
        assert attention._ar_core_attention is not None
    separate = torch.cat([attention(q[:5], k[:5], v[:5]), attention(q[5:], k[5:], v[5:])])
    torch.testing.assert_close(packed, separate, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(packed)
    packed_grads = torch.autograd.grad((packed * probe).sum(), (q, k, v), retain_graph=True)
    separate_grads = torch.autograd.grad((separate * probe).sum(), (q, k, v))
    for actual, expected in zip(packed_grads, separate_grads):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_cp_requires_packing_and_causal_cacheless_ar() -> None:
    attention = make_attention(cp_size=2)
    q = torch.randn(8, 1, 4, 8)
    k = torch.randn(8, 1, 2, 8)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="requires packed"):
        attention(q, k, v)
    attention.set_inference_params(causal=False, cache_enabled=False)
    with pytest.raises(ValueError, match="causal=True"):
        attention(q, k, v, packed_seq_params=packed_params([8]))
    attention.set_inference_params(causal=True, cache_enabled=True)
    with pytest.raises(ValueError, match="cache_enabled=False"):
        attention(q, k, v, packed_seq_params=packed_params([8]))
    attention.set_inference_mode(False)
    with pytest.raises(ValueError, match="packed causal AR"):
        attention(q, k, v)
