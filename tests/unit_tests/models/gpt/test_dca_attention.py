# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gpt.dca_attention import (
    HAVE_FLASH_ATTN,
    DualChunkAttention,
    _merge_chunk_attention_outputs,
)
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit


def _make_config(
    *,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    num_query_groups: int = 4,
) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        apply_rope_fusion=False,
    )


def _make_dca(
    config: TransformerConfig,
    *,
    dca_chunk_size: int = 16,
    dca_local_size: int = 4,
) -> DualChunkAttention:
    return DualChunkAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        dca_chunk_size=dca_chunk_size,
        dca_local_size=dca_local_size,
    )


def _make_rotary_pos_emb(
    seq_len: int,
    head_dim: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(1).unsqueeze(1)
    return emb, emb


def _standard_attention(
    config: TransformerConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    seq_len, batch_size, num_heads, head_dim = query.shape
    q_pos_emb, k_pos_emb = rotary_pos_emb
    query = apply_rotary_pos_emb(query, q_pos_emb[:seq_len], config=config)
    key = apply_rotary_pos_emb(key, k_pos_emb[:seq_len], config=config)
    if key.size(2) < num_heads:
        repeat_factor = num_heads // key.size(2)
        key = key.repeat_interleave(repeat_factor, dim=2)
        value = value.repeat_interleave(repeat_factor, dim=2)

    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3)
    v = value.permute(1, 2, 0, 3)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    scores = scores + mask.unsqueeze(0).unsqueeze(0)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probs, v)
    return output.permute(2, 0, 1, 3).contiguous().reshape(seq_len, batch_size, num_heads * head_dim)


def test_merge_chunk_attention_outputs_single_output_passthrough() -> None:
    output = torch.randn(2, 4, 8, 16)
    lse = torch.randn(2, 4, 8, 1)
    assert torch.equal(_merge_chunk_attention_outputs([output], [lse]), output)


def test_merge_chunk_attention_outputs_uses_lse_weights() -> None:
    out1 = torch.ones(1, 1, 4, 8)
    out2 = torch.zeros(1, 1, 4, 8)
    lse = torch.zeros(1, 1, 4, 1)
    expected = torch.full_like(out1, 0.5)
    assert torch.allclose(_merge_chunk_attention_outputs([out1, out2], [lse, lse]), expected)


def test_short_sequence_matches_standard_causal_attention() -> None:
    config = _make_config(hidden_size=32, num_attention_heads=2, num_query_groups=2)
    dca = _make_dca(config, dca_chunk_size=16, dca_local_size=4)
    seq_len, batch_size, head_dim = 8, 2, 16
    torch.manual_seed(1234)
    query = torch.randn(seq_len, batch_size, 2, head_dim)
    key = torch.randn(seq_len, batch_size, 2, head_dim)
    value = torch.randn(seq_len, batch_size, 2, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
    expected = _standard_attention(config, query, key, value, rotary)

    assert torch.allclose(output, expected, atol=1e-5)


def test_long_sequence_runs_on_cpu_and_backpropagates() -> None:
    config = _make_config()
    dca = _make_dca(config, dca_chunk_size=8, dca_local_size=2)
    seq_len, batch_size, num_heads, head_dim = 18, 2, 4, 8
    torch.manual_seed(1234)
    query = torch.randn(seq_len, batch_size, num_heads, head_dim, requires_grad=True)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim, requires_grad=True)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim, requires_grad=True)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)

    output.sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


def test_long_sequence_runs_across_multiple_non_divisible_chunks() -> None:
    config = _make_config()
    dca = _make_dca(config, dca_chunk_size=8, dca_local_size=2)
    seq_len, batch_size, num_heads, head_dim = 23, 2, 4, 8
    query = torch.randn(seq_len, batch_size, num_heads, head_dim)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)

    assert output.shape == (seq_len, batch_size, config.hidden_size)


def test_prefix_outputs_do_not_depend_on_future_tokens() -> None:
    config = _make_config()
    dca = _make_dca(config, dca_chunk_size=8, dca_local_size=2)
    seq_len, prefix_len, batch_size, num_heads, head_dim = 18, 10, 1, 4, 8
    torch.manual_seed(1234)
    query = torch.randn(seq_len, batch_size, num_heads, head_dim)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
    future_key = key.clone()
    future_value = value.clone()
    future_key[prefix_len:] = torch.randn_like(future_key[prefix_len:])
    future_value[prefix_len:] = torch.randn_like(future_value[prefix_len:])
    future_output = dca(query, future_key, future_value, attention_mask=None, rotary_pos_emb=rotary)

    assert torch.allclose(output[:prefix_len], future_output[:prefix_len], atol=1e-5)


def test_gqa_runs_on_cpu() -> None:
    config = _make_config(hidden_size=32, num_attention_heads=4, num_query_groups=2)
    dca = _make_dca(config, dca_chunk_size=8, dca_local_size=2)
    seq_len, batch_size, head_dim = 18, 2, 8
    query = torch.randn(seq_len, batch_size, 4, head_dim)
    key = torch.randn(seq_len, batch_size, 2, head_dim)
    value = torch.randn(seq_len, batch_size, 2, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)

    assert output.shape == (seq_len, batch_size, config.hidden_size)


def test_forward_rejects_unsupported_attention_inputs() -> None:
    config = _make_config()
    dca = _make_dca(config)
    seq_len, batch_size, num_heads, head_dim = 4, 1, 4, 8
    query = torch.randn(seq_len, batch_size, num_heads, head_dim)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim)

    with pytest.raises(ValueError, match="packed_seq_params"):
        dca(query, key, value, attention_mask=None, packed_seq_params=object())
    with pytest.raises(ValueError, match="causal"):
        dca(query, key, value, attention_mask=None, attn_mask_type=AttnMaskType.padding)


def test_boundary_sequence_above_chunk_len_requires_extended_rotary_table() -> None:
    config = _make_config()
    dca = _make_dca(config, dca_chunk_size=8, dca_local_size=2)
    seq_len, batch_size, num_heads, head_dim = 7, 1, 4, 8
    query = torch.randn(seq_len, batch_size, num_heads, head_dim)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim)

    too_short_rotary = _make_rotary_pos_emb(seq_len, head_dim)
    with pytest.raises(ValueError, match="rotary_pos_emb is too short"):
        dca(query, key, value, attention_mask=None, rotary_pos_emb=too_short_rotary)

    extended_rotary = _make_rotary_pos_emb(9, head_dim)
    output = dca(query, key, value, attention_mask=None, rotary_pos_emb=extended_rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU DCA coverage.")
def test_long_sequence_runs_on_cuda() -> None:
    config = _make_config()
    dca = _make_dca(config, dca_chunk_size=8, dca_local_size=2).cuda()
    seq_len, batch_size, num_heads, head_dim = 18, 2, 4, 8
    dtype = torch.float16
    query = torch.randn(seq_len, batch_size, num_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    rotary = _make_rotary_pos_emb(seq_len, head_dim, device=query.device)

    output = dca(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)
    assert output.is_cuda

    output.float().sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_FLASH_ATTN,
    reason="CUDA and flash-attn are required for FlashAttention DCA coverage.",
)
def test_flash_attention_path_matches_unfused_path(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _make_config()
    dca_flash = _make_dca(config, dca_chunk_size=8, dca_local_size=2).cuda()
    dca_unfused = _make_dca(config, dca_chunk_size=8, dca_local_size=2).cuda()
    dca_flash.eval()
    dca_unfused.eval()
    monkeypatch.setattr(dca_unfused, "_use_flash_attention", lambda query: False)

    seq_len, batch_size, num_heads, head_dim = 18, 2, 4, 8
    dtype = torch.float16
    torch.manual_seed(1234)
    query = torch.randn(seq_len, batch_size, num_heads, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim, device="cuda", dtype=dtype)
    rotary = _make_rotary_pos_emb(seq_len, head_dim, device=query.device)

    flash_output = dca_flash(query, key, value, attention_mask=None, rotary_pos_emb=rotary)
    unfused_output = dca_unfused(query, key, value, attention_mask=None, rotary_pos_emb=rotary)

    assert torch.allclose(flash_output.float(), unfused_output.float(), atol=3e-2, rtol=3e-2)
