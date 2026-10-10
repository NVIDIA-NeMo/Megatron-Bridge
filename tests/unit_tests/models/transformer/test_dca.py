# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest
import torch
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_concentration_factor
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.transformer.dca import (
    HAVE_FLASH_ATTN,
    DualChunkAttention,
    DualChunkTransformerConfig,
    _extend_rotary_frequencies,
    _merge_chunk_attention_outputs,
    validate_dual_chunk_transformer_config,
)


pytestmark = pytest.mark.unit


def _make_config(**overrides: object) -> DualChunkTransformerConfig:
    defaults = {
        "num_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "attention_dropout": 0.0,
        "apply_rope_fusion": False,
        "use_cpu_initialization": True,
        "transformer_impl": "local",
        "dca_chunk_size": 8,
        "dca_local_size": 2,
    }
    defaults.update(overrides)
    return DualChunkTransformerConfig(**defaults)


def _make_dca(config: DualChunkTransformerConfig) -> DualChunkAttention:
    pg_collection = ProcessGroupCollection(cp=object())
    return DualChunkAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=pg_collection,
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
    config: DualChunkTransformerConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    seq_len, batch_size, num_heads, head_dim = query.shape
    q_pos_emb, k_pos_emb = rotary_pos_emb
    cp_group = object()
    query = apply_rotary_pos_emb(query, q_pos_emb[:seq_len], config=config, cp_group=cp_group)
    key = apply_rotary_pos_emb(key, k_pos_emb[:seq_len], config=config, cp_group=cp_group)
    if key.size(2) < num_heads:
        repeat_factor = num_heads // key.size(2)
        key = key.repeat_interleave(repeat_factor, dim=2)
        value = value.repeat_interleave(repeat_factor, dim=2)

    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3)
    v = value.permute(1, 2, 0, 3)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=scores.dtype, device=scores.device),
        diagonal=1,
    )
    probabilities = torch.softmax(scores + mask.unsqueeze(0).unsqueeze(0), dim=-1)
    output = torch.matmul(probabilities, v)
    return output.permute(2, 0, 1, 3).contiguous().reshape(seq_len, batch_size, num_heads * head_dim)


def _partial_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query, key.transpose(-2, -1))
    return torch.matmul(torch.softmax(scores, dim=-1), value), torch.logsumexp(scores, dim=-1, keepdim=True)


def _direct_dca_attention(
    config: DualChunkTransformerConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    seq_len, batch_size, num_heads, head_dim = query.shape
    chunk_len = config.dca_chunk_size - config.dca_local_size
    positions = torch.arange(seq_len, device=query.device)
    local_positions = positions % chunk_len
    q_pos_emb, k_pos_emb = rotary_pos_emb
    cp_group = object()
    key = apply_rotary_pos_emb(key, k_pos_emb[local_positions], config=config, cp_group=cp_group)
    query_intra = apply_rotary_pos_emb(query, q_pos_emb[local_positions], config=config, cp_group=cp_group)
    query_successor = apply_rotary_pos_emb(
        query,
        q_pos_emb[(local_positions + chunk_len).clamp(max=config.dca_chunk_size)],
        config=config,
        cp_group=cp_group,
    )
    inter_position = min(2 * chunk_len - 1, config.dca_chunk_size)
    query_inter = apply_rotary_pos_emb(
        query,
        q_pos_emb[inter_position : inter_position + 1].expand(seq_len, -1, -1, -1),
        config=config,
        cp_group=cp_group,
    )

    key = key.permute(1, 2, 0, 3)
    value = value.permute(1, 2, 0, 3)
    output_rows = []
    for query_position in range(seq_len):
        chunk_index = query_position // chunk_len
        chunk_start = chunk_index * chunk_len
        score_parts = []
        value_parts = []

        if chunk_index >= 2:
            inter_end = (chunk_index - 1) * chunk_len
            inter_query = query_inter[query_position].unsqueeze(-2)
            score_parts.append(torch.matmul(inter_query, key[:, :, :inter_end].transpose(-2, -1)))
            value_parts.append(value[:, :, :inter_end])
        if chunk_index >= 1:
            previous_start = (chunk_index - 1) * chunk_len
            successor_query = query_successor[query_position].unsqueeze(-2)
            score_parts.append(torch.matmul(successor_query, key[:, :, previous_start:chunk_start].transpose(-2, -1)))
            value_parts.append(value[:, :, previous_start:chunk_start])

        intra_query = query_intra[query_position].unsqueeze(-2)
        score_parts.append(torch.matmul(intra_query, key[:, :, chunk_start : query_position + 1].transpose(-2, -1)))
        value_parts.append(value[:, :, chunk_start : query_position + 1])

        scores = torch.cat(score_parts, dim=-1) * (head_dim**-0.5)
        attended_values = torch.cat(value_parts, dim=-2)
        output_rows.append(torch.matmul(torch.softmax(scores, dim=-1), attended_values).permute(2, 0, 1, 3))

    return torch.cat(output_rows, dim=0).reshape(seq_len, batch_size, num_heads * head_dim)


def test_merge_chunk_attention_outputs_single_output_passthrough() -> None:
    output = torch.randn(2, 4, 8, 16)
    lse = torch.randn(2, 4, 8, 1)
    assert torch.equal(_merge_chunk_attention_outputs([output], [lse]), output)


def test_merge_chunk_attention_outputs_matches_direct_output_and_gradients() -> None:
    torch.manual_seed(1234)
    shape = (1, 2, 3, 4)
    query = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    key_a = torch.randn((1, 2, 2, 4), dtype=torch.float64, requires_grad=True)
    key_b = torch.randn((1, 2, 5, 4), dtype=torch.float64, requires_grad=True)
    value_a = torch.randn((1, 2, 2, 4), dtype=torch.float64, requires_grad=True)
    value_b = torch.randn((1, 2, 5, 4), dtype=torch.float64, requires_grad=True)

    output_a, lse_a = _partial_attention(query, key_a, value_a)
    output_b, lse_b = _partial_attention(query, key_b, value_b)
    merged = _merge_chunk_attention_outputs([output_a, output_b], [lse_a, lse_b])

    direct_key = torch.cat((key_a, key_b), dim=-2)
    direct_value = torch.cat((value_a, value_b), dim=-2)
    direct, _ = _partial_attention(query, direct_key, direct_value)
    torch.testing.assert_close(merged, direct, atol=1e-10, rtol=1e-10)

    output_gradient = torch.randn_like(merged)
    inputs = (query, key_a, key_b, value_a, value_b)
    merged_gradients = torch.autograd.grad(merged, inputs, output_gradient, retain_graph=True)
    direct_gradients = torch.autograd.grad(direct, inputs, output_gradient)
    for merged_gradient, direct_gradient in zip(merged_gradients, direct_gradients):
        torch.testing.assert_close(merged_gradient, direct_gradient, atol=1e-10, rtol=1e-10)


def test_merge_chunk_attention_outputs_preserves_low_precision_dtype() -> None:
    output = torch.randn(1, 2, 3, 8, dtype=torch.float16)
    lse = torch.randn(1, 2, 3, 1, dtype=torch.float32)
    merged = _merge_chunk_attention_outputs([output, output], [lse, lse])
    assert merged.dtype == output.dtype


def test_extend_rotary_frequencies_preserves_linear_positions() -> None:
    rotary, _ = _make_rotary_pos_emb(7, 8)
    extended = _extend_rotary_frequencies(rotary, 10)
    expected, _ = _make_rotary_pos_emb(10, 8)
    torch.testing.assert_close(extended, expected)


def test_yarn_concentration_factor_comes_from_transformer_config() -> None:
    config = _make_config(
        yarn_rotary_scaling_factor=4.0,
        yarn_mscale=1.0,
        yarn_mscale_all_dim=0.0,
    )
    dca = _make_dca(config)

    expected = _yarn_get_concentration_factor(4.0, 1.0, 0.0)
    assert dca.mscale == pytest.approx(expected)


def test_short_sequence_matches_standard_causal_attention() -> None:
    config = _make_config(dca_chunk_size=16, dca_local_size=4, num_attention_heads=2, num_query_groups=2)
    dca = _make_dca(config)
    seq_len, batch_size, head_dim = 8, 2, 16
    torch.manual_seed(1234)
    query = torch.randn(seq_len, batch_size, 2, head_dim)
    key = torch.randn(seq_len, batch_size, 2, head_dim)
    value = torch.randn(seq_len, batch_size, 2, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, None, rotary_pos_emb=rotary)
    expected = _standard_attention(config, query, key, value, rotary)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_long_sequence_runs_on_cpu_and_backpropagates() -> None:
    config = _make_config()
    dca = _make_dca(config)
    seq_len, batch_size, num_heads, head_dim = 18, 2, 4, 8
    torch.manual_seed(1234)
    query = torch.randn(seq_len, batch_size, num_heads, head_dim, requires_grad=True)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim, requires_grad=True)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim, requires_grad=True)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, None, rotary_pos_emb=rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)

    output.square().mean().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


def test_long_sequence_matches_direct_dca_output_and_gradients() -> None:
    config = _make_config(hidden_size=8, num_attention_heads=2, num_query_groups=2)
    dca = _make_dca(config)
    shape = (14, 1, 2, 4)
    torch.manual_seed(1234)
    dca_inputs = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    direct_inputs = [tensor.detach().clone().requires_grad_() for tensor in dca_inputs]
    rotary = _make_rotary_pos_emb(shape[0], shape[-1])

    output = dca(*dca_inputs, None, rotary_pos_emb=rotary)
    expected = _direct_dca_attention(config, *direct_inputs, rotary)
    torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-10)

    output_gradient = torch.randn_like(output)
    gradients = torch.autograd.grad(output, dca_inputs, output_gradient)
    expected_gradients = torch.autograd.grad(expected, direct_inputs, output_gradient)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient, atol=1e-10, rtol=1e-10)


def test_long_sequence_runs_across_non_divisible_chunks() -> None:
    config = _make_config()
    dca = _make_dca(config)
    seq_len, batch_size, num_heads, head_dim = 23, 2, 4, 8
    query = torch.randn(seq_len, batch_size, num_heads, head_dim)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, None, rotary_pos_emb=rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)


def test_prefix_outputs_do_not_depend_on_future_tokens() -> None:
    config = _make_config()
    dca = _make_dca(config)
    seq_len, prefix_len, batch_size, num_heads, head_dim = 18, 10, 1, 4, 8
    torch.manual_seed(1234)
    query = torch.randn(seq_len, batch_size, num_heads, head_dim)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, None, rotary_pos_emb=rotary)
    future_key = key.clone()
    future_value = value.clone()
    future_key[prefix_len:] = torch.randn_like(future_key[prefix_len:])
    future_value[prefix_len:] = torch.randn_like(future_value[prefix_len:])
    future_output = dca(query, future_key, future_value, None, rotary_pos_emb=rotary)
    torch.testing.assert_close(output[:prefix_len], future_output[:prefix_len], atol=1e-5, rtol=1e-5)


def test_gqa_runs_on_cpu() -> None:
    config = _make_config(num_query_groups=2)
    dca = _make_dca(config)
    seq_len, batch_size, head_dim = 18, 2, 8
    query = torch.randn(seq_len, batch_size, 4, head_dim)
    key = torch.randn(seq_len, batch_size, 2, head_dim)
    value = torch.randn(seq_len, batch_size, 2, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, None, rotary_pos_emb=rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)


def test_boundary_sequence_extends_short_rotary_table() -> None:
    config = _make_config()
    dca = _make_dca(config)
    seq_len, batch_size, num_heads, head_dim = 7, 1, 4, 8
    query = torch.randn(seq_len, batch_size, num_heads, head_dim)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim)
    value = torch.randn(seq_len, batch_size, num_heads, head_dim)
    rotary = _make_rotary_pos_emb(seq_len, head_dim)

    output = dca(query, key, value, None, rotary_pos_emb=rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)


def test_forward_rejects_unsupported_attention_inputs() -> None:
    config = _make_config()
    dca = _make_dca(config)
    query = torch.randn(4, 1, 4, 8)
    key = torch.randn(4, 1, 4, 8)
    value = torch.randn(4, 1, 4, 8)

    with pytest.raises(ValueError, match="explicit attention masks"):
        dca(query, key, value, torch.ones(1), rotary_pos_emb=_make_rotary_pos_emb(4, 8))
    with pytest.raises(ValueError, match="packed_seq_params"):
        dca(query, key, value, None, packed_seq_params=object())
    with pytest.raises(ValueError, match="causal"):
        dca(query, key, value, None, attn_mask_type=AttnMaskType.padding)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dca_chunk_size": 4, "dca_local_size": 4}, "dca_chunk_size must be greater"),
        ({"context_parallel_size": 2}, "context_parallel_size"),
        ({"apply_rope_fusion": True}, "apply_rope_fusion"),
        ({"fused_single_qkv_rope": True}, "fused_single_qkv_rope"),
        ({"attention_output_gate": True}, "attention_output_gate"),
        ({"fine_grained_activation_offloading": True}, "fine_grained_activation_offloading"),
        ({"cuda_graph_impl": "full_iteration"}, "CUDA graphs"),
        ({"mtp_num_layers": 1}, "MTP"),
        ({"no_rope_freq": [0, 1]}, "disabling RoPE"),
    ],
)
def test_transformer_config_validation_rejects_unsupported_modes(
    override: dict[str, object],
    message: str,
) -> None:
    config = _make_config(**override)
    with pytest.raises(ValueError, match=message):
        validate_dual_chunk_transformer_config(config)


def test_selective_core_attention_recompute_is_supported() -> None:
    config = _make_config(recompute_granularity="selective", recompute_modules=["core_attn"])
    validate_dual_chunk_transformer_config(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU DCA coverage.")
def test_long_sequence_runs_on_cuda() -> None:
    config = _make_config()
    dca = _make_dca(config).cuda()
    seq_len, batch_size, num_heads, head_dim = 18, 2, 4, 8
    query = torch.randn(
        seq_len,
        batch_size,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    rotary = _make_rotary_pos_emb(seq_len, head_dim, device=query.device)

    output = dca(query, key, value, None, rotary_pos_emb=rotary)
    assert output.shape == (seq_len, batch_size, config.hidden_size)
    output.float().square().mean().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_FLASH_ATTN,
    reason="CUDA and flash-attn are required for FlashAttention DCA coverage.",
)
@pytest.mark.parametrize("num_query_groups", [4, 2])
def test_flash_attention_matches_unfused_output_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
    num_query_groups: int,
) -> None:
    config = _make_config(num_query_groups=num_query_groups)
    dca_flash = _make_dca(config).cuda().eval()
    dca_unfused = _make_dca(config).cuda().eval()
    monkeypatch.setattr(dca_unfused, "_use_flash_attention", lambda query: False)

    query_shape = (18, 2, 4, 8)
    key_value_shape = (18, 2, num_query_groups, 8)
    torch.manual_seed(1234)
    flash_inputs = [
        torch.randn(query_shape, device="cuda", dtype=torch.float16, requires_grad=True),
        torch.randn(key_value_shape, device="cuda", dtype=torch.float16, requires_grad=True),
        torch.randn(key_value_shape, device="cuda", dtype=torch.float16, requires_grad=True),
    ]
    unfused_inputs = [tensor.detach().clone().requires_grad_() for tensor in flash_inputs]
    rotary = _make_rotary_pos_emb(query_shape[0], query_shape[-1], device=flash_inputs[0].device)

    flash_output = dca_flash(*flash_inputs, None, rotary_pos_emb=rotary)
    unfused_output = dca_unfused(*unfused_inputs, None, rotary_pos_emb=rotary)
    torch.testing.assert_close(flash_output.float(), unfused_output.float(), atol=3e-2, rtol=3e-2)

    output_gradient = torch.randn_like(flash_output)
    flash_gradients = torch.autograd.grad(flash_output, flash_inputs, output_gradient)
    unfused_gradients = torch.autograd.grad(unfused_output, unfused_inputs, output_gradient)
    for flash_gradient, unfused_gradient in zip(flash_gradients, unfused_gradients):
        torch.testing.assert_close(flash_gradient.float(), unfused_gradient.float(), atol=5e-2, rtol=5e-2)
