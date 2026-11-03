# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-GPU native-RoPE causal CP parity and 128K attention memory smoke.

Run with uv run python -m torch.distributed.run --nproc_per_node=2 this_file.py.
This tests the attention core, not full-model 128K training memory.
"""

import json
import logging
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import (
    NemotronLabsDiffusionAttention,
)
from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
    NemotronLabsDiffusionModelProvider,
)


logger = logging.getLogger(__name__)


def make_attention(cp_size: int) -> NemotronLabsDiffusionAttention:
    config = NemotronLabsDiffusionModelProvider(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=64,
        seq_length=131072,
        context_parallel_size=cp_size,
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        apply_llama4_style_query_key_layer_scaling=True,
        llama4_scaling_beta=0.1,
        rope_type="yarn",
        yarn_rotary_scaling_factor=8.0,
        yarn_original_max_position_embeddings=16384,
        attention_dropout=0.0,
        rotary_base=10000.0,
    )
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    attention = NemotronLabsDiffusionAttention(config, 1, AttnMaskType.causal, "self", pg_collection=pg)
    attention.to(device="cuda", dtype=torch.bfloat16)
    attention.set_inference_mode(True)
    attention.set_inference_params(causal=True, cache_enabled=False)
    return attention


def local_indices(padded: list[int], rank: int) -> torch.Tensor:
    indices = []
    offset = 0
    for length in padded:
        segment = length // 4
        indices.extend(range(offset + rank * segment, offset + (rank + 1) * segment))
        back = 3 - rank
        indices.extend(range(offset + back * segment, offset + (back + 1) * segment))
        offset += length
    return torch.tensor(indices, device="cuda")


def params_for(lengths: list[int], padded: list[int]) -> PackedSeqParams:
    cu_pad = torch.tensor([0] + padded, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        # Match the standard NeMo-RL packed CP loader: right padding is part
        # of each causal sequence, and its loss is masked out.
        cu_seqlens_q=cu_pad,
        cu_seqlens_kv=cu_pad,
        cu_seqlens_q_padded=cu_pad,
        cu_seqlens_kv_padded=cu_pad,
        max_seqlen_q=max(padded),
        max_seqlen_kv=max(padded),
        pad_between_seqs=False,
        cp_group=parallel_state.get_context_parallel_group(),
    )


def check_parity(attention: NemotronLabsDiffusionAttention, lengths: list[int], padded: list[int]) -> None:
    rank = dist.get_rank()
    torch.manual_seed(42)
    total = sum(padded)
    full = [
        torch.randn(total, heads, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for heads in [4, 2, 2]
    ]
    indices = local_indices(padded, rank)
    local = [x.detach().index_select(0, indices).requires_grad_() for x in full]
    actual = attention(*local, packed_seq_params=params_for(lengths, padded))
    reference_attention = make_attention(1)
    outputs = []
    valid = torch.zeros(total, dtype=torch.bool, device="cuda")
    start = 0
    for length, physical in zip(lengths, padded):
        outputs.append(reference_attention(*(x[start : start + length].unsqueeze(1) for x in full)))
        if physical > length:
            outputs.append(torch.zeros(physical - length, 1, 256, device="cuda", dtype=torch.bfloat16))
        valid[start : start + length] = True
        start += physical
    reference = torch.cat(outputs)
    local_valid = valid.index_select(0, indices)
    expected = reference.index_select(0, indices)
    torch.testing.assert_close(actual[local_valid], expected[local_valid], rtol=0.06, atol=0.025)
    probe = torch.randn_like(reference)
    (actual[local_valid].float() * probe.index_select(0, indices)[local_valid].float()).sum().backward()
    (reference[valid].float() * probe[valid].float()).sum().backward()
    errors = []
    for observed, source in zip(local, full):
        wanted = source.grad.index_select(0, indices)
        torch.testing.assert_close(observed.grad[local_valid], wanted[local_valid], rtol=0.08, atol=0.06)
        errors.append(float((observed.grad[local_valid] - wanted[local_valid]).float().abs().max()))
    logger.info(
        "CP_PARITY_PASS %s",
        json.dumps(
            {
                "rank": rank,
                "lengths": lengths,
                "padded": padded,
                "max_output_error": float((actual[local_valid] - expected[local_valid]).float().abs().max()),
                "max_gradient_errors": errors,
            }
        ),
    )
    dist.barrier()


def check_long_attention(attention: NemotronLabsDiffusionAttention) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = torch.randn(65536, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(65536, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = attention(q, k, v, packed_seq_params=params_for([131072], [131072]))
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(x.grad).all() for x in [q, k, v])
    logger.info(
        "CP_128K_ATTENTION_PASS %s",
        json.dumps(
            {
                "rank": dist.get_rank(),
                "local_tokens": 65536,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        ),
    )
    dist.barrier()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    attention = make_attention(2)
    try:
        check_parity(attention, [128, 256], [128, 256])
        check_parity(attention, [115, 241], [128, 256])
        check_long_attention(attention)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
