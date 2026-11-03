"""Compare unpacked causal CP forward/backward against CP=1 at MBS2.

Run with ``uv run python -m torch.distributed.run --nproc_per_node=8``.
"""

import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import torch
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import NemotronLabsDiffusionAttention
from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
    NemotronLabsDiffusionModelProvider,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[1024, 131072])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    cp_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    model_parallel_cuda_manual_seed(42)
    groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(), cp=parallel_state.get_context_parallel_group()
    )
    rows = []
    for seq in args.sequence_lengths:
        config = NemotronLabsDiffusionModelProvider(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=64,
            context_parallel_size=cp_size,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            seq_length=seq,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            rotary_base=10000.0,
            apply_llama4_style_query_key_layer_scaling=True,
            llama4_scaling_beta=0.1,
            yarn_original_max_position_embeddings=1024,
        )
        cp = NemotronLabsDiffusionAttention(config, 1, AttnMaskType.causal, "self", pg_collection=groups).cuda()
        cp.set_inference_mode(True)
        cp.set_inference_params(causal=True, cache_enabled=False)
        # The reference uses the unchanged CP=1 SDPA path, with the same RoPE and scaling.
        ref = NemotronLabsDiffusionAttention(
            replace(config, context_parallel_size=1), 1, AttnMaskType.causal, "self", pg_collection=groups
        ).cuda()
        ref.set_inference_mode(True)
        ref.set_inference_params(causal=True, cache_enabled=False)
        torch.manual_seed(123)
        full = [
            torch.randn(seq, 2, heads, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for heads in (4, 2, 2)
        ]
        segment = seq // (2 * cp_size)
        indices = torch.cat(
            (
                torch.arange(rank * segment, (rank + 1) * segment, device="cuda"),
                torch.arange((2 * cp_size - rank - 1) * segment, (2 * cp_size - rank) * segment, device="cuda"),
            )
        )
        local = [x.detach().index_select(0, indices).requires_grad_() for x in full]
        torch.cuda.reset_peak_memory_stats()
        expected = ref(*full)
        actual = cp(*local)
        torch.testing.assert_close(actual, expected.index_select(0, indices), atol=0.02, rtol=0.03)
        weight = torch.randn_like(expected)
        (expected * weight).sum().backward()
        (actual * weight.index_select(0, indices)).sum().backward()
        errors = {}
        for name, x, y in zip(("q", "k", "v"), local, full):
            expected_grad = y.grad.index_select(0, indices)
            torch.testing.assert_close(x.grad, expected_grad, atol=0.04, rtol=0.05)
            errors[name] = float(
                (x.grad.float() - expected_grad.float()).norm() / expected_grad.float().norm().clamp_min(1e-12)
            )
            assert errors[name] < 0.02, errors
        error = float(
            (actual.float() - expected.index_select(0, indices).float()).norm()
            / expected.index_select(0, indices).float().norm().clamp_min(1e-12)
        )
        assert error < 0.01, error
        metrics = dict(
            sequence_length=seq,
            micro_batch_size=2,
            cp_size=cp_size,
            rank=rank,
            forward_relative_error=error,
            gradient_relative_errors=errors,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        gathered = [None] * cp_size
        torch.distributed.all_gather_object(gathered, metrics)
        rows.extend(gathered)
        if rank == 0:
            logging.getLogger(__name__).info("UNPACKED_CP_PARITY %s", json.dumps(gathered))
        del cp, ref, full, local, expected, actual, weight
        torch.cuda.empty_cache()
    if rank == 0:
        args.report.write_text(json.dumps(dict(status="PASS", results=rows), indent=2) + "\n")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
