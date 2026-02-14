#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Benchmark the Hugging Face ↔ Megatron-LM round-trip conversion path.

The script mirrors the multi-GPU round-trip example but focuses solely on timing
the import (HF ➔ Megatron) and export (Megatron ➔ HF weights) phases. No
checkpoints are written; the goal is to provide a lightweight way to measure
converter performance across different parallelism configurations.

Usage examples:

    uv run python examples/conversion/hf_megatron_roundtrip_benchmark.py \
        --hf-model-id meta-llama/Llama-3.2-1B

    uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_benchmark.py \
        --hf-model-id Qwen/Qwen3-30B-A3B --tp 1 --pp 1 --ep 8
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter

import torch
from rich.console import Console
from rich.table import Table

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import weights_verification_table
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()


def _env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return _env_rank() == 0


def _maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _configure_model_provider(
    model_provider,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
) -> None:
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)


def _render_results(import_duration: float, export_duration: float) -> None:
    table = Table(title="HF ↔ Megatron Round-Trip Benchmark")
    table.add_column("Stage", style="cyan")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Description")
    table.add_row("Import", f"{import_duration:.2f}", "HF tensors → Megatron weights")
    table.add_row("Export", f"{export_duration:.2f}", "Megatron weights → HF tensors")
    table.add_row("Total", f"{import_duration + export_duration:.2f}", "Import + export")
    console.print(table)


def _maybe_warn_about_world_size() -> None:
    if _env_world_size() == 1 and not torch.distributed.is_initialized():
        console.print("[yellow]WORLD_SIZE is 1. Launch with torchrun for multi-GPU benchmarks if desired.[/yellow]")


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    trust_remote_code: bool | None = None,
    show_progress: bool = True,
) -> None:
    """Benchmark the HF ↔ Megatron conversion path without saving checkpoints."""
    # Import (HF -> Megatron)
    import_start = perf_counter()
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(
            trust_remote_code=trust_remote_code,
            hf_path=hf_model_id,
        ),
        torch_dtype=torch.bfloat16,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    _configure_model_provider(model_provider, tp, pp, ep, etp)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)
    _sync_cuda()
    _maybe_barrier()
    import_duration = perf_counter() - import_start

    # Export (Megatron -> HF weights iteration only)
    export_start = perf_counter()
    table = weights_verification_table(bridge, megatron_model)
    if _is_rank_zero():
        console.print(table)
    _sync_cuda()
    _maybe_barrier()
    export_duration = perf_counter() - export_start

    if _is_rank_zero():
        console.print(f"[bold]Benchmarking round-trip for[/bold] [green]{hf_model_id}[/green]")
        console.print(f"[yellow]TP={tp} | PP={pp} | EP={ep} | ETP={etp} | world_size={_env_world_size()}[/yellow]")
        _maybe_warn_about_world_size()
        _render_results(import_duration, export_duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the Hugging Face ↔ Megatron round-trip import/export times."
    )
    parser.add_argument("--hf-model-id", type=str, required=True, help="Hugging Face model ID to benchmark.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size.")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size.")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size.")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow loading remote code from the Hub.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the export progress bar for quieter benchmarking logs.",
    )
    args = parser.parse_args()
    main(
        hf_model_id=args.hf_model_id,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        trust_remote_code=args.trust_remote_code,
        show_progress=not args.no_progress,
    )
