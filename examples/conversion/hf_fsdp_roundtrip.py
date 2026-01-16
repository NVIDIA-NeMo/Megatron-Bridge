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
Round-trip conversion between Hugging Face and Megatron FSDP.

Usage examples:
    python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_fsdp_roundtrip.py --hf-model-id Qwen/Qwen3-30B-A3B --ep 8
"""

from __future__ import annotations

import argparse
import os

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import weights_verification_table
from megatron.bridge.models.decorators import torchrun_main
from megatron.core.distributed import DistributedDataParallelConfig


console = Console()
HF_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _configure_model_provider(model_provider, ep: int, torch_dtype: torch.dtype) -> None:
    model_provider.pipeline_dtype = torch_dtype
    model_provider.params_dtype = torch_dtype
    model_provider.expert_model_parallel_size = ep
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    output_dir: str | None = None,
    ep: int = 1,
    torch_dtype: str = "bf16",
) -> None:
    """Perform round-trip conversion between HuggingFace and Megatron-FSDP models."""
    model_name = hf_model_id.split("/")[-1]
    save_path = os.path.join(output_dir, model_name) if output_dir else model_name

    dtype = _parse_dtype(torch_dtype)
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        torch_dtype=dtype,
    )

    model_provider = bridge.to_megatron_provider(load_weights=True)
    _configure_model_provider(model_provider, ep, dtype)

    ddp_config=DistributedDataParallelConfig(
        use_distributed_optimizer=True, 
        check_for_nan_in_grad=True,
        use_megatron_fsdp=True, 
        data_parallel_sharding_strategy='optim_grads_params',
    )

    megatron_model = model_provider.provide_distributed_model(
        ddp_config=ddp_config,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
    )

    table = weights_verification_table(bridge, megatron_model)
    if _is_rank_zero():
        console.print(table)
    _maybe_barrier()

    if _is_rank_zero():
        console.print(f"Saving HF-ckpt in {save_path}...")
    bridge.save_hf_pretrained(megatron_model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between HuggingFace and Megatron-FSDP model formats."
    )
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID to convert")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where the converted model directory will be created. Defaults to CWD.",
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size.")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="Weights dtype for Megatron model initialization.",
    )

    args = parser.parse_args()
    main(
        hf_model_id=args.hf_model_id,
        output_dir=args.output_dir,
        ep=args.ep,
        torch_dtype=args.torch_dtype,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
