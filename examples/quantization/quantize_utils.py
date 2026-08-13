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
Shared utilities for quantization scripts.

This module provides common functionality used across different quantization
scripts (LLM and VLM) to avoid code duplication.
"""

import argparse
import copy
import os
import sys

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.distributed as dist
import torch
from rich.console import Console
from rich.table import Table

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


# Shared console instance for rich output
console = Console()

# Quantization configuration choices
QUANT_CFG_CHOICES = {
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "fp8_blockwise": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "mamba_moe_fp8_aggressive": mtq.MAMBA_MOE_FP8_AGGRESSIVE_CFG,
    "mamba_moe_fp8_conservative": mtq.MAMBA_MOE_FP8_CONSERVATIVE_CFG,
    "mamba_moe_nvfp4_aggressive": mtq.MAMBA_MOE_NVFP4_AGGRESSIVE_CFG,
    "mamba_moe_nvfp4_conservative": mtq.MAMBA_MOE_NVFP4_CONSERVATIVE_CFG,
}


def get_modelopt_torch_quantization_config(
    export_quant_cfg: str, export_kv_cache_quant: bool = False, weight_only: bool = False
) -> dict:
    """Return a quantization config based on the specified configuration.

    Args:
        export_quant_cfg: Quantization configuration name (e.g., "fp8", "int8_sq").
        export_kv_cache_quant: Whether to enable KV cache quantization.
        weight_only: Whether to disable input quantization (weight-only quantization).

    Returns:
        ModelOpt quantization configuration dictionary.

    Raises:
        KeyError: If export_quant_cfg is not a valid configuration name.
    """
    # Use deepcopy to avoid mutating the original config in QUANT_CFG_CHOICES
    mtq_config = copy.deepcopy(QUANT_CFG_CHOICES[export_quant_cfg])

    fp8_cfg = {"num_bits": (4, 3), "axis": None}
    fp4_cfg = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
    }

    if "fp8" == export_quant_cfg:
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"].append({"quantizer_name": "*medusa_heads**", "cfg": fp8_cfg})
    if "fp4" in export_quant_cfg:
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"].append({"quantizer_name": "*medusa_heads**", "cfg": fp4_cfg})
    if "awq" in export_quant_cfg:
        weight_entry = mtq.config.find_quant_cfg_entry_by_path(mtq_config["quant_cfg"], "*weight_quantizer")
        weight_cfg = weight_entry["cfg"]
        if isinstance(weight_cfg, list):
            weight_cfg = weight_cfg[0]
        weight_cfg["block_sizes"][-1] = 128
    if export_kv_cache_quant:
        mtq_config["quant_cfg"].append({"quantizer_name": "*linear_qkv.output_quantizer", "cfg": fp8_cfg})
    if weight_only:
        mtq_config["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})

    return mtq_config


def require_torchrun() -> None:
    """Exit with guidance unless the script was launched under torchrun.

    These examples all build a distributed model, so they cannot run as a plain
    `python script.py` invocation.
    """
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)


def require_checkpoint(megatron_load_path: str, *, quantize_command: str) -> None:
    """Exit with guidance unless the quantized Megatron checkpoint exists.

    Args:
        megatron_load_path: Path the script expects to load the quantized checkpoint from.
        quantize_command: The command that produces that checkpoint, echoed to the user.
    """
    if not os.path.exists(megatron_load_path):
        console.print(f"[red]Error: Quantized checkpoint path {megatron_load_path} does not exist![/red]")
        console.print("[yellow]Please run the quantization process first:[/yellow]")
        console.print(f"[yellow]{quantize_command}[/yellow]")
        sys.exit(1)


def build_bridge_and_provider(
    hf_model_id: str,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    load_weights: bool,
    pipeline_dtype: torch.dtype,
    trust_remote_code: bool | None = None,
) -> tuple[AutoBridge, GPTModelProvider]:
    """Build an AutoBridge for a HuggingFace model and a provider configured for multi-GPU execution.

    The provider is finalized and model parallel is initialized, so the caller can immediately
    call `provide_distributed_model` (to build from HF weights) or `bridge.load_megatron_model`
    (to load an existing Megatron checkpoint).

    Args:
        hf_model_id: HuggingFace model ID or local path.
        tp: Tensor parallel size.
        pp: Pipeline parallel size.
        ep: Expert parallel size.
        etp: Expert tensor parallel size.
        load_weights: Whether the bridge should load HF weights. Pass False when the weights
            will come from a Megatron checkpoint instead.
        pipeline_dtype: Pipeline dtype for the provider.
        trust_remote_code: Forwarded to `is_safe_repo` to decide whether remote code is allowed.

    Returns:
        The bridge and its finalized model provider.
    """
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model_id),
    )

    model_provider = bridge.to_megatron_provider(load_weights=load_weights)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = pipeline_dtype

    # All models use TE spec (default) for quantization.
    # Once all overrides are set, finalize the provider so its post-initialization logic runs.
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    return bridge, model_provider


def load_quantized_megatron_model(
    bridge: AutoBridge,
    megatron_load_path: str,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
) -> list[torch.nn.Module]:
    """Load a quantized Megatron checkpoint, re-sharded to the requested parallelism.

    Args:
        bridge: Bridge built by `build_bridge_and_provider`.
        megatron_load_path: Path to the quantized Megatron checkpoint.
        tp: Tensor parallel size.
        pp: Pipeline parallel size.
        ep: Expert parallel size.
        etp: Expert tensor parallel size.

    Returns:
        The loaded model chunks (not DDP-wrapped).
    """
    return bridge.load_megatron_model(
        megatron_load_path,
        mp_overrides={
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
        wrap_with_ddp=False,
    )


def print_parallelism_summary(model_provider: GPTModelProvider) -> None:
    """Print the resolved parallelism sizes on the master rank.

    Args:
        model_provider: The finalized provider whose sizes are reported.
    """
    if not dist.is_master():
        return
    console.print(f"[green]Tensor parallel size: {model_provider.tensor_model_parallel_size}[/green]")
    console.print(f"[green]Pipeline parallel size: {model_provider.pipeline_model_parallel_size}[/green]")
    console.print(f"[green]Expert parallel size: {model_provider.expert_model_parallel_size}[/green]")
    console.print(f"[green]Expert tensor parallel size: {model_provider.expert_tensor_parallel_size}[/green]")


def create_quantization_stats_table() -> Table:
    """Create a rich Table for displaying quantization statistics.

    Returns:
        Configured Table instance for quantization statistics.
    """
    table = Table(title="Quantization Statistics")
    table.add_column("Parameter Name", style="cyan")
    table.add_column("Shape")
    table.add_column("Max Value", justify="right")
    return table


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the model and parallelism arguments shared by every quantization example.

    These are the arguments needed to build a bridge and provider, so they apply equally to the
    scripts that quantize a model and to the ones that load an already-quantized checkpoint.

    Args:
        parser: The argparse.ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--hf-model-id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g. meta-llama/Llama-3.2-1B).",
    )

    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading HuggingFace models.",
    )


def add_common_quantization_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by the scripts that perform quantization."""
    add_common_model_args(parser)

    parser.add_argument(
        "--megatron-save-path",
        type=str,
        default=None,
        help="Path to save the quantized model in Megatron checkpoint format. "
        "If not provided, will use default path: {model_name}_quantized_{config}",
    )
    parser.add_argument(
        "--export-quant-cfg",
        type=str,
        default="fp8",
        choices=list(QUANT_CFG_CHOICES.keys()),
        help="Quantization configuration to use.",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=512,
        help="Samples to use for PTQ calibration.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable real low-bit quantization.",
    )
    parser.add_argument(
        "--weight-only",
        action="store_true",
        help="Disable input quantization.",
    )
    parser.add_argument(
        "--export-kv-cache-quant",
        action="store_true",
        help="Enable KV cache quantization.",
    )
