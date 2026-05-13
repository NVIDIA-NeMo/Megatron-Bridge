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
import inspect

import modelopt.torch.quantization as mtq
from rich.console import Console
from rich.table import Table


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


def patch_modelopt_te_linear_tuple_output() -> None:
    """Patch ModelOpt TE Linear quantization for TE 2.15 tuple outputs."""
    try:
        import modelopt.torch.quantization.plugins.transformer_engine as modelopt_te
        import transformer_engine.pytorch.module.linear as te_linear
    except ImportError:
        return

    quant_te_linear = getattr(modelopt_te, "_QuantTELinear", None)
    if quant_te_linear is None or getattr(quant_te_linear, "_bridge_tuple_output_patch", False):
        return

    # Temporary compatibility shim for nvidia-modelopt==0.44.0rc4 with TE 2.15.
    # ModelOpt rc4 already handles TE's changed Linear functional signature by
    # locating `weight` and `inp` by name. It still assumes the TE functional
    # returns a tensor, while TE 2.15 returns `(output, weight_workspace)` in this
    # path. Remove this patch once ModelOpt ships the tuple-output fix upstream.
    def te_quantized_linear_fn(package, func_name, self, *args, **kwargs):
        modelopt_te._assert_te_fp8_enabled()
        # `replace_function` stores the original forward as `_forward` while the
        # patch is active. The `_forward` path receives no leading autograd ctx

    orig_forward = getattr(te_linear._Linear, "_forward", te_linear._Linear.forward)
    names = list(inspect.signature(orig_forward).parameters)

    def te_quantized_linear_fn(package, func_name, self, *args, **kwargs):
        modelopt_te._assert_te_fp8_enabled()
        ctx_offset = 0 if func_name == "_forward" else 1
        weight_pos = names.index("weight") - ctx_offset
        inp_pos = names.index("inp") - ctx_offset
        new_args = list(args)
        new_args[weight_pos] = self.weight_quantizer(args[weight_pos])
        new_args[inp_pos] = self.input_quantizer(args[inp_pos])
        output = getattr(package, func_name)(*new_args, **kwargs)
        if isinstance(output, tuple):
            # TE returns auxiliary workspace metadata in later tuple elements.
            # Only the activation tensor participates in output quantization.
            quantized_output = self.output_quantizer(output[0])
            return (quantized_output, *output[1:])
        return self.output_quantizer(output)

    quant_te_linear.te_quantized_linear_fn = staticmethod(te_quantized_linear_fn)
    quant_te_linear._quantized_linear_fn = staticmethod(te_quantized_linear_fn)
    quant_te_linear._bridge_tuple_output_patch = True


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


def add_common_quantization_args(parser: argparse.ArgumentParser) -> None:
    """Add common quantization arguments to an argument parser.

    Args:
        parser: The argparse.ArgumentParser to add arguments to.
    """
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")

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
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading HuggingFace models.",
    )
