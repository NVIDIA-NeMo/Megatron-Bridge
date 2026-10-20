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

"""Save a ModelOpt fake-quantized toy Qwen3-MoE to Hugging Face format (run with torch.distributed.run).

``--mode activations`` enables NVFP4 input and FP8 KV-cache quantizers and pins each range to
:func:`expected_amax` of its global layer, kind and expert, so the export can be checked exactly at
any parallelism. ``--mode weights`` folds INT8 weight quantizers into the weights, leaving no
quantizer enabled.
"""

import argparse
import re

import modelopt.torch.quantization as mtq
import torch
from megatron.core.utils import unwrap_model
from modelopt.torch.quantization.nn import TensorQuantizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main


NVFP4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}, "axis": None}
FP8 = {"num_bits": (4, 3), "axis": None}
KIND_IDS = {"linear_qkv": 1, "linear_proj": 2, "linear_fc1": 3, "linear_fc2": 4, "k_bmm": 5, "v_bmm": 6}


def expected_amax(layer: int, kind: str, expert: int = 0) -> float:
    """Range pinned to a quantizer, from its global layer, kind and expert (or EP rank)."""
    return 1.0 + 0.01 * layer + 0.1 * KIND_IDS[kind] + 0.001 * expert


def _pin_ranges(chunk: torch.nn.Module, num_experts: int, ep_rank: int, ep_size: int) -> None:
    for name, quantizer in chunk.named_modules():
        if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
            continue
        layer_prefix = re.match(r"^.*?decoder\.layers\.\d+", name).group(0)
        layer = chunk.get_submodule(layer_prefix).layer_number - 1
        kind = re.search(r"(linear_\w+)\.input_quantizer|([kv]_bmm)_quantizer", name)
        kind = kind.group(1) or kind.group(2)
        local_expert = re.search(r"local_experts\.(\d+)\.", name)
        if local_expert:  # SequentialMLP: one quantizer per expert
            expert = int(local_expert.group(1)) + ep_rank * num_experts // ep_size
        else:  # Grouped GEMM: one quantizer per EP rank's experts
            expert = ep_rank if ".experts." in name else 0
        quantizer.amax = torch.tensor(expected_amax(layer, kind, expert), device="cuda")


@torchrun_main
def main(hf_model_path: str, out: str, mode: str, tp: int, pp: int, ep: int, grouped: bool) -> None:
    bridge = AutoBridge.from_hf_pretrained(hf_model_path, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.moe_grouped_gemm = grouped
    provider.pipeline_dtype = torch.bfloat16
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    chunks = [unwrap_model(chunk) for chunk in model]

    disable_all = {"quantizer_name": "*", "enable": False}
    if mode == "weights":
        config = {
            "quant_cfg": [disable_all, {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}}],
            "algorithm": "max",
        }
    else:
        config = {
            "quant_cfg": [
                disable_all,
                {"quantizer_name": "*input_quantizer", "cfg": NVFP4},
                {"quantizer_name": "*[kv]_bmm_quantizer", "cfg": FP8},
                {"quantizer_name": "*router*", "enable": False},
                {"quantizer_name": "*output_layer*", "enable": False},
            ],
            "algorithm": None,
        }
    for chunk in chunks:
        mtq.quantize(chunk, config)  # Weight-only "max" calibrates from the weights alone.
        if mode == "weights":
            mtq.fold_weight(chunk)
        else:
            ep_rank = torch.distributed.get_rank(provider._pg_collection.ep) if ep > 1 else 0
            _pin_ranges(chunk, provider.num_moe_experts, ep_rank, ep)
    bridge.save_hf_pretrained(model, out)

    if mode == "activations" and tp == pp == ep == 1:
        # Quantizer state the export cannot carry must fail it rather than be dropped: an enabled
        # quantizer without a Hugging Face name, and a tensor other than ``_amax``.
        attention = chunks[0].decoder.layers[0].self_attention
        attention.linear_qkv.output_quantizer.enable()
        attention.linear_proj.input_quantizer.register_buffer("_pre_quant_scale", torch.ones(1, device="cuda"))
        try:
            bridge.save_hf_pretrained(model, f"{out}_rejected")
        except ValueError as error:
            assert "linear_qkv.output_quantizer" in str(error), error
            assert "_pre_quant_scale" in str(error), error
            print("Unexportable quantizer state rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["activations", "weights"], default="activations")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    main(args.hf_model_path, args.out, args.mode, args.tp, args.pp, args.ep, args.grouped)
