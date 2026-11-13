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

"""Save a ModelOpt-quantized toy model as a quantized Hugging Face checkpoint (run with torch.distributed.run).

Weight ranges come from max calibration of the weights, which ModelOpt syncs across tensor parallelism,
and every input range is pinned to :func:`input_amax` of its global layer and kind, so exports at any
parallelism describe the same quantized model and can be compared bit for bit.
"""

import argparse
import re

import modelopt.torch.quantization as mtq
import torch
from megatron.core.utils import unwrap_model
from modelopt.torch.export.unified_export_megatron import export_mcore_gpt_to_hf
from modelopt.torch.quantization.nn import TensorQuantizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main


NVFP4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}, "axis": None}
FP8 = {"num_bits": (4, 3), "axis": None}
MXFP4 = {"num_bits": (2, 1), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}}
DISABLED = [
    {"quantizer_name": "*", "enable": False},
]
UNQUANTIZED = [
    {"quantizer_name": "*router*", "enable": False},
    {"quantizer_name": "*output_layer*", "enable": False},
    {"quantizer_name": "*conv1d*", "enable": False},
    # Kept in BF16 by deployments (vision towers, DSA indexers, DeepSeek-V4 compressors), or not
    # linear layers in HF (hyper-connection mixers) or router-like (Kimi-K3 attention residuals).
    {"quantizer_name": "*vision_model*", "enable": False},
    {"quantizer_name": "*visual*", "enable": False},
    {"quantizer_name": "*indexer*", "enable": False},
    {"quantizer_name": "*compressor*", "enable": False},
    {"quantizer_name": "*hyper_connection*", "enable": False},
    {"quantizer_name": "*res_proj*", "enable": False},
]
CONFIGS = {
    "nvfp4": [
        {"quantizer_name": "*weight_quantizer", "cfg": NVFP4},
        {"quantizer_name": "*input_quantizer", "cfg": NVFP4},
    ],
    "fp8": [
        {"quantizer_name": "*weight_quantizer", "cfg": FP8},
        {"quantizer_name": "*input_quantizer", "cfg": FP8},
    ],
    # NVFP4 MLP and FP8 attention, as in mixed-precision recipes.
    "mixed": [
        {"quantizer_name": "*mlp*weight_quantizer", "cfg": NVFP4},
        {"quantizer_name": "*mlp*input_quantizer", "cfg": NVFP4},
        {"quantizer_name": "*self_attention*weight_quantizer", "cfg": FP8},
        {"quantizer_name": "*self_attention*input_quantizer", "cfg": FP8},
    ],
    # Weight-only MXFP4 experts, as GPT-OSS ships.
    "mxfp4_experts": [
        {"quantizer_name": "*experts*weight_quantizer", "cfg": MXFP4},
    ],
}
# Projections deployments fuse into one GEMM read the same input; ModelOpt calibrates them to one weight range.
MLA_DOWN_PROJECTIONS = (
    "linear_q_down_proj",
    "linear_kv_down_proj",
    "linear_kv_proj",
    "q_a_proj",
    "kv_a_proj_with_mqa",
)
KIND_IDS = {
    "linear_qkv": 1,
    "linear_proj": 2,
    "linear_fc1": 3,
    "linear_fc2": 4,
    "in_proj": 5,
    "out_proj": 6,
    **dict.fromkeys((*MLA_DOWN_PROJECTIONS, "g_proj"), 7),  # With Kimi-K3's MLA gate.
    **dict.fromkeys(("q_proj", "k_proj", "v_proj"), 8),
    "linear_q_up_proj": 9,
    "linear_kv_up_proj": 10,
    "q_b_proj": 12,
    "kv_b_proj": 13,
    "o_proj": 14,
    "b_proj": 16,
    "beta_proj": 17,
    "f_a_proj": 18,
    "f_b_proj": 19,
    "g_a_proj": 20,
    "g_b_proj": 21,
    "fc1_latent_proj": 22,
    "fc2_latent_proj": 23,
}


def input_amax(layer: int, kind: str) -> float:
    """Input range pinned to a quantizer, from its global layer and kind."""
    return 1.0 + 0.01 * layer + 0.1 * KIND_IDS[kind]


def _pin_input_ranges(chunk: torch.nn.Module) -> None:
    for name, quantizer in chunk.named_modules():
        if not (isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled and name.endswith("input_quantizer")):
            continue
        layer = chunk.get_submodule(re.match(r"^.*?decoder\.layers\.\d+", name).group(0))
        # Hyper-connection layers wrap the numbered layer.
        layer = getattr(layer, "layer_number", None) or layer.inner_layer.layer_number
        layer -= 1
        kind = re.search(r"(\w+)\.input_quantizer", name).group(1)
        quantizer.amax = torch.tensor(input_amax(layer, kind), device="cuda")


@torchrun_main
def main(
    hf_model_path: str, out: str, config: str, tp: int, pp: int, ep: int, etp: int, offline_out: str | None
) -> None:
    # Kimi-K3's config class ships with the checkpoint.
    bridge = AutoBridge.from_hf_pretrained(hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    # Also sets bridge-specific pipeline layouts, e.g. for DeepSeek-V4's hash-routed MoE layers.
    provider.apply_overrides_and_finalize(
        overrides={
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
            "moe_grouped_gemm": False,
            "gradient_accumulation_fusion": False,
            "pipeline_dtype": torch.bfloat16,
        }
    )
    provider.initialize_model_parallel(seed=0)
    model = provider.provide_distributed_model(wrap_with_ddp=False)

    quant_cfg = {"quant_cfg": [*DISABLED, *CONFIGS[config], *UNQUANTIZED], "algorithm": "max"}
    for chunk in model:
        chunk = unwrap_model(chunk)
        mtq.quantize(chunk, quant_cfg)  # Without a forward loop, "max" calibrates the weights only.
        _pin_input_ranges(chunk)
    bridge.save_hf_pretrained_modelopt(model, out)
    if offline_out is not None:
        export_mcore_gpt_to_hf(unwrap_model(model[0]), hf_model_path, export_dir=offline_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--offline-out", default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    args = parser.parse_args()
    main(args.hf_model_path, args.out, args.config, args.tp, args.pp, args.ep, args.etp, args.offline_out)
