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

"""Provider-neutral configuration and build helpers for Sarvam models."""

from dataclasses import dataclass

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def sarvam_mla_layer_spec(config: BridgeGPTModelConfig, vp_stage: int | None = None) -> ModuleSpec:
    """Build the Transformer Engine decoder block used by Sarvam MLA."""
    return get_gpt_decoder_block_spec(
        config.transformer,
        use_transformer_engine=True,
        normalization="RMSNorm",
        vp_stage=vp_stage,
    )


@dataclass(kw_only=True)
class SarvamMLAModelConfig(BridgeGPTModelConfig):
    """Serializable builder config for Sarvam MLA models."""

    transformer_layer_spec = sarvam_mla_layer_spec


def get_sarvam_moe_pipeline_layout(pipeline_size: int) -> list[list[str]] | None:
    """Return supported pipeline layouts for Sarvam MoE's 19 decoder layers."""
    layouts = {
        1: None,
        2: [["embedding"] + ["decoder"] * 10, ["decoder"] * 9 + ["loss"]],
        4: [["embedding"] + ["decoder"] * 5, ["decoder"] * 5, ["decoder"] * 5, ["decoder"] * 4 + ["loss"]],
        8: [
            ["embedding"] + ["decoder"] * 3,
            ["decoder"] * 3,
            ["decoder"] * 3,
            ["decoder"] * 2,
            ["decoder"] * 2,
            ["decoder"] * 2,
            ["decoder"] * 2,
            ["decoder"] * 2 + ["loss"],
        ],
    }
    if pipeline_size not in layouts:
        raise ValueError(
            f"Unsupported PP size {pipeline_size} for Sarvam MoE pipeline layout. "
            f"Supported sizes: {sorted(layouts)}. "
            "Set pipeline_model_parallel_layout explicitly for other PP sizes."
        )
    layout = layouts[pipeline_size]
    return None if layout is None else [list(stage) for stage in layout]


__all__ = ["SarvamMLAModelConfig", "get_sarvam_moe_pipeline_layout", "sarvam_mla_layer_spec"]
