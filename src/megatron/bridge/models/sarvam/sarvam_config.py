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

"""Builder-backed model config for Sarvam MoE."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


if TYPE_CHECKING:
    from megatron.bridge.models.gpt_provider import GPTModelProvider


def _get_sarvam_moe_pipeline_layout(pp_size: int):
    """Return supported pipeline layouts for Sarvam MoE's 19 decoder layers."""
    map_pp_to_layout = {
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
    if pp_size not in map_pp_to_layout:
        raise ValueError(
            f"Unsupported PP size {pp_size} for Sarvam MoE pipeline layout. "
            f"Supported sizes: {sorted(map_pp_to_layout)}. "
            "Set pipeline_model_parallel_layout explicitly for other PP sizes."
        )
    layout = map_pp_to_layout[pp_size]
    if layout is not None:
        layout = [list(stage) for stage in layout]
    return layout


def apply_sarvam_moe_pipeline_layout(config: "GPTModelProvider | BridgeGPTModelConfig") -> None:
    """Select a supported pipeline layout when the decoder layers do not divide evenly across PP.

    Sarvam MoE has 19 transformer layers, so uneven PP requires an explicit
    layout rather than a generic first-stage override. Use known-good layouts
    for PP=2/4/8 unless the caller has already configured a custom flexible
    pipeline split.

    Args:
        config: Sarvam MoE provider or builder-backed model config, updated in place.
    """
    pp = config.pipeline_model_parallel_size or 1
    has_explicit_flexible_pp = (
        getattr(config, "pipeline_model_parallel_layout", None) is not None
        or getattr(config, "num_layers_in_first_pipeline_stage", None) is not None
        or getattr(config, "num_layers_in_last_pipeline_stage", None) is not None
    )
    if pp > 1 and config.num_layers % pp != 0 and not has_explicit_flexible_pp:
        config.pipeline_model_parallel_layout = _get_sarvam_moe_pipeline_layout(pp)


@dataclass(kw_only=True)
class SarvamMoEModelConfig(BridgeGPTModelConfig):
    """GPT model config that keeps Sarvam MoE's supported uneven pipeline layouts.

    Sarvam MoE builds with the standard GPT builder. The only addition over
    ``BridgeGPTModelConfig`` is the pipeline layout ``finalize()`` selects when
    the pipeline size does not divide the decoder layer count, matching the
    legacy ``SarvamMoEModelProvider``.
    """

    def finalize(self) -> None:
        """Apply a supported layout for uneven pipeline parallelism, then finalize."""
        apply_sarvam_moe_pipeline_layout(self)
        super().finalize()


__all__ = ["SarvamMoEModelConfig", "apply_sarvam_moe_pipeline_layout"]
