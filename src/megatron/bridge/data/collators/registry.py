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

"""Lazy resolution of model-owned VLM collators by HF processor type."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class _ModelCollateSpec:
    module_name: str
    symbol_name: str
    required_for_all_examples: bool = False


_MODEL_COLLATE_SPECS = {
    "Qwen2_5_VLProcessor": _ModelCollateSpec("megatron.bridge.models.qwen_vl.data.collate_fn", "qwen2_5_collate_fn"),
    "Qwen3VLProcessor": _ModelCollateSpec("megatron.bridge.models.qwen_vl.data.collate_fn", "qwen2_5_collate_fn"),
    "Qwen3OmniMoeProcessor": _ModelCollateSpec(
        "megatron.bridge.models.qwen_omni.data.collate_fn",
        "qwen3_omni_collate_fn",
    ),
    "NemotronNanoVLV2Processor": _ModelCollateSpec(
        "megatron.bridge.models.nemotron_vl.data.collate_fn",
        "nemotron_nano_v2_vl_collate_fn",
    ),
    "NemotronH_Nano_Omni_Reasoning_V3Processor": _ModelCollateSpec(
        "megatron.bridge.models.nemotron_omni.data.collate_fn",
        "nemotron_omni_collate_fn",
        required_for_all_examples=True,
    ),
    "PixtralProcessor": _ModelCollateSpec(
        "megatron.bridge.models.ministral3.data.collate_fn", "ministral3_collate_fn"
    ),
    "Gemma3Processor": _ModelCollateSpec("megatron.bridge.models.gemma_vl.data.collate_fn", "gemma3_vl_collate_fn"),
    "Gemma4Processor": _ModelCollateSpec("megatron.bridge.models.gemma_vl.data.collate_fn", "gemma4_vl_collate_fn"),
    "Qwen2AudioProcessor": _ModelCollateSpec(
        "megatron.bridge.models.qwen_audio.data.collate_fn", "qwen2_audio_collate_fn"
    ),
    "Glm4vProcessor": _ModelCollateSpec("megatron.bridge.models.glm_vl.data.collate_fn", "glm4v_collate_fn"),
    "KimiK25Processor": _ModelCollateSpec("megatron.bridge.models.kimi_vl.data.collate_fn", "kimi_k25_vl_collate_fn"),
}


def model_collate_required_for_all_examples(processor_type: str) -> bool:
    """Return whether a processor must always use its model-owned collator."""
    spec = _MODEL_COLLATE_SPECS.get(processor_type)
    return spec is not None and spec.required_for_all_examples


@lru_cache(maxsize=None)
def resolve_model_collate(processor_type: str) -> Callable[..., dict[str, Any]]:
    """Resolve a model-owned collator without importing unrelated model modules."""
    try:
        spec = _MODEL_COLLATE_SPECS[processor_type]
    except KeyError as error:
        raise ValueError(
            f"No VLM collate function is registered for processor type '{processor_type}'. "
            "Register a model-owned collator or pass a collate function explicitly."
        ) from error
    collate = getattr(import_module(spec.module_name), spec.symbol_name)
    if not callable(collate):
        raise TypeError(f"Registered collator {spec.module_name}.{spec.symbol_name} is not callable.")
    return collate
