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

"""Stable facade for Megatron-Core GPT model-builder APIs."""

from typing import Any, ClassVar


def _load_gpt_symbols() -> tuple[Any, type, Any]:
    try:
        from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig, mtp_block_spec
    except ModuleNotFoundError as error:
        if error.name != "megatron.training.models.gpt":
            raise

        # Megatron-Core 0.18.x predates megatron.training.models.gpt.
        from megatron.bridge.compat.mcore_gpt_fallback import GPTModelBuilder, GPTModelConfig, mtp_block_spec

    return GPTModelBuilder, GPTModelConfig, mtp_block_spec


GPTModelBuilder, _GPTModelConfig, mtp_block_spec = _load_gpt_symbols()


_STABLE_CONFIG_TARGET = "megatron.bridge.compat.mcore_gpt.GPTModelConfig"
_STABLE_BUILDER_TARGET = "megatron.bridge.compat.mcore_gpt.GPTModelBuilder"
_UPSTREAM_CONFIG_TARGET = "megatron.training.models.gpt.GPTModelConfig"
_UPSTREAM_BUILDER_TARGET = "megatron.training.models.gpt.GPTModelBuilder"


class GPTModelConfig(_GPTModelConfig):
    """GPT config with a serialized identity that is stable across MCore releases."""

    builder: ClassVar[str] = _STABLE_BUILDER_TARGET


def normalize_gpt_config_targets(data: dict[str, Any]) -> dict[str, Any]:
    """Map pre-facade MCore GPT serialization targets to their stable Bridge paths."""
    normalized = data.copy()
    if normalized.get("_target_") == _UPSTREAM_CONFIG_TARGET:
        normalized["_target_"] = _STABLE_CONFIG_TARGET
    if normalized.get("_builder_") == _UPSTREAM_BUILDER_TARGET:
        normalized["_builder_"] = _STABLE_BUILDER_TARGET
    return normalized


__all__ = ["GPTModelBuilder", "GPTModelConfig", "mtp_block_spec", "normalize_gpt_config_targets"]
