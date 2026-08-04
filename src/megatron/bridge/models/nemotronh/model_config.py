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

"""Serializable pure model configuration for Nemotron-H."""

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.transformer import ModuleSpec
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig

from megatron.bridge.models.config_proxy import FlatTransformerConfigMixin
from megatron.bridge.models.gpt.model_config import ACTIVATION_FUNC_METADATA_KEY
from megatron.bridge.models.hybrid.hybrid_builder import get_default_hybrid_stack_spec
from megatron.bridge.utils.activation_map import callable_to_str


DEFAULT_MAMBA_CHUNK_SIZE = 128


def _configure_mamba_chunk_size(stack_spec: ModuleSpec, chunk_size: int) -> ModuleSpec:
    """Return a stack spec whose Mamba mixer uses the requested scan chunk size."""
    if chunk_size == DEFAULT_MAMBA_CHUNK_SIZE:
        return stack_spec

    configured_spec = copy.deepcopy(stack_spec)
    mixer_spec = configured_spec.submodules.mamba_layer.submodules.mixer
    mixer_spec.params = {**mixer_spec.params, "chunk_size": chunk_size}
    return configured_spec


@dataclass(kw_only=True)
class NemotronHModelConfig(FlatTransformerConfigMixin, HybridModelConfig):
    """Builder config that preserves Nemotron-H activation and MTP pattern."""

    builder: ClassVar[str] = "megatron.bridge.models.nemotronh.model_config.NemotronHModelBuilder"
    mamba_chunk_size: int = DEFAULT_MAMBA_CHUNK_SIZE
    mtp_hybrid_override_pattern: str | None = None
    keep_mtp_spec_in_bf16: bool = False
    extra_checkpoint_metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Normalize the MTP pattern before the upstream hybrid builder runs."""
        if self.mamba_chunk_size < 1:
            raise ValueError("mamba_chunk_size must be at least 1.")
        if self.mtp_hybrid_override_pattern and self.mtp_num_layers:
            separator = Symbols.MTP_SEPARATOR
            main_pattern = (self.hybrid_layer_pattern or "").split(separator)[0]
            self.hybrid_layer_pattern = (
                main_pattern + separator + separator.join([self.mtp_hybrid_override_pattern] * self.mtp_num_layers)
            )
        super().finalize()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the nested activation as stable metadata."""
        data = super().as_dict()
        activation_name = callable_to_str(self.transformer.activation_func)
        if activation_name is None:
            raise ValueError("Cannot serialize an unregistered Nemotron-H activation callable.")
        metadata = dict(data.get("extra_checkpoint_metadata") or {})
        metadata[ACTIVATION_FUNC_METADATA_KEY] = activation_name
        data["extra_checkpoint_metadata"] = metadata
        return data


class NemotronHModelBuilder(HybridModelBuilder):
    """Build Nemotron-H while preserving its configurable Mamba scan chunk size."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build a Hybrid model with the configured Mamba mixer chunk size."""
        original_stack_spec = self._model_config.hybrid_stack_spec
        stack_spec = original_stack_spec or get_default_hybrid_stack_spec(self._model_config)
        self._model_config.hybrid_stack_spec = _configure_mamba_chunk_size(
            stack_spec,
            self._model_config.mamba_chunk_size,
        )
        try:
            return super().build_model(pg_collection, pre_process, post_process, vp_stage)
        finally:
            self._model_config.hybrid_stack_spec = original_stack_spec


__all__ = [
    "DEFAULT_MAMBA_CHUNK_SIZE",
    "NemotronHModelBuilder",
    "NemotronHModelConfig",
]
