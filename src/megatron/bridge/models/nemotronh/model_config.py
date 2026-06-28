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

from dataclasses import dataclass, field
from typing import Any

from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.training.models.hybrid import HybridModelConfig

from megatron.bridge.models.gpt.model_config import ACTIVATION_FUNC_METADATA_KEY
from megatron.bridge.utils.activation_map import callable_to_str


@dataclass(kw_only=True)
class NemotronHModelConfig(HybridModelConfig):
    """Builder config that preserves Nemotron-H activation and MTP pattern."""

    mtp_hybrid_override_pattern: str | None = None
    keep_mtp_spec_in_bf16: bool = False
    extra_checkpoint_metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Normalize the MTP pattern before the upstream hybrid builder runs."""
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


__all__ = ["NemotronHModelConfig"]
