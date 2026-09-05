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

"""Hugging Face to MCore checkpoint conversion for BAGEL-7B-MoT."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from megatron.core.models.bagel.bagel_mimo import BagelMimoModel
from transformers import AutoConfig, PretrainedConfig

from megatron.bridge.models.bagel.checkpoint import initialize_bagel_from_native_checkpoint
from megatron.bridge.models.bagel.provider import BagelModelProvider
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


logger = logging.getLogger(__name__)


class BagelConfig(PretrainedConfig):
    """Config-only representation of the native BAGEL Hugging Face assets."""

    model_type = "bagel"
    architectures = ["BagelForConditionalGeneration"]

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        vit_config: dict[str, Any] | None = None,
        vae_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.llm_config = llm_config or {}
        self.vit_config = vit_config or {}
        self.vae_config = vae_config or {}
        super().__init__(**kwargs)


AutoConfig.register("bagel", BagelConfig, exist_ok=True)


@MegatronModelBridge.register_bridge(
    source="BagelForConditionalGeneration",
    target=BagelMimoModel,
    provider=BagelModelProvider,
    model_type="bagel",
)
class BagelBridge(MegatronModelBridge):
    """Import official BAGEL tensors with the strict native checkpoint mapper."""

    SUPPORTS_HF_PRETRAINED_EXPORT = False
    MODEL_CONFIG_CLASS = None

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> BagelModelProvider:
        """Build the BAGEL provider from the nested official configuration."""
        config = hf_pretrained.config
        llm = config.llm_config
        return BagelModelProvider(
            num_layers=llm["num_hidden_layers"],
            hidden_size=llm["hidden_size"],
            num_attention_heads=llm["num_attention_heads"],
            num_query_groups=llm["num_key_value_heads"],
            ffn_hidden_size=llm["intermediate_size"],
            max_position_embeddings=llm["max_position_embeddings"],
            vocab_size=llm["vocab_size"],
            rms_norm_eps=llm["rms_norm_eps"],
            layernorm_epsilon=llm["rms_norm_eps"],
            rotary_base=llm["rope_theta"],
            attention_dropout=llm["attention_dropout"],
            official_config_values=config.to_dict(),
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return an empty registry because import uses the native mapper."""
        return MegatronMappingRegistry()

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained: PreTrainedCausalLM,
        megatron_model,
        allowed_mismatched_params: list[str] | None = None,
    ) -> list:
        """Load the complete native BAGEL state before distributed wrapping."""
        if allowed_mismatched_params:
            raise ValueError("BAGEL conversion does not allow mismatched parameters")
        models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        if len(models) != 1:
            raise ValueError("BAGEL conversion requires one unsharded model chunk")
        source = hf_pretrained.state.source
        if not isinstance(source, SafeTensorsStateSource):
            raise TypeError("BAGEL conversion requires a safetensors state source")
        # Official EMA omits export seed/world metadata; exact tensor coverage remains enforced.
        report = initialize_bagel_from_native_checkpoint(
            models[0],
            str(source.path / "ema.safetensors"),
            expected_model_seed=0,
            expected_world_size=0,
            validate_metadata=False,
            llm_config=SimpleNamespace(**hf_pretrained.config.llm_config),
        )
        logger.info(
            "Loaded native BAGEL state: source=%d target=%d fp32_main=%d",
            report.source_tensors_consumed,
            report.target_tensors_verified,
            report.fp32_main_tensors_preserved,
        )
        return models
