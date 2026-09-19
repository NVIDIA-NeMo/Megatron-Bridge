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

"""Thin legacy MBridge facade backed by the current conversion stack."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Unpack, cast

import torch
from megatron.core.transformer.module import MegatronModule
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge as CurrentAutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
from megatron.bridge.models.model_provider import GetModelKwargs, ModelProviderMixin


class AutoBridge:
    """Compatibility facade for the bounded legacy MBridge API.

    The facade keeps the legacy list-of-models contract while delegating model
    construction, conversion, and saving to the current Megatron Bridge
    implementation.
    """

    def __init__(
        self,
        bridge: CurrentAutoBridge[MegatronModule],
        provider: ModelProviderMixin[MegatronModule],
    ) -> None:
        self._bridge = bridge
        self._provider = provider

    @classmethod
    def from_config(cls, hf_config: PretrainedConfig, **provider_overrides: object) -> AutoBridge:
        """Create the legacy facade from a Hugging Face configuration.

        Args:
            hf_config: Supported Hugging Face model configuration.
            **provider_overrides: Current provider fields to override. The
                legacy ``dtype`` keyword is accepted as an alias for the
                provider precision fields.

        Returns:
            A facade wrapping the current :class:`AutoBridge`.

        Raises:
            AttributeError: If an override is not a real provider field.
            TypeError: If ``dtype`` is not a ``torch.dtype``.
        """
        current_bridge = cast(
            CurrentAutoBridge[MegatronModule],
            CurrentAutoBridge.from_hf_config(hf_config),
        )
        provider = cast(
            ModelProviderMixin[MegatronModule],
            current_bridge.to_megatron_provider(load_weights=False),
        )
        facade = cls(current_bridge, provider)

        overrides = dict(provider_overrides)
        dtype = overrides.pop("dtype", None)
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}.")
        facade._apply_provider_overrides(dtype=dtype, overrides=overrides)
        return facade

    def get_model(
        self,
        weight_path: str | Path | None = None,
        **kwargs: Unpack[GetModelKwargs],
    ) -> list[MegatronModule]:
        """Build a list of Megatron model chunks and optionally load HF weights.

        Args:
            weight_path: Optional Hugging Face checkpoint path or model ID.
            **kwargs: Current provider model-construction options.

        Returns:
            The legacy list of Megatron model chunks.
        """
        self._provider.finalize()
        models = cast(list[MegatronModule], self._provider.provide_distributed_model(**kwargs))
        if weight_path is not None:
            self.load_weights(models, weight_path)
        return models

    def load_weights(
        self,
        models: list[MegatronModule],
        weights_path: str | Path,
        memory_efficient: bool = False,
    ) -> None:
        """Load Hugging Face weights into legacy-shaped model chunks.

        Args:
            models: List of Megatron model chunks.
            weights_path: Hugging Face checkpoint path or model ID.
            memory_efficient: Legacy loading mode. ``True`` is not representable
                by the current API.

        Raises:
            NotImplementedError: If ``memory_efficient`` is enabled.
            TypeError: If ``models`` is not a list.
        """
        self._require_model_list(models)
        if memory_efficient:
            raise NotImplementedError(
                "memory_efficient=True is not representable by the current AutoBridge loading API. "
                "Use memory_efficient=False; the current checkpoint source manages weight materialization."
            )
        self._bridge.load_hf_weights(models, hf_path=weights_path)

    def export_weights(
        self,
        models: list[MegatronModule],
        keep_stacked_experts: bool = True,
    ) -> Iterable[HFWeightTuple]:
        """Export legacy-shaped model chunks as Hugging Face weights.

        Args:
            models: List of Megatron model chunks.
            keep_stacked_experts: Legacy expert-layout selector. The current API
                supports only its source-native layout, represented by ``True``.

        Returns:
            An iterable of Hugging Face weight tuples.

        Raises:
            NotImplementedError: If separate expert output is requested.
            TypeError: If ``models`` is not a list.
        """
        self._require_model_list(models)
        if not keep_stacked_experts:
            raise NotImplementedError(
                "keep_stacked_experts=False is not representable by the current AutoBridge export API. "
                "Use keep_stacked_experts=True to preserve the current bridge's source-native expert layout."
            )
        return self._bridge.export_hf_weights(models)

    def save_weights(
        self,
        models: list[MegatronModule],
        weights_path: str | Path,
        memory_efficient: bool = False,
    ) -> None:
        """Save model chunks through the current Hugging Face save path.

        Args:
            models: List of Megatron model chunks.
            weights_path: Output directory.
            memory_efficient: Legacy save mode. ``True`` is not representable
                by the current API.

        Raises:
            NotImplementedError: If ``memory_efficient`` is enabled.
            TypeError: If ``models`` is not a list.
        """
        self._require_model_list(models)
        if memory_efficient:
            raise NotImplementedError(
                "memory_efficient=True is not representable by the current AutoBridge save API. "
                "Use memory_efficient=False; the current save path owns streaming and sharding."
            )
        self._bridge.save_hf_pretrained(models, weights_path)

    def set_extra_args(self, **provider_overrides: object) -> None:
        """Apply validated current-provider overrides and rebuild provider state.

        Args:
            **provider_overrides: Fields that already exist on the current
                provider/config object.

        Raises:
            AttributeError: If any override is not a real provider field. No
                overrides are applied when validation fails.
        """
        self._apply_provider_overrides(dtype=None, overrides=provider_overrides)

    def _apply_provider_overrides(
        self,
        *,
        dtype: torch.dtype | None,
        overrides: Mapping[str, object],
    ) -> None:
        fields_to_validate = set(overrides)
        if dtype is not None:
            fields_to_validate.update(("params_dtype", "fp16", "bf16"))
        unknown_fields = sorted(name for name in fields_to_validate if not hasattr(self._provider, name))
        if unknown_fields:
            formatted_fields = ", ".join(repr(name) for name in unknown_fields)
            raise AttributeError(
                f"{type(self._provider).__name__} has no provider/config field(s): {formatted_fields}. "
                "Only existing current-provider fields can be passed to the legacy facade."
            )
        self._provider.apply_overrides_and_finalize(dtype=dtype, overrides=overrides)

    @staticmethod
    def _require_model_list(models: object) -> None:
        if not isinstance(models, list):
            raise TypeError("Legacy MBridge APIs require models to be provided as a list of model chunks.")
