# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""MegatronMIMO HF<->Megatron weight conversion orchestrator."""

from __future__ import annotations

import contextlib
import copy
import logging
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.models.mimo import MimoModel
    from megatron.core.transformer.spec_utils import ModuleSpec

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
        MegatronMIMOInfra,
        MegatronMIMOProvider,
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MIMOComponent:
    """One MIMO component route in the source bridge mapping registry."""

    name: str
    source_prefix: str
    target_module_path: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MIMOComponent.name must be a non-empty string")
        if not self.source_prefix:
            raise ValueError(f"MIMOComponent[{self.name!r}].source_prefix must be a non-empty string")
        if not self.source_prefix.endswith("."):
            raise ValueError(
                f"MIMOComponent[{self.name!r}].source_prefix must end with '.', got {self.source_prefix!r}"
            )
        if not self.target_module_path:
            raise ValueError(f"MIMOComponent[{self.name!r}].target_module_path must be a non-empty string")


def validate_route_table(
    routes: list[MIMOComponent],
    *,
    parallelism_config: "MegatronMIMOParallelismConfig",
    modality_submodules_spec: Optional[dict[str, "ModuleSpec"]] = None,
) -> None:
    """Validate component routes against the MIMO parallelism config."""
    if not routes:
        raise ValueError("Route table must contain at least one MIMOComponent")

    names = [route.name for route in routes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate route names: {duplicates}")

    declared_names = set(names)
    config_names = set(parallelism_config.module_parallelisms.keys())

    extra_routes = sorted(declared_names - config_names)
    if extra_routes:
        raise ValueError(
            "Route names not present in parallelism_config.module_parallelisms: "
            f"{extra_routes}. Available: {sorted(config_names)}"
        )

    missing_routes = sorted(config_names - declared_names)
    if missing_routes:
        raise ValueError(
            "parallelism_config.module_parallelisms entries without a route: "
            f"{missing_routes}. Declared routes: {sorted(declared_names)}"
        )

    if modality_submodules_spec is not None:
        from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

        expected_names = set(modality_submodules_spec.keys()) | {MIMO_LANGUAGE_MODULE_KEY}
        extra_vs_modality = sorted(declared_names - expected_names)
        missing_vs_modality = sorted(expected_names - declared_names)
        if extra_vs_modality or missing_vs_modality:
            raise ValueError(
                "Route names do not align with modality_submodules_spec keys + "
                f"{{{MIMO_LANGUAGE_MODULE_KEY!r}}}. Routes: {sorted(declared_names)}; "
                f"expected: {sorted(expected_names)}. "
                f"Extra in routes: {extra_vs_modality}; missing from routes: {missing_vs_modality}. "
                "Route names, modality keys, and parallelism-config component keys must match."
            )

    _check_no_prefix_overlap(routes)


def _check_no_prefix_overlap(routes: list[MIMOComponent]) -> None:
    prefixes = [(route.source_prefix, route.name) for route in routes]
    for i, (prefix_a, name_a) in enumerate(prefixes):
        for prefix_b, name_b in prefixes[i + 1 :]:
            if prefix_a == prefix_b:
                raise ValueError(f"Routes {name_a!r} and {name_b!r} share source_prefix {prefix_a!r}")
            shorter, longer = sorted((prefix_a, prefix_b), key=len)
            if longer.startswith(shorter):
                raise ValueError(
                    f"Route source_prefix {longer!r} (route {name_b!r} or {name_a!r}) "
                    f"nests inside {shorter!r}. Routes must use disjoint prefixes."
                )


MIMOConversionSpecBuilder = Callable[
    ["MegatronModelBridge", Any, "MegatronMIMOParallelismConfig"],
    tuple[Any, list[MIMOComponent]],
]


_CONVERSION_SPECS: dict[type, MIMOConversionSpecBuilder] = {}


def register_mimo_conversion_spec(
    source_bridge_class: type,
) -> Callable[[MIMOConversionSpecBuilder], MIMOConversionSpecBuilder]:
    """Register a MIMO conversion spec builder for a standard bridge class."""

    def _decorator(conversion_spec: MIMOConversionSpecBuilder) -> MIMOConversionSpecBuilder:
        if source_bridge_class in _CONVERSION_SPECS:
            existing = _CONVERSION_SPECS[source_bridge_class]
            if (
                existing.__module__ == conversion_spec.__module__
                and existing.__qualname__ == conversion_spec.__qualname__
            ):
                _CONVERSION_SPECS[source_bridge_class] = conversion_spec
                return conversion_spec
            raise ValueError(
                f"MIMO conversion spec already registered for {source_bridge_class.__name__}: "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _CONVERSION_SPECS[source_bridge_class] = conversion_spec
        return conversion_spec

    return _decorator


def _build_default_mimo_provider(
    source_bridge: "MegatronModelBridge",
    hf_pretrained: Any,
    parallelism_config: "MegatronMIMOParallelismConfig",
) -> "MegatronMIMOProvider":
    """Build a MIMO provider from the source bridge's standard provider."""
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider

    standard_provider = source_bridge.provider_bridge(hf_pretrained)
    return MegatronMIMOProvider.from_standard_provider(
        standard_provider=standard_provider,
        megatron_mimo_parallelism_config=parallelism_config,
    )


def _build_default_mimo_conversion_spec(
    source_bridge: "MegatronModelBridge",
    hf_pretrained: Any,
    parallelism_config: "MegatronMIMOParallelismConfig",
) -> tuple["MegatronMIMOProvider", list[MIMOComponent]]:
    """Build MIMO provider/routes from standard bridge/provider metadata."""
    mimo_provider = _build_default_mimo_provider(source_bridge, hf_pretrained, parallelism_config)
    standard_provider = mimo_provider.standard_provider
    if standard_provider is None:
        raise TypeError("Default MIMO conversion requires a standard_provider.")
    return mimo_provider, _build_default_mimo_routes(source_bridge, standard_provider)


def _build_default_mimo_routes(source_bridge: "MegatronModelBridge", standard_provider: object) -> list[MIMOComponent]:
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    source_prefixes = getattr(source_bridge, "mimo_source_prefixes", None)
    modality_keys = getattr(standard_provider, "modality_keys", None)
    if source_prefixes is None or modality_keys is None:
        raise TypeError(
            "Default MIMO conversion requires source_bridge.mimo_source_prefixes and standard_provider.modality_keys."
        )

    source_prefixes = dict(source_prefixes)
    modality_keys = dict(modality_keys)
    component_names = {MIMO_LANGUAGE_MODULE_KEY, *modality_keys.keys()}
    source_prefix_names = set(source_prefixes)
    missing = sorted(component_names - source_prefix_names)
    extra = sorted(source_prefix_names - component_names)
    if missing or extra:
        raise ValueError(
            "mimo_source_prefixes keys must match the language component plus provider modality keys. "
            f"Missing: {missing}; extra: {extra}."
        )

    routes = [
        MIMOComponent(
            name=MIMO_LANGUAGE_MODULE_KEY,
            source_prefix=source_prefixes[MIMO_LANGUAGE_MODULE_KEY],
            target_module_path="language_model",
        )
    ]
    routes.extend(
        MIMOComponent(
            name=modality_name,
            source_prefix=source_prefixes[modality_name],
            target_module_path=f"modality_submodules.{modality_name}.encoders.{encoder_key}",
        )
        for modality_name, encoder_key in modality_keys.items()
    )
    return routes


def get_mimo_conversion_spec(source_bridge_class: type) -> MIMOConversionSpecBuilder:
    """Resolve an explicit or metadata-derived MIMO conversion spec builder."""
    try:
        return _CONVERSION_SPECS[source_bridge_class]
    except KeyError as exc:
        if bool(getattr(source_bridge_class, "mimo_source_prefixes", None)):
            return _build_default_mimo_conversion_spec
        registered = sorted(cls.__name__ for cls in _CONVERSION_SPECS)
        raise KeyError(
            f"No MIMO conversion spec registered for {source_bridge_class.__name__}, and the class does not define "
            f"mimo_source_prefixes for default route construction. Registered: {registered}"
        ) from exc


def supports_mimo_conversion(source_bridge_class: type) -> bool:
    """Return whether a standard bridge advertises MIMO conversion support."""
    return source_bridge_class in _CONVERSION_SPECS or bool(getattr(source_bridge_class, "mimo_source_prefixes", None))


def _reset_registry_for_tests() -> None:
    """Clear all registered conversion specs. Test-only helper."""
    _CONVERSION_SPECS.clear()


def build_route_local_registry(
    source_registry: MegatronMappingRegistry,
    route: MIMOComponent,
) -> MegatronMappingRegistry:
    """Filter and prefix-strip a source mapping registry for one MIMO route."""
    prefix = route.source_prefix
    stripped: list[MegatronParamMapping] = []
    for mapping in source_registry.mappings:
        if not mapping.megatron_param.startswith(prefix):
            continue
        cloned = copy.copy(mapping)
        cloned.megatron_param = mapping.megatron_param[len(prefix) :]
        stripped.append(cloned)
    return MegatronMappingRegistry(*stripped)


def make_route_local_bridge(
    source_bridge: "MegatronModelBridge",
    route: MIMOComponent,
    *,
    route_local_registry: MegatronMappingRegistry | None = None,
) -> "MegatronModelBridge":
    """Clone a source bridge and override its registry for one route."""
    if route_local_registry is None:
        route_local_registry = build_route_local_registry(source_bridge.mapping_registry(), route)

    wrapper = copy.copy(source_bridge)

    def _mapping_registry(self: "MegatronModelBridge") -> MegatronMappingRegistry:
        return route_local_registry

    wrapper.mapping_registry = types.MethodType(_mapping_registry, wrapper)
    return wrapper


@dataclass(frozen=True)
class MIMOConversionTask:
    """A standard conversion task annotated with its MIMO route."""

    route: MIMOComponent
    task: "WeightConversionTask"


class MegatronMIMOBridge(AutoBridge):
    """AutoBridge subclass for MegatronMIMO checkpoint conversion."""

    def __init__(
        self,
        hf_pretrained: PreTrainedCausalLM | PretrainedConfig,
        *,
        parallelism_config: "MegatronMIMOParallelismConfig",
        source_bridge: Optional["MegatronModelBridge"] = None,
    ) -> None:
        super().__init__(hf_pretrained)
        self.parallelism_config = parallelism_config
        self._source_bridge_override = source_bridge
        self._mimo_provider: Optional["MegatronMIMOProvider"] = None
        self._routes: Optional[list[MIMOComponent]] = None
        self._infra: Optional["MegatronMIMOInfra"] = None

    @classmethod
    def from_bridge(
        cls,
        bridge: AutoBridge,
        *,
        parallelism_config: "MegatronMIMOParallelismConfig",
    ) -> "MegatronMIMOBridge":
        """Create a MIMO bridge from a resolved standard bridge."""
        mimo_bridge = cls(
            bridge.hf_pretrained,
            parallelism_config=parallelism_config,
            source_bridge=bridge._model_bridge,
        )
        mimo_bridge.export_weight_dtype = bridge.export_weight_dtype
        mimo_bridge.hf_model_id = bridge.hf_model_id
        return mimo_bridge

    @classmethod
    def from_hf_pretrained(
        cls,
        path: Union[str, Path],
        *,
        parallelism_config: "MegatronMIMOParallelismConfig",
        **kwargs,
    ) -> "MegatronMIMOBridge":
        """Resolve the standard bridge from HF, then wrap it for MIMO."""
        bridge = AutoBridge.from_hf_pretrained(path, **kwargs)
        return cls.from_bridge(bridge, parallelism_config=parallelism_config)

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> "MegatronMIMOBridge":
        raise NotImplementedError("MegatronMIMOBridge does not support config-only construction yet.")

    @classmethod
    def from_auto_config(cls, megatron_path: str, hf_model_id: str, trust_remote_code: bool = False) -> AutoBridge:
        raise NotImplementedError("MegatronMIMOBridge does not support config-only checkpoint export yet.")

    @property
    def _model_bridge(self) -> "MegatronModelBridge":
        if self._source_bridge_override is not None:
            self._source_bridge_override.export_weight_dtype = self.export_weight_dtype
            return self._source_bridge_override
        return super()._model_bridge

    @property
    def routes(self) -> list[MIMOComponent]:
        """Return the route table resolved for this source bridge."""
        if self._routes is None:
            self.to_megatron_mimo_provider(load_weights=False)
        assert self._routes is not None
        return self._routes

    def to_megatron_provider(
        self,
        load_weights: bool = False,
        hf_path: str | Path | None = None,
    ) -> "MegatronMIMOProvider":
        """Use to_megatron_mimo_provider() for MegatronMIMO conversion."""
        raise NotImplementedError("MegatronMIMOBridge uses to_megatron_mimo_provider(), not to_megatron_provider().")

    def to_megatron_mimo_provider(
        self,
        load_weights: bool = False,
        hf_path: str | Path | None = None,
    ) -> "MegatronMIMOProvider":
        """Build the MIMO provider and route table for this HF source."""
        if hf_path is not None:
            raise NotImplementedError("MegatronMIMOBridge.to_megatron_mimo_provider does not support hf_path yet.")
        if load_weights:
            raise NotImplementedError("Use to_megatron_model(load_weights=True) for MegatronMIMO weight loading.")

        if self._mimo_provider is not None and self._routes is not None:
            return self._mimo_provider

        source_bridge = self._model_bridge
        conversion_spec = get_mimo_conversion_spec(type(source_bridge))
        mimo_provider, routes = conversion_spec(source_bridge, self.hf_pretrained, self.parallelism_config)
        validate_route_table(
            routes,
            parallelism_config=self.parallelism_config,
            modality_submodules_spec=mimo_provider.modality_submodules_spec,
        )
        self._mimo_provider = mimo_provider
        self._routes = routes
        return mimo_provider

    def validate_mimo_conversion_support(self) -> None:
        """Validate MIMO conversion support by resolving the real provider and routes."""
        self.to_megatron_mimo_provider(load_weights=False)

    def to_megatron_model(
        self,
        load_weights: bool = True,
        hf_path: str | Path | None = None,
        **kwargs,
    ) -> list[MegatronModule]:
        """Build a distributed MIMO model and optionally import HF weights."""
        mimo_provider = self.to_megatron_mimo_provider(load_weights=False)
        mimo_model = self.build_mimo_model(mimo_provider=mimo_provider, **kwargs)
        if load_weights:
            self.load_hf_weights(mimo_model, hf_path=hf_path)
        return [mimo_model]

    def build_mimo_model(
        self,
        *,
        mimo_provider: Optional["MegatronMIMOProvider"] = None,
        ddp_config: Optional["DistributedDataParallelConfig"] = None,
        fp16: bool = False,
        bf16: bool = True,
        seed: int = 0,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
    ) -> "MimoModel":
        """Build the MIMO model and cache its infrastructure."""
        from megatron.bridge.models.megatron_mimo import build_megatron_mimo_model

        mimo_provider = mimo_provider or self.to_megatron_mimo_provider(load_weights=False)
        mimo_model, infra = build_megatron_mimo_model(
            mimo_provider,
            ddp_config=ddp_config,
            fp16=fp16,
            bf16=bf16,
            seed=seed,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
        )
        self._mimo_provider = mimo_provider
        self._infra = infra
        return mimo_model

    def load_hf_weights(
        self,
        model: "MimoModel" | list["MimoModel"],
        hf_path: str | Path | None = None,
        allowed_mismatched_params: list[str] | None = None,
        *,
        infra: Optional["MegatronMIMOInfra"] = None,
    ) -> "MimoModel":
        """Load HF weights into a constructed MegatronMIMO model."""
        mimo_model = self._coerce_mimo_model(model)
        infra = self._require_infra(infra)
        hf_pretrained = self._resolve_hf_pretrained(hf_path)
        import_hf_to_megatron_mimo(
            source_bridge=self._model_bridge,
            hf_pretrained=hf_pretrained,
            mimo_model=mimo_model,
            routes=self.routes,
            pg_collections=infra.pg_collections,
            allowed_mismatched_params=allowed_mismatched_params,
        )
        return mimo_model

    def export_hf_weights(
        self,
        model: "MimoModel" | list["MimoModel"],
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: dict[str, list["WeightConversionTask"]] | None = None,
        merge_adapter_weights: bool = True,
        *,
        infra: Optional["MegatronMIMOInfra"] = None,
    ) -> Iterable[HFWeightTuple]:
        """Export MIMO weights as a rank-0 HF tensor stream."""
        mimo_model = self._coerce_mimo_model(model)
        infra = self._require_infra(infra)
        if conversion_tasks is not None:
            raise NotImplementedError("Custom MIMO export conversion tasks are not wired into rank-0 streaming yet.")
        if not merge_adapter_weights:
            raise NotImplementedError("MegatronMIMO export without adapter merging is not implemented yet.")
        yield from _stream_mimo_weights_to_rank0(
            source_bridge=self._model_bridge,
            hf_pretrained=self.hf_pretrained,
            mimo_model=mimo_model,
            routes=self.routes,
            pg_collections=infra.pg_collections,
            show_progress=show_progress,
        )

    def get_conversion_tasks(
        self,
        megatron_model: "MimoModel" | list["MimoModel"],
        hf_path: str | Path | None = None,
        *,
        infra: Optional["MegatronMIMOInfra"] = None,
    ) -> list[MIMOConversionTask]:
        """Return route-annotated conversion tasks for active MIMO components."""
        mimo_model = self._coerce_mimo_model(megatron_model)
        infra = self._require_infra(infra)
        hf_pretrained = self._resolve_hf_pretrained(hf_path)

        tasks: list[MIMOConversionTask] = []
        for route, pg_collection in _iter_active_routes(self.routes, infra.pg_collections):
            submodule = mimo_model.get_submodule(route.target_module_path)
            wrapped = make_route_local_bridge(self._model_bridge, route)
            with _bridged_parallel_state(pg_collection), component_pg_context(submodule, pg_collection):
                for task in wrapped.build_conversion_tasks(hf_pretrained, [submodule]):
                    if task is not None:
                        tasks.append(MIMOConversionTask(route=route, task=task))
        return tasks

    def save_hf_pretrained(
        self,
        model: "MimoModel" | list["MimoModel"],
        path: str | Path,
        show_progress: bool = True,
        source_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
        *,
        infra: Optional["MegatronMIMOInfra"] = None,
    ) -> None:
        """Save a MegatronMIMO model in HuggingFace format."""
        mimo_model = self._coerce_mimo_model(model)
        infra = self._require_infra(infra)
        save_hf_pretrained_mimo(
            self,
            mimo_model,
            routes=self.routes,
            pg_collections=infra.pg_collections,
            path=path,
            source_path=source_path,
            strict=strict,
            show_progress=show_progress,
        )

    def save_megatron_model(
        self,
        model: "MimoModel" | list["MimoModel"],
        path: str | Path,
        hf_tokenizer_path: Optional[str | Path] = None,
        low_memory_save: bool = False,
        hf_tokenizer_kwargs: Optional[dict] = None,
        *,
        infra: Optional["MegatronMIMOInfra"] = None,
        mimo_provider: Optional["MegatronMIMOProvider"] = None,
    ) -> None:
        """Save a MegatronMIMO checkpoint."""
        from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import save_megatron_mimo_model

        if low_memory_save:
            raise NotImplementedError("MegatronMIMO checkpoint save does not support low_memory_save.")
        mimo_model = self._coerce_mimo_model(model)
        infra = self._require_infra(infra)
        mimo_provider = mimo_provider or self._require_provider()
        save_megatron_mimo_model(
            mimo_model,
            infra,
            mimo_provider,
            path,
            hf_tokenizer_path=hf_tokenizer_path,
            hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        )

    def load_megatron_model(
        self,
        path: str | Path,
        *,
        parallelism_config: Optional["MegatronMIMOParallelismConfig"] = None,
        ddp_config: Optional["DistributedDataParallelConfig"] = None,
        fp16: bool = False,
        bf16: bool = True,
        wrap_with_ddp: bool = False,
        data_parallel_random_init: bool = False,
    ) -> "MimoModel":
        """Load a MegatronMIMO checkpoint and cache its provider/infra."""
        from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import load_megatron_mimo_model

        self.to_megatron_mimo_provider(load_weights=False)
        mimo_model, infra, mimo_provider = load_megatron_mimo_model(
            path,
            parallelism_config=parallelism_config or self.parallelism_config,
            ddp_config=ddp_config,
            fp16=fp16,
            bf16=bf16,
            wrap_with_ddp=wrap_with_ddp,
            data_parallel_random_init=data_parallel_random_init,
        )
        self._mimo_provider = mimo_provider
        self._infra = infra
        return mimo_model

    def import_ckpt(
        self,
        megatron_path: str | Path,
        *,
        hf_tokenizer_path: Optional[str | Path] = None,
        hf_tokenizer_kwargs: Optional[dict] = None,
    ) -> None:
        """Import HF weights and write a MegatronMIMO checkpoint."""
        model = self.to_megatron_model(
            load_weights=True,
            wrap_with_ddp=False,
            data_parallel_random_init=False,
        )[0]
        self.save_megatron_model(
            model,
            megatron_path,
            hf_tokenizer_path=hf_tokenizer_path or self._hf_identifier(),
            hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        )

    def export_ckpt(
        self,
        megatron_path: str | Path,
        hf_path: str | Path,
        show_progress: bool = True,
        strict: bool = False,
        source_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Load a MegatronMIMO checkpoint and export it to HuggingFace."""
        model = self.load_megatron_model(megatron_path)
        self.save_hf_pretrained(
            model,
            hf_path,
            show_progress=show_progress,
            source_path=source_path,
            strict=strict,
        )

    def export_adapter_weights(self, *args, **kwargs):
        raise NotImplementedError("MegatronMIMO adapter export is not implemented yet.")

    def save_hf_adapter(self, *args, **kwargs) -> None:
        raise NotImplementedError("MegatronMIMO adapter export is not implemented yet.")

    def export_adapter_ckpt(self, *args, **kwargs) -> None:
        raise NotImplementedError("MegatronMIMO adapter checkpoint export is not implemented yet.")

    def _resolve_hf_pretrained(self, hf_path: str | Path | None) -> Any:
        if hf_path is None:
            if not isinstance(self.hf_pretrained, PreTrainedCausalLM):
                raise ValueError("hf_path is required when hf_pretrained does not include weights.")
            return self.hf_pretrained
        trust_remote_code = getattr(self.hf_pretrained, "trust_remote_code", False)
        return PreTrainedCausalLM.from_pretrained(hf_path, trust_remote_code=trust_remote_code)

    def _require_infra(self, infra: Optional["MegatronMIMOInfra"] = None) -> "MegatronMIMOInfra":
        infra = infra or self._infra
        if infra is None:
            raise ValueError(
                "MegatronMIMO infrastructure is required. Build or load the model through "
                "this bridge, or pass infra= explicitly."
            )
        return infra

    def _require_provider(self) -> "MegatronMIMOProvider":
        if self._mimo_provider is None:
            raise ValueError("MegatronMIMO provider is required. Call to_megatron_mimo_provider() first.")
        return self._mimo_provider

    @staticmethod
    def _coerce_mimo_model(model: "MimoModel" | list["MimoModel"]) -> "MimoModel":
        if isinstance(model, list):
            if len(model) != 1:
                raise ValueError(f"MegatronMIMO expects a single MimoModel, got {len(model)} model chunks.")
            return model[0]
        return model

    def _hf_identifier(self) -> str | None:
        if self.hf_model_id:
            return str(self.hf_model_id)
        hf_name_or_path = getattr(self.hf_pretrained, "model_name_or_path", None)
        if hf_name_or_path is None:
            hf_name_or_path = getattr(self.hf_pretrained, "name_or_path", None)
        return str(hf_name_or_path) if hf_name_or_path else None


@contextlib.contextmanager
def _bridged_parallel_state(pg_collection: Any) -> Iterator[None]:
    """Temporarily set Megatron-Core ``parallel_state`` globals from a MIMO pg_collection.

    MIMO never initialises the MCore parallel_state globals, but the standard
    bridge reads them directly. This context bridges per-route groups in and
    restores them on exit.
    """
    from megatron.core import parallel_state as mpu

    bridge_map = {
        "_TENSOR_MODEL_PARALLEL_GROUP": "tp",
        "_DATA_PARALLEL_GROUP": "dp",
        "_DATA_PARALLEL_GROUP_WITH_CP": "dp_cp",
        "_PIPELINE_MODEL_PARALLEL_GROUP": "pp",
        "_CONTEXT_PARALLEL_GROUP": "cp",
        "_EXPERT_MODEL_PARALLEL_GROUP": "ep",
        "_EXPERT_TENSOR_PARALLEL_GROUP": "expt_tp",
        "_MODEL_PARALLEL_GROUP": "mp",
        "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP": "tp_ep",
        "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP": "tp_ep_pp",
        "_EMBEDDING_GROUP": "embd",
        "_POSITION_EMBEDDING_GROUP": "pos_embd",
    }

    saved = {global_name: getattr(mpu, global_name, None) for global_name in bridge_map}
    try:
        for global_name, attr in bridge_map.items():
            value = getattr(pg_collection, attr, None)
            if value is not None:
                setattr(mpu, global_name, value)
        # Fallbacks for fields absent on dense components but still queried by
        # the standard bridge (e.g. ``get_expert_*``, ``dp_cp``).
        if mpu._DATA_PARALLEL_GROUP_WITH_CP is saved["_DATA_PARALLEL_GROUP_WITH_CP"]:
            mpu._DATA_PARALLEL_GROUP_WITH_CP = pg_collection.dp
        if mpu._EXPERT_TENSOR_PARALLEL_GROUP is saved["_EXPERT_TENSOR_PARALLEL_GROUP"]:
            mpu._EXPERT_TENSOR_PARALLEL_GROUP = pg_collection.tp
        if mpu._EXPERT_MODEL_PARALLEL_GROUP is saved["_EXPERT_MODEL_PARALLEL_GROUP"]:
            mpu._EXPERT_MODEL_PARALLEL_GROUP = pg_collection.tp
        if mpu._MODEL_PARALLEL_GROUP is saved["_MODEL_PARALLEL_GROUP"]:
            mpu._MODEL_PARALLEL_GROUP = pg_collection.tp
        if mpu._CONTEXT_PARALLEL_GROUP is saved["_CONTEXT_PARALLEL_GROUP"]:
            mpu._CONTEXT_PARALLEL_GROUP = pg_collection.tp
        yield
    finally:
        for global_name, value in saved.items():
            setattr(mpu, global_name, value)


@contextlib.contextmanager
def component_pg_context(module: nn.Module, pg_collection: Any) -> Iterator[None]:
    """Temporarily attach ``pg_collection`` to a module for the duration of conversion.

    If the module already carries a pg_collection (the normal MIMO-provider
    path), it is trusted and not overwritten. Otherwise the supplied
    pg_collection is attached and removed on exit.
    """
    existing = getattr(module, "pg_collection", None)
    if existing is not None:
        yield
        return

    module.pg_collection = pg_collection
    try:
        yield
    finally:
        # Use try/delattr so we always clean up, even if downstream code
        # mutated the attribute. delattr raises if missing — guard with hasattr.
        if hasattr(module, "pg_collection"):
            try:
                delattr(module, "pg_collection")
            except AttributeError:
                # ``module.pg_collection`` may be a property/descriptor on some
                # module classes (rare). Fall back to setting None so we do not
                # leave the route's group dangling on the module.
                module.pg_collection = None


def _iter_active_routes(
    routes: list[MIMOComponent],
    pg_collections: dict[str, Any],
) -> Iterator[tuple[MIMOComponent, Any]]:
    """Yield (route, pg_collection) pairs for components this rank owns.

    Skips any route whose ``pg_collections.get(route.name)`` is ``None``.
    Raises if a route name is missing from ``pg_collections`` entirely — that
    means the MIMO infra was built with a different component set than the
    route table declares.
    """
    for route in routes:
        if route.name not in pg_collections:
            raise KeyError(
                f"Route {route.name!r} is not present in MegatronMIMOInfra.pg_collections "
                f"(available: {sorted(pg_collections.keys())}). Route table and parallelism "
                f"config are out of sync."
            )
        pg_collection = pg_collections[route.name]
        if pg_collection is None:
            logger.debug("Skipping route %r on this rank: pg_collection is None", route.name)
            continue
        yield route, pg_collection


def import_hf_to_megatron_mimo(
    *,
    source_bridge: "MegatronModelBridge",
    hf_pretrained: Any,
    mimo_model: nn.Module,
    routes: list[MIMOComponent],
    pg_collections: dict[str, Any],
    allowed_mismatched_params: list[str] | None = None,
) -> nn.Module:
    """Import HF weights into a constructed MegatronMIMO model.

    Drives ``MegatronModelBridge.load_weights_hf_to_megatron`` once per
    active route with a prefix-stripped registry and the route's
    pg_collection. Returns ``mimo_model`` for convenience.
    """
    for route, pg_collection in _iter_active_routes(routes, pg_collections):
        submodule = mimo_model.get_submodule(route.target_module_path)
        wrapped = make_route_local_bridge(source_bridge, route)

        logger.info(
            "Importing HF weights into MIMO component %r (target_module_path=%r)",
            route.name,
            route.target_module_path,
        )
        with _bridged_parallel_state(pg_collection), component_pg_context(submodule, pg_collection):
            if allowed_mismatched_params is None:
                wrapped.load_weights_hf_to_megatron(hf_pretrained, submodule)
            else:
                wrapped.load_weights_hf_to_megatron(
                    hf_pretrained,
                    submodule,
                    allowed_mismatched_params=allowed_mismatched_params,
                )

    return mimo_model


def export_megatron_mimo_to_hf(
    *,
    source_bridge: "MegatronModelBridge",
    hf_pretrained: Any,
    mimo_model: nn.Module,
    routes: list[MIMOComponent],
    pg_collections: dict[str, Any],
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks: dict[str, list[Any]] | None = None,
    merge_adapter_weights: bool = True,
) -> Iterator["HFWeightTuple"]:
    """Export a MegatronMIMO model to HF format, yielding ``(name, tensor)`` pairs.

    Drives ``MegatronModelBridge.stream_weights_megatron_to_hf`` once per
    active route. HF names are unchanged from the source bridge — only the
    Megatron-side ``megatron_param`` is prefix-stripped, so routes produce
    disjoint subsets of the HF state dict.
    """
    for route, pg_collection in _iter_active_routes(routes, pg_collections):
        submodule = mimo_model.get_submodule(route.target_module_path)
        wrapped = make_route_local_bridge(source_bridge, route)

        logger.info(
            "Exporting MIMO component %r to HF (target_module_path=%r)",
            route.name,
            route.target_module_path,
        )
        with _bridged_parallel_state(pg_collection), component_pg_context(submodule, pg_collection):
            export_kwargs = {
                "cpu": cpu,
                "show_progress": show_progress,
            }
            if conversion_tasks is not None:
                export_kwargs["conversion_tasks"] = conversion_tasks.get(route.name)
            if not merge_adapter_weights:
                export_kwargs["merge_adapter_weights"] = merge_adapter_weights
            yield from wrapped.stream_weights_megatron_to_hf(submodule, hf_pretrained, **export_kwargs)


def save_hf_pretrained_mimo(
    bridge: AutoBridge,
    mimo_model: nn.Module,
    routes: list[MIMOComponent],
    pg_collections: dict[str, Any],
    path: Union[str, Path],
    *,
    source_path: Optional[Union[str, Path]] = None,
    strict: bool = False,
    show_progress: bool = True,
) -> None:
    """Save a MegatronMIMO model in HuggingFace format."""
    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

    if not _is_safetensors_source(bridge):
        raise ValueError(
            "save_hf_pretrained_mimo requires the source HF model to ship "
            "weights as safetensors. The source's state.source is "
            f"{type(getattr(bridge.hf_pretrained, 'state', None)).__name__}; "
            "pre-safetensors checkpoints are not supported."
        )

    output_path = Path(path)
    if dist.is_initialized():
        if dist.get_rank() == 0:
            _copy_hf_artifacts(bridge, output_path, source_path=source_path)
    else:
        _copy_hf_artifacts(bridge, output_path, source_path=source_path)

    if dist.is_initialized():
        dist.barrier()

    state_source = bridge.hf_pretrained.state.source
    assert isinstance(state_source, SafeTensorsStateSource), "checked above"
    state_source.save_generator(
        _stream_mimo_weights_to_rank0(
            source_bridge=bridge._model_bridge,
            hf_pretrained=bridge.hf_pretrained,
            mimo_model=mimo_model,
            routes=routes,
            pg_collections=pg_collections,
            show_progress=show_progress,
        ),
        output_path,
        strict=strict,
    )

    if dist.is_initialized():
        dist.barrier()

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        logger.info("save_hf_pretrained_mimo: wrote HF checkpoint to %s", output_path)


def _copy_hf_artifacts(
    bridge: AutoBridge,
    output_path: Path,
    *,
    source_path: Optional[Union[str, Path]] = None,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    additional_files = getattr(bridge._model_bridge, "ADDITIONAL_FILE_PATTERNS", None) or None
    bridge.hf_pretrained.save_artifacts(
        output_path,
        original_source_path=source_path,
        additional_files=additional_files,
    )


def _stream_mimo_weights_to_rank0(
    *,
    source_bridge: Any,
    hf_pretrained: Any,
    mimo_model: nn.Module,
    routes: list[MIMOComponent],
    pg_collections: dict[str, Any],
    show_progress: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream global HF tensors on rank 0 while all active ranks drain collectives."""
    if not dist.is_initialized():
        yield from export_megatron_mimo_to_hf(
            source_bridge=source_bridge,
            hf_pretrained=hf_pretrained,
            mimo_model=mimo_model,
            routes=routes,
            pg_collections=pg_collections,
            cpu=True,
            show_progress=show_progress,
        )
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for route in routes:
        if route.name not in pg_collections:
            raise ValueError(
                f"Route {route.name!r} is not present in MegatronMIMOInfra.pg_collections "
                f"(available: {sorted(pg_collections.keys())})."
            )

        pg_collection = pg_collections[route.name]
        is_active = pg_collection is not None
        is_representative = is_active and _is_component_export_representative(pg_collection)
        route_iter = None
        local_pairs = []
        if is_active:
            route_iter = iter(
                export_megatron_mimo_to_hf(
                    source_bridge=source_bridge,
                    hf_pretrained=hf_pretrained,
                    mimo_model=mimo_model,
                    routes=[route],
                    pg_collections=pg_collections,
                    cpu=is_representative,
                    show_progress=show_progress and rank == 0,
                )
            )

        route_tensor_count = 0
        if rank == 0:
            logger.info("save_hf_pretrained_mimo: exporting route=%r", route.name)

        if route_iter is not None:
            for payload in route_iter:
                if is_representative:
                    route_tensor_count += 1
                    if rank == 0:
                        yield payload
                    else:
                        local_pairs.append(payload)

        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_pairs if is_representative and rank != 0 else None, gathered, dst=0)
        if rank == 0:
            remote_chunks = [chunk for chunk in gathered if chunk is not None]
            if len(remote_chunks) > 1:
                raise RuntimeError(
                    f"Expected at most one non-rank-0 representative chunk for route {route.name!r}, "
                    f"got {len(remote_chunks)}. Check component replica/export rank selection."
                )
            for chunk in remote_chunks:
                route_tensor_count = len(chunk)
                for payload in chunk:
                    yield payload

            logger.info("save_hf_pretrained_mimo: finished route=%r, tensors=%d", route.name, route_tensor_count)


def _is_component_export_representative(pg_collection: Any) -> bool:
    for group_name in ("tp", "pp", "cp", "dp"):
        group = getattr(pg_collection, group_name, None)
        if group is not None and _process_group_rank(group) != 0:
            return False
    return True


def _process_group_rank(group: Any) -> int:
    if hasattr(group, "rank"):
        return group.rank()
    return dist.get_rank(group=group)


def _is_safetensors_source(bridge: AutoBridge) -> bool:
    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

    state = getattr(bridge.hf_pretrained, "state", None)
    if state is None:
        return False
    source = getattr(state, "source", None)
    return isinstance(source, SafeTensorsStateSource)
