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

"""Helpers for recipe unit tests."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import socket
import sys
from collections.abc import Callable, Iterator
from types import ModuleType, SimpleNamespace

import pytest


def _iter_recipe_owner_modules(module: ModuleType) -> Iterator[ModuleType]:
    """Yield modules that define recipe functions exported by a compatibility module."""
    exported_names = getattr(module, "__all__", ())
    for name in exported_names:
        value = getattr(module, name, None)
        if callable(value) and hasattr(value, "__module__"):
            yield importlib.import_module(value.__module__)


def patch_recipe_module_global(
    monkeypatch: pytest.MonkeyPatch,
    module_or_recipe_func: ModuleType | Callable[..., object],
    name: str,
    value: object,
) -> None:
    """Patch a global on the module that owns a recipe function.

    Compatibility recipe modules re-export canonical H100 recipe functions with
    direct import aliases. Tests should patch the canonical function owner rather
    than the compatibility module, because patching the compatibility module no
    longer affects the aliased function body.
    """
    if callable(module_or_recipe_func) and hasattr(module_or_recipe_func, "__module__"):
        owner = importlib.import_module(module_or_recipe_func.__module__)
        monkeypatch.setattr(owner, name, value)
        return

    module = module_or_recipe_func
    patched_ids: set[int] = set()
    if hasattr(module, name):
        monkeypatch.setattr(module, name, value)
        patched_ids.add(id(module))
    for owner in _iter_recipe_owner_modules(module):
        if hasattr(owner, name) and id(owner) not in patched_ids:
            monkeypatch.setattr(owner, name, value)
            patched_ids.add(id(owner))


def discover_recipe_factories(
    package: ModuleType,
    *,
    exclude_module_prefixes: tuple[str, ...] = (),
) -> tuple[Callable[..., object], ...]:
    """Import and return every public recipe factory defined below ``package``.

    Compatibility modules often re-export the same callable under a second
    name. Restricting discovery to functions owned by each leaf module keeps
    the construction matrix focused on unique implementations.
    """
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise ValueError(f"{package.__name__} is not a package")

    factories: dict[tuple[str, str], Callable[..., object]] = {}
    prefix = f"{package.__name__}."
    for module_info in pkgutil.walk_packages(package_path, prefix=prefix):
        if module_info.name.startswith(exclude_module_prefixes):
            continue

        module = importlib.import_module(module_info.name)
        for name, value in vars(module).items():
            if name.startswith("_") or not name.endswith("_config") or not inspect.isfunction(value):
                continue
            if value.__module__ != module.__name__:
                continue
            factories[(value.__module__, value.__qualname__)] = value

    return tuple(factories[key] for key in sorted(factories))


def recipe_factory_key(factory: Callable[..., object]) -> tuple[str, str]:
    """Return a stable identity for a recipe factory or compatibility alias."""
    return factory.__module__, factory.__qualname__


def recipe_factory_id(factory: Callable[..., object]) -> str:
    """Return a readable pytest parameter ID for a recipe factory."""
    module, name = recipe_factory_key(factory)
    return f"{module}:{name}"


def exported_recipe_factory_keys(module: ModuleType) -> set[tuple[str, str]]:
    """Return identities of public ``*_config`` callables exported by a module."""
    return {
        recipe_factory_key(value)
        for name, value in vars(module).items()
        if not name.startswith("_") and name.endswith("_config") and callable(value)
    }


class _OfflineModelProvider:
    """Mutable provider with the fields HF-backed recipes read before writing.

    Add newly read provider fields here when extending the recipe surface.
    """

    def __init__(self) -> None:
        self.apply_rope_fusion = False
        self.context_parallel_size = 1
        self.cross_entropy_fusion_impl = "native"
        self.csa_compress_ratios = [0] * 33
        self.dsa_indexer_skip_topk_offset = 0
        self.dsa_indexer_topk_freq = 1
        self.experimental_attention_variant = "dsa"
        self.make_vocab_size_divisible_by = 128
        self.moe_flex_dispatcher_backend = None
        self.mtp_num_layers = 1
        self.num_layers = 32
        self.num_moe_experts = 8
        self.rotary_base = 10000.0
        self.rotary_scaling_factor = 1
        self.seq_length = 4096
        self.tensor_model_parallel_size = 1
        self.use_te_rng_tracker = False
        self.use_transformer_engine_op_fuser = False
        self.vocab_size = 256000
        self.yarn_original_max_position_embeddings = 32768

    def finalize(self) -> None:
        """Match the model-provider interface without initializing a model."""


class _OfflineAutoBridge:
    """Build a local provider without reading a Hugging Face configuration."""

    @classmethod
    def from_hf_config(cls, *args: object, **kwargs: object) -> "_OfflineAutoBridge":
        del args, kwargs
        return cls()

    @classmethod
    def from_hf_pretrained(cls, *args: object, **kwargs: object) -> "_OfflineAutoBridge":
        del args, kwargs
        return cls()

    def to_megatron_provider(self, *args: object, **kwargs: object) -> _OfflineModelProvider:
        del args, kwargs
        return _OfflineModelProvider()


class _OfflineTokenizer:
    """Small tokenizer stand-in used by Gemma vocabulary adjustment helpers."""

    def __len__(self) -> int:
        return 256000


class _OfflinePreTrainedFlux:
    """Expose the local config surface consumed by ``FluxBridge``."""

    def __init__(self, model_name_or_path: object, **kwargs: object) -> None:
        del kwargs
        self.model_name_or_path = str(model_name_or_path)
        self.config = SimpleNamespace(
            num_attention_heads=24,
            attention_head_dim=128,
            in_channels=64,
            patch_size=1,
            num_layers=19,
            num_single_layers=38,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            axes_dims_rope=[16, 56, 56],
            ffn_dim=12288,
        )


def patch_recipe_construction_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep exhaustive recipe construction deterministic, CPU-only, and offline."""
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    recipe_prefixes = ("megatron.bridge.recipes.", "megatron.bridge.perf_recipes.")

    def skip_flex_dispatcher_hardware_probe(*args: object, **kwargs: object) -> None:
        del args, kwargs

    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(recipe_prefixes):
            continue
        if hasattr(module, "AutoBridge"):
            monkeypatch.setattr(module, "AutoBridge", _OfflineAutoBridge)
        if hasattr(module, "apply_flex_dispatcher_backend"):
            monkeypatch.setattr(module, "apply_flex_dispatcher_backend", skip_flex_dispatcher_hardware_probe)

    flux_recipe_module = importlib.import_module("megatron.bridge.recipes.flux.h100.flux")
    monkeypatch.setattr(flux_recipe_module, "PreTrainedFlux", _OfflinePreTrainedFlux)

    deepseek_v4_recipe_module = importlib.import_module("megatron.bridge.recipes.deepseek.h100.deepseek_v4")
    monkeypatch.setattr(deepseek_v4_recipe_module, "deepseek_v4_supports_blackwell_fused_kernels", lambda: False)

    from transformers import AutoConfig, AutoTokenizer

    def load_offline_auto_config(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(text_config=SimpleNamespace(architectures=None))

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(load_offline_auto_config),
    )

    def load_offline_tokenizer(*args: object, **kwargs: object) -> _OfflineTokenizer:
        del args, kwargs
        return _OfflineTokenizer()

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(load_offline_tokenizer),
    )

    def reject_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("recipe construction attempted network access")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
