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


# Nemotron 3 Ultra's 108-layer decoder pattern from the Hugging Face config.
NEMOTRON_3_ULTRA_HYBRID_LAYER_PATTERN = (
    "MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
)


class _OfflineModelProvider:
    """Mutable provider with the fields HF-backed recipes read before writing.

    Add newly read provider fields here when extending the recipe surface.
    """

    def __init__(self) -> None:
        self.apply_rope_fusion = False
        self.batch_p2p_comm = False
        self.bf16 = False
        self.context_parallel_size = 1
        self.cross_entropy_fusion_impl = "native"
        self.csa_compress_ratios = [0] * 33
        self.cuda_graph_modules = None
        self.cuda_graph_scope = None
        self.dsa_indexer_skip_topk_offset = 0
        self.dsa_indexer_topk_freq = 1
        self.delay_wgrad_compute = False
        self.experimental_attention_variant = "dsa"
        # Builder-backed configs record their Hugging Face source here.
        self.extra_checkpoint_metadata = {}
        self.fp16 = False
        # Preconditions of the MXFP8 CuTe DSL fused grouped MLP, asserted by the
        # perf recipe tests: SwiGLU, and FC1/FC2 dims divisible by 64. Absent here
        # the assertions raise AttributeError rather than failing on the value.
        # Values mirror Qwen3.5-VL 35B-A3B.
        self.gated_linear_unit = True
        # DeepSeek's bridge enables this before communication-overlap setup.
        self.gradient_accumulation_fusion = True
        self.hidden_size = 2048
        self.make_vocab_size_divisible_by = 128
        self.moe_combine_bwd_dtype = "bf16"
        self.moe_dispatch_fwd_dtype = "bf16"
        self.moe_flex_dispatcher_backend = None
        self.moe_ffn_hidden_size = 768
        self.mtp_num_layers = 1
        self.num_layers = 32
        self.num_moe_experts = 8
        self.overlap_moe_expert_parallel_comm = False
        self.overlap_p2p_comm = False
        self.rotary_base = 10000.0
        self.rotary_scaling_factor = 1
        self.seq_length = 4096
        self.tensor_model_parallel_size = 1
        self.tp_comm_overlap = False
        self.use_te_rng_tracker = False
        self.use_transformer_engine_op_fuser = False
        self.vocab_size = 256000
        self.yarn_original_max_position_embeddings = 32768

    def to_text_provider(self) -> "_OfflineModelProvider":
        """Match multimodal provider conversion without initializing a model."""
        return self

    def finalize(self) -> None:
        """Match the model-provider interface without initializing a model."""


class _OfflineAutoBridge:
    """Build a local model configuration without reading a Hugging Face configuration."""

    @classmethod
    def from_hf_config(cls, *args: object, **kwargs: object) -> "_OfflineAutoBridge":
        del args, kwargs
        return cls()

    @classmethod
    def from_hf_pretrained(cls, *args: object, **kwargs: object) -> "_OfflineAutoBridge":
        bridge = cls()
        bridge._model_id = str(args[0]) if args else str(kwargs.get("pretrained_model_name_or_path", ""))
        return bridge

    def to_megatron_provider(self, *args: object, **kwargs: object) -> _OfflineModelProvider:
        del args, kwargs
        provider = _OfflineModelProvider()
        model_id = getattr(self, "_model_id", "").lower()
        if model_id in {"zai-org/glm-5", "zai-org/glm-5.1", "zai-org/glm-5.2", "zai-org/glm-5.3"}:
            provider.hybrid_layer_pattern = "D-" * 3 + "DE" * 75
            provider.num_layers = len(provider.hybrid_layer_pattern)
            if model_id in {"zai-org/glm-5.2", "zai-org/glm-5.3"}:
                provider.dsa_indexer_topk_freq = 4
                provider.dsa_indexer_skip_topk_offset = 3
        if model_id == "nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16":
            provider.hybrid_layer_pattern = NEMOTRON_3_ULTRA_HYBRID_LAYER_PATTERN
            provider.num_layers = len(provider.hybrid_layer_pattern)
        if "deepseek-v4-" in model_id:
            logical_layers = 61 if "deepseek-v4-pro" in model_id else 43
            provider.hybrid_layer_pattern = "WEWE" + "".join(
                "CE" if i % 2 == 0 else "HE" for i in range(logical_layers - 2)
            )
            provider.num_layers = 2 * logical_layers
            provider.mtp_hybrid_override_pattern = "WE"
            provider.moe_num_hash_layers = 3
            provider.csa_compress_ratios = [
                {"C": 4, "H": 128}.get(symbol, 0) for symbol in provider.hybrid_layer_pattern
            ] + [0, 0]
        return provider

    def get_model_config(self) -> _OfflineModelProvider:
        """Return the same mutable stand-in for builder-backed recipe construction."""
        return self.to_megatron_provider()


# A small Nemotron-H architecture with Mamba, attention, dense MLP and MoE layers.
_NEMOTRON_H_HF_CONFIG = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "mamba_num_heads": 8,
    "mamba_head_dim": 32,
    "ssm_state_size": 16,
    "n_groups": 2,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 128,
    "moe_shared_expert_intermediate_size": 256,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 2.5,
    "vocab_size": 1024,
    "max_position_embeddings": 8192,
    "hybrid_override_pattern": "MEM*EM-E",
    "dtype": "bfloat16",
}


def nemotron_h_model_config(model_id: str = "", **hf_overrides: object):
    """Build a strict builder-backed Nemotron-H model config from a small Hugging Face config.

    Args:
        model_id: Source id recorded in the config's checkpoint metadata.
        **hf_overrides: Hugging Face ``NemotronHConfig`` fields to override.

    Returns:
        The ``BridgeHybridModelConfig`` that ``AutoBridge.get_model_config()`` returns.
    """
    from transformers import NemotronHConfig

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    values = {**_NEMOTRON_H_HF_CONFIG, **hf_overrides}
    values.setdefault("num_hidden_layers", len(values["hybrid_override_pattern"]))
    hf_config = NemotronHConfig(**values)
    hf_config.architectures = ["NemotronHForCausalLM"]
    hf_config._name_or_path = model_id
    return AutoBridge.from_hf_config(hf_config).get_model_config()


class NemotronHBuilderOnlyBridge:
    """Stand-in for ``AutoBridge`` that returns strict Nemotron-H model configs offline.

    Subclasses set ``hf_config_overrides`` to select another architecture. The
    legacy provider API fails, so recipes must use ``get_model_config()``.
    """

    hf_config_overrides: dict[str, object] = {}

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        self.model_id = model_id
        self.hf_model_revision = revision

    @classmethod
    def from_hf_pretrained(cls, model_id: object, **kwargs: object) -> "NemotronHBuilderOnlyBridge":
        return cls(str(model_id), kwargs.get("revision"))

    def get_model_config(self):
        model_config = nemotron_h_model_config(self.model_id, **self.hf_config_overrides)
        if self.hf_model_revision:
            model_config.extra_checkpoint_metadata["hf_model_revision"] = self.hf_model_revision
        return model_config

    def to_megatron_provider(self, *args: object, **kwargs: object):
        raise AssertionError("Nemotron-H recipes must use get_model_config(), not the legacy provider API")


class NemotronHUltraBuilderOnlyBridge(NemotronHBuilderOnlyBridge):
    """Builder-only stand-in with Nemotron 3 Ultra's decoder pattern and shared MTP block."""

    hf_config_overrides = {
        "hybrid_override_pattern": NEMOTRON_3_ULTRA_HYBRID_LAYER_PATTERN,
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*E",
    }


# GLM-5.2's architecture from its Hugging Face config.
_GLM5_HF_CONFIG = {
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "hidden_size": 6144,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "index_topk_freq": 4,
    "index_skip_topk_offset": 3,
    "indexer_rope_interleave": True,
    "rope_parameters": {"rope_theta": 8_000_000, "rope_type": "default"},
    "vocab_size": 154880,
    "max_position_embeddings": 1048576,
    "num_nextn_predict_layers": 1,
    "dtype": "bfloat16",
}

# GLM-5 and GLM-5.1 compute DSA top-k indices on every layer.
GLM5_1_HF_CONFIG_OVERRIDES = {
    "index_topk_freq": 1,
    "index_skip_topk_offset": 0,
    "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
    "max_position_embeddings": 202752,
}


def glm5_model_config(model_id: str = "", **hf_overrides: object):
    """Build a strict builder-backed GLM-5 model config from GLM-5.2's architecture.

    Args:
        model_id: Source id recorded in the config's checkpoint metadata.
        **hf_overrides: Hugging Face ``GlmMoeDsaConfig`` fields to override.

    Returns:
        The ``GLM5ModelConfig`` that ``AutoBridge.get_model_config()`` returns.
    """
    from transformers import GlmMoeDsaConfig

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    hf_config = GlmMoeDsaConfig(**{**_GLM5_HF_CONFIG, **hf_overrides})
    hf_config.architectures = ["GlmMoeDsaForCausalLM"]
    hf_config._name_or_path = model_id
    return AutoBridge.from_hf_config(hf_config).get_model_config()


class GLM5BuilderOnlyBridge:
    """Stand-in for ``AutoBridge`` that returns strict GLM-5 model configs offline.

    GLM-5 and GLM-5.1 sources get their DSA and RoPE settings. The legacy provider API
    fails, so recipes must use ``get_model_config()``.
    """

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        self.model_id = model_id
        self.hf_model_revision = revision

    @classmethod
    def from_hf_pretrained(cls, model_id: object, **kwargs: object) -> "GLM5BuilderOnlyBridge":
        return cls(str(model_id), kwargs.get("revision"))

    def get_model_config(self):
        overrides = GLM5_1_HF_CONFIG_OVERRIDES if self.model_id in ("zai-org/GLM-5", "zai-org/GLM-5.1") else {}
        model_config = glm5_model_config(self.model_id, **overrides)
        if self.hf_model_revision:
            model_config.extra_checkpoint_metadata["hf_model_revision"] = self.hf_model_revision
        return model_config

    def to_megatron_provider(self, *args: object, **kwargs: object):
        raise AssertionError("GLM-5 recipes must use get_model_config(), not the legacy provider API")


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

    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(recipe_prefixes):
            continue
        if hasattr(module, "AutoBridge"):
            monkeypatch.setattr(module, "AutoBridge", _OfflineAutoBridge)

    flux_recipe_module = importlib.import_module("megatron.bridge.recipes.flux.h100.flux")
    monkeypatch.setattr(flux_recipe_module, "PreTrainedFlux", _OfflinePreTrainedFlux)

    deepseek_v4_recipe_module = importlib.import_module("megatron.bridge.recipes.deepseek.h100.deepseek_v4")
    monkeypatch.setattr(deepseek_v4_recipe_module, "deepseek_v4_supports_blackwell_fused_kernels", lambda: False)
    deepseek_v4_gb300_module = importlib.import_module("megatron.bridge.recipes.deepseek.gb300.deepseek_v4")
    monkeypatch.setattr(deepseek_v4_gb300_module, "deepseek_v4_supports_blackwell_fused_kernels", lambda: False)

    from transformers import AutoConfig, AutoTokenizer

    def load_offline_auto_config(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            tie_word_embeddings=False,
            text_config=SimpleNamespace(
                architectures=None,
                model_type="qwen3_5_moe_text",
                max_position_embeddings=262144,
            ),
        )

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
