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

"""Lazy resolution of model-owned collators by HF processor identity."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast


@dataclass(frozen=True)
class _ModelCollateSpec:
    module_name: str
    symbol_name: str
    required_for_all_examples: bool = False
    model_name_prefixes: tuple[str, ...] = ()
    supported_chat_loss_modes: tuple[str, ...] = ("assistant",)


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
        "nemotron_omni_expanded_collate_fn",
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
    "deepseek-v4": _ModelCollateSpec(
        "megatron.bridge.models.deepseek.data.collate_fn",
        "deepseek_v4_collate_fn",
        required_for_all_examples=True,
        model_name_prefixes=("deepseek-v4",),
        supported_chat_loss_modes=("assistant", "last_turn", "full"),
    ),
}


def _normalize_model_identifier(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1].lower().replace("_", "-")


def _local_model_type(name_or_path: str) -> str | None:
    config_path = Path(name_or_path).expanduser() / "config.json"
    try:
        with config_path.open(encoding="utf-8") as config_file:
            model_type = json.load(config_file).get("model_type")
    except (AttributeError, json.JSONDecodeError, OSError):
        return None
    return _normalize_model_identifier(model_type) if isinstance(model_type, str) else None


def _model_identifiers_from_processor(processor: Any) -> tuple[str, ...]:
    identifiers: list[str] = []
    current = processor
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name_or_path = getattr(current, "name_or_path", None)
        if isinstance(name_or_path, str) and name_or_path:
            identifiers.append(_normalize_model_identifier(name_or_path))
            for path_part in Path(name_or_path).parts:
                normalized_part = _normalize_model_identifier(path_part)
                identifiers.append(normalized_part)
                if normalized_part.startswith("models--"):
                    identifiers.append(normalized_part.rsplit("--", 1)[-1])
            local_model_type = _local_model_type(name_or_path)
            if local_model_type is not None:
                identifiers.append(local_model_type)
        config = getattr(current, "config", None)
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str) and model_type:
            identifiers.append(_normalize_model_identifier(model_type))
        nested = getattr(current, "tokenizer", None)
        current = nested if nested is not current else None
    return tuple(dict.fromkeys(identifiers))


def _model_collate_spec_for_processor(processor: Any) -> _ModelCollateSpec | None:
    spec = _MODEL_COLLATE_SPECS.get(type(processor).__name__)
    if spec is not None:
        return spec
    identifiers = _model_identifiers_from_processor(processor)
    return next(
        (
            spec
            for spec in _MODEL_COLLATE_SPECS.values()
            if any(identifier.startswith(prefix) for prefix in spec.model_name_prefixes for identifier in identifiers)
        ),
        None,
    )


def _resolve_model_collate_spec(spec: _ModelCollateSpec) -> Callable[..., dict[str, Any]]:
    collate = getattr(import_module(spec.module_name), spec.symbol_name)
    if not callable(collate):
        raise TypeError(f"Registered collator {spec.module_name}.{spec.symbol_name} is not callable.")
    return cast(Callable[..., dict[str, Any]], collate)


def model_collate_required_for_all_examples(processor_type: str) -> bool:
    """Return whether a processor must always use its model-owned collator."""
    spec = _MODEL_COLLATE_SPECS.get(processor_type)
    return spec is not None and spec.required_for_all_examples


def model_collate_required_for_processor(processor: Any) -> bool:
    """Return whether a processor or tokenizer must use its model-owned collator."""
    spec = _model_collate_spec_for_processor(processor)
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
    return _resolve_model_collate_spec(spec)


def resolve_model_collate_for_processor(
    processor: Any,
    *,
    loss_mode: Literal["assistant", "last_turn", "full"] = "assistant",
) -> Callable[..., dict[str, Any]]:
    """Resolve a model-owned collator from processor type or tokenizer model name."""
    spec = _model_collate_spec_for_processor(processor)
    if spec is None:
        processor_type = type(processor).__name__
        model_identifiers = _model_identifiers_from_processor(processor)
        details = f" type '{processor_type}'" + (f" and models {model_identifiers}" if model_identifiers else "")
        raise ValueError(f"No model collate function is registered for processor{details}.")
    if loss_mode not in spec.supported_chat_loss_modes:
        raise ValueError(
            f"Registered collator {spec.module_name}.{spec.symbol_name} supports only chat loss modes "
            f"{spec.supported_chat_loss_modes}, not '{loss_mode}'."
        )
    collate = _resolve_model_collate_spec(spec)
    if loss_mode == "assistant":
        return collate
    return partial(collate, loss_mode=loss_mode)
