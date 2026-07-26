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

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch.distributed as dist


_REMOVAL_VERSION = "0.7.0"

_LEGACY_NEMOTRON_NAME = "the legacy Nemotron bridge (NemotronForCausalLM, previously documented for Nemotron-4 340B)"

_DEPRECATED_ARCHITECTURES = {
    "GemmaForCausalLM": "Gemma 1 (2B and 7B)",
    "Gemma2ForCausalLM": "Gemma 2 (2B, 9B, and 27B)",
    "NemotronForCausalLM": _LEGACY_NEMOTRON_NAME,
}

_LLAMA2_SHAPES = {(32000, 4096)}
_MISTRAL_SHAPES = {(4096, 32), (5120, 40)}
_NEMOTRON_H_V1_SHAPES = {(3072, 52), (4096, 52), (8192, 98), (8192, 118)}


def _model_identifier(config: Any, model_name_or_path: str | Path | None) -> str:
    candidates = [model_name_or_path, getattr(config, "name_or_path", None), getattr(config, "_name_or_path", None)]
    for candidate in candidates:
        if isinstance(candidate, (str, Path)) and candidate:
            return str(candidate).lower()
    return ""


def _architectures(config: Any) -> set[str]:
    architectures = getattr(config, "architectures", None)
    if not isinstance(architectures, (list, tuple)):
        return set()
    return {architecture for architecture in architectures if isinstance(architecture, str)}


def _deprecated_model_name(config: Any, model_name_or_path: str | Path | None = None) -> str | None:
    architectures = _architectures(config)

    for architecture, model_name in _DEPRECATED_ARCHITECTURES.items():
        if architecture in architectures:
            return model_name

    identifier = _model_identifier(config, model_name_or_path)

    if "LlamaForCausalLM" in architectures:
        shape = (getattr(config, "vocab_size", None), getattr(config, "max_position_embeddings", None))
        if "llama-2" in identifier or "llama2" in identifier or shape in _LLAMA2_SHAPES:
            return "Llama 2"

    if "MistralForCausalLM" in architectures:
        shape = (getattr(config, "hidden_size", None), getattr(config, "num_hidden_layers", None))
        if "mistral-7b" in identifier or "mistral-small-24b" in identifier or shape in _MISTRAL_SHAPES:
            return "Mistral 7B and Mistral Small 3 24B"

    if "NemotronHForCausalLM" in architectures:
        shape = (getattr(config, "hidden_size", None), getattr(config, "num_hidden_layers", None))
        if "nemotron-h-" in identifier or shape in _NEMOTRON_H_V1_SHAPES:
            return "Nemotron H v1 (4B, 8B, 47B, and 56B)"

    return None


def warn_deprecated_model(model_name: str, *, stacklevel: int = 2) -> None:
    """Warn that support for a legacy model is scheduled for removal.

    Args:
        model_name: User-facing model family or variant name.
        stacklevel: Warning stack level passed to :func:`warnings.warn`.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    warnings.warn(
        f"Support for {model_name} is deprecated and is no longer actively maintained or tested against "
        f"current upstream checkpoints. It will be removed in Megatron Bridge {_REMOVAL_VERSION}.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def warn_if_deprecated_model(config: Any, model_name_or_path: str | Path | None = None) -> None:
    """Warn when an HF configuration belongs to a deprecated model family.

    Args:
        config: Hugging Face model configuration.
        model_name_or_path: Optional Hugging Face model identifier or local path.
    """
    model_name = _deprecated_model_name(config, model_name_or_path)
    if model_name is not None:
        warn_deprecated_model(model_name, stacklevel=4)


def warn_if_legacy_nemotron_path(model_name_or_path: str | Path) -> None:
    """Warn before loading the legacy Nemotron-4 repository without an HF config.

    Args:
        model_name_or_path: Hugging Face model identifier or local path.
    """
    if "nemotron-4-340b" in str(model_name_or_path).lower():
        warn_deprecated_model(_LEGACY_NEMOTRON_NAME, stacklevel=4)
