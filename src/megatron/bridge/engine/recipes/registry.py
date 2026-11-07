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
"""Recipe lookup by model, hardware and context length.

Each entry is a recipe verified for one model on one or more GPU families up to one context length.
``resolve_recipe`` matches the model by its HF config (not by name, so local directories and
fine-tuned copies resolve too), keeps the entries for the detected hardware, and takes the smallest
context bucket that covers the requested ``seq_length``. The recipe's parallelism fixes the minimum
GPU count (``min_gpus``); larger world sizes scale data parallelism.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig

from megatron.bridge.engine.recipes.base import TrainingRecipe
from megatron.bridge.engine.recipes.qwen35 import qwen35_text_35b_a3b_rl_16gpu_gb200_bf16


# GPU families, matched as substrings of torch's device name ("NVIDIA GB300"), as Megatron's
# emerging_optimizers hardware profiles do.
KNOWN_HARDWARE = ("GB300", "GB200", "B300", "B200", "H200", "H100")


@dataclass(frozen=True)
class RecipeEntry:
    """One registered recipe.

    Attributes:
        name: Recipe function name (what ``--recipe`` takes in the examples).
        model_type: HF ``model_type`` of the text config.
        signature: HF text-config fields and values that identify the model size.
        hardware: GPU families the recipe was verified on.
        max_seq_length: Longest ``seq_length`` the recipe's parallelism is sized for.
        build: Recipe function; called with ``hf_path``, ``hf_revision`` and ``seq_length``.
    """

    name: str
    model_type: str
    signature: dict[str, int]
    hardware: tuple[str, ...]
    max_seq_length: int
    build: Callable[..., TrainingRecipe]


REGISTRY: tuple[RecipeEntry, ...] = (
    RecipeEntry(
        name="qwen35_text_35b_a3b_rl_16gpu_gb200_bf16",
        model_type="qwen3_5_moe_text",
        signature={"num_hidden_layers": 40, "hidden_size": 2048, "num_experts": 256},
        hardware=("GB300",),
        max_seq_length=131072,
        build=qwen35_text_35b_a3b_rl_16gpu_gb200_bf16,
    ),
)


def detect_hardware() -> str:
    """GPU family of the local device, e.g. ``"GB300"``."""
    name = torch.cuda.get_device_properties(0).name
    for family in KNOWN_HARDWARE:
        if family in name:
            return family
    raise ValueError(f"Unrecognized GPU {name!r}; pass hardware= explicitly (one of {KNOWN_HARDWARE}).")


def _describe(entries: tuple[RecipeEntry, ...] | list[RecipeEntry]) -> str:
    return "\n".join(
        f"  - {e.name}: {e.model_type} {e.signature}, hardware {e.hardware}, up to {e.max_seq_length} tokens"
        for e in entries
    )


def resolve_recipe(
    model: str,
    *,
    seq_length: int,
    hardware: str | None = None,
    hf_revision: str | None = None,
) -> TrainingRecipe:
    """Pick the registered recipe for ``model`` on this hardware that covers ``seq_length``.

    Args:
        model: HF model ID or local directory; its config identifies the model.
        seq_length: Packed token budget per microbatch; rounded up to the entry's context bucket for
            the lookup, passed as is to the recipe function.
        hardware: GPU family, or ``None`` to detect it from the local CUDA device.
        hf_revision: HF revision when ``model`` is a model ID.

    Returns:
        The recipe, built with ``hf_path=model``.
    """
    if seq_length <= 0:
        raise ValueError("seq_length must be positive.")
    hardware = hardware or detect_hardware()
    hf_config = AutoConfig.from_pretrained(model, revision=hf_revision)
    text_config = getattr(hf_config, "text_config", hf_config)
    model_type = getattr(text_config, "model_type", None)

    matching_model = [
        e
        for e in REGISTRY
        if e.model_type == model_type
        and all(getattr(text_config, key, None) == value for key, value in e.signature.items())
    ]
    if not matching_model:
        raise ValueError(
            f"No recipe registered for model {model!r} ({model_type}). Registered:\n{_describe(REGISTRY)}"
        )
    matching_hardware = [e for e in matching_model if hardware in e.hardware]
    if not matching_hardware:
        raise ValueError(
            f"No recipe for {model!r} is verified on {hardware}. Registered for this model:\n"
            f"{_describe(matching_model)}"
        )
    covering = [e for e in matching_hardware if e.max_seq_length >= seq_length]
    if not covering:
        raise ValueError(
            f"No recipe for {model!r} on {hardware} covers seq_length {seq_length}. Registered:\n"
            f"{_describe(matching_hardware)}"
        )
    entry = min(covering, key=lambda e: e.max_seq_length)
    return entry.build(hf_path=model, hf_revision=hf_revision, seq_length=seq_length)


def apply_recipe_overrides(recipe: TrainingRecipe, overrides: dict[str, Any]) -> None:
    """Set recipe fields by dotted path in place, e.g. ``{"optimizer.lr": 1e-6, "router_replay": True}``.

    Every path must name an existing attribute; typos raise instead of adding fields.
    """
    for path, value in overrides.items():
        target = recipe
        *parents, leaf = path.split(".")
        for name in parents:
            if not hasattr(target, name):
                raise ValueError(f"Unknown recipe field {path!r} ({name!r} does not exist).")
            target = getattr(target, name)
        if not hasattr(target, leaf):
            raise ValueError(f"Unknown recipe field {path!r} ({leaf!r} does not exist).")
        setattr(target, leaf, value)
