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
"""HF weight export over AutoBridge: directory mode and streaming mode with cached conversion tasks.

The bridge is rebuilt from the recipe's ``hf_path`` and ``hf_revision`` on first use (config and
weight layout only; no weights are read). The source must match the model the recipe built: for a
text-only model a multimodal wrapper would select a different bridge with different parameter names,
so it is rejected; a multimodal model exports through its multimodal source.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.transformer import MegatronModule
from transformers import AutoConfig

from megatron.bridge.engine.recipes.base import TrainingRecipe, is_multimodal
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.peft.base import PEFT


@dataclass
class ExportState:
    """Per-engine export cache: the bridge and the conversion tasks planned for this model."""

    bridge: Any
    tasks: list[Any] | None = None


def make_export_state(recipe: TrainingRecipe) -> ExportState:
    """Build the bridge for ``recipe.hf_path`` at ``recipe.hf_revision``.

    Raises:
        ValueError: If the recipe names no HF source.
        NotImplementedError: With ``ddp.overlap_param_gather``, or when a text-only model names a
            multimodal wrapper (its config carries ``text_config``) as the source.
    """
    if recipe.hf_path is None:
        raise ValueError("export needs recipe.hf_path (the HF checkpoint whose layout is written).")
    if getattr(recipe.ddp, "overlap_param_gather", False):
        # TODO(export): force start_param_sync(force_sync=True) on every chunk before reading the weights.
        raise NotImplementedError("export with ddp.overlap_param_gather=True is not supported yet.")
    revision = getattr(recipe, "hf_revision", None)
    multimodal_source = hasattr(AutoConfig.from_pretrained(recipe.hf_path, revision=revision), "text_config")
    if multimodal_source and not is_multimodal(recipe):
        # TODO(export): a text-only model exported through a multimodal source needs the bridge built from
        # the extracted text config (as the recipe does) plus the tokenizer files copied separately.
        raise NotImplementedError(
            f"export needs a text-only HF source; {recipe.hf_path} is a multimodal wrapper with text_config."
        )
    return ExportState(bridge=AutoBridge.from_hf_pretrained(recipe.hf_path, revision=revision))


def export_hf_weights(
    state: ExportState, model: list[MegatronModule], *, path: str | None, streaming: bool
) -> Iterator[tuple[str, torch.Tensor]] | None:
    """Write an HF checkpoint directory, or yield ``(name, tensor)`` pairs with full tensors on every rank.

    Both modes are collective: every rank calls this, and in streaming mode every rank must drain the
    iterator because the TP/EP/PP gathers happen inside it. The conversion plan is built once per
    engine and reused by later streaming calls. LoRA adapters are merged into their base tensors: Bridge
    computes ``W + (alpha / rank) * B @ A`` in fp32 and casts the sum back to the base dtype, and it
    plans the adapter gathers again on every call (only the base plan is cached).
    """
    if streaming:
        if state.tasks is None:
            state.tasks = state.bridge.get_conversion_tasks(model)
        # TODO(lora-refit): a merged refit rounds W + s * B @ A to bf16, which the trainer (adapters kept
        # apart) never does; measure the train/inference gap before choosing it over adapter-only refit.
        return (
            (item.param_name, item.weight)
            for item in state.bridge.export_hf_weights(
                model, cpu=False, show_progress=False, conversion_tasks=state.tasks, merge_adapter_weights=True
            )
        )
    if path is None:
        raise ValueError("export needs a path unless streaming=True.")
    state.bridge.save_hf_pretrained(model, path, show_progress=False)
    return None


def export_hf_adapter(
    state: ExportState,
    model: list[MegatronModule],
    peft: PEFT,
    *,
    path: str | None,
    streaming: bool,
    base_model_name_or_path: str | None,
) -> Iterator[tuple[str, torch.Tensor]] | None:
    """Export the LoRA adapters alone, unmerged, in the HF PEFT layout of the recipe's model.

    Streaming yields ``(hf_name, tensor)`` for every ``lora_A`` / ``lora_B`` with full tensors on every
    rank; like the merged stream it is collective and every rank must drain it. ``path`` writes a PEFT
    directory (``adapter_config.json`` + ``adapter_model.safetensors``) that
    ``peft.PeftModel.from_pretrained`` loads over ``base_model_name_or_path``. Expert adapters come out
    under per-expert names in every layout: an ``"outer_shared"`` adapter's shared factor is repeated for
    each expert (Bridge's ``expand_shared_outer``, which leaves the other layouts unchanged).
    """
    if streaming:
        return (
            (item.param_name, item.weight)
            for item in state.bridge.export_adapter_weights(
                model, cpu=False, show_progress=False, expand_shared_outer=True
            )
        )
    if path is None:
        raise ValueError("export needs a path unless streaming=True.")
    state.bridge.save_hf_adapter(
        model,
        path,
        peft_config=peft,
        base_model_name_or_path=base_model_name_or_path,
        show_progress=False,
        expand_shared_outer=True,
    )
    return None
