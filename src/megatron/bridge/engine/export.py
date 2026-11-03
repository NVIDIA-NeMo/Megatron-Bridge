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

The bridge is rebuilt from the recipe's ``hf_path`` on first use (config and weight layout only; no
weights are read), so the exported layout is the one the recipe was built from, for example the
text-only Qwen3.5 checkpoint rather than the vision-language one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.transformer import MegatronModule

from megatron.bridge.engine.recipes.base import TrainingRecipe
from megatron.bridge.models.conversion.auto_bridge import AutoBridge


@dataclass
class ExportState:
    """Per-engine export cache: the bridge and the conversion tasks planned for this model."""

    bridge: Any
    tasks: list[Any] | None = None


def make_export_state(recipe: TrainingRecipe) -> ExportState:
    """Build the bridge for ``recipe.hf_path``; raises when the recipe names no HF source."""
    if recipe.hf_path is None:
        raise ValueError("export needs recipe.hf_path (the HF checkpoint whose layout is written).")
    if getattr(recipe.ddp, "overlap_param_gather", False):
        # TODO(export): force start_param_sync(force_sync=True) on every chunk before reading the weights.
        raise NotImplementedError("export with ddp.overlap_param_gather=True is not supported yet.")
    return ExportState(bridge=AutoBridge.from_hf_pretrained(recipe.hf_path))


def export_hf_weights(
    state: ExportState, model: list[MegatronModule], *, path: str | None, streaming: bool
) -> Iterator[tuple[str, torch.Tensor]] | None:
    """Write an HF checkpoint directory, or yield ``(name, tensor)`` pairs with full tensors on every rank.

    Both modes are collective: every rank calls this, and in streaming mode every rank must drain the
    iterator because the TP/EP/PP gathers happen inside it. The conversion plan is built once per
    engine and reused by later streaming calls.
    """
    if streaming:
        if state.tasks is None:
            state.tasks = state.bridge.get_conversion_tasks(model)
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
