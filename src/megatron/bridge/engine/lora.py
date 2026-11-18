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
"""LoRA engines (``from_recipe(lora=LoraConfig(...))``) over Bridge's ``LoRA``.

The engine only converts a ``LoraConfig`` into Bridge's ``LoRA`` (``build_lora``) and applies it with
Bridge's PEFT hook (``peft.utils.create_peft_hook``), so targets match, the base is frozen and checkpoints
keep the adapters exactly as in Bridge's own LoRA training.

Construction follows Bridge's PEFT setup (``_create_peft_pre_wrap_hook`` in ``training/setup.py``): the
pretrained base weights are loaded into the bare model, LoRA then wraps the target modules and freezes
every base parameter, and only afterwards are DDP and the optimizer built, so both hold the adapters
alone. A LoRA engine therefore loads ``checkpoint_path`` inside its pre-wrap hook, as the value head does,
instead of after the build.

The rest of the engine needs three LoRA facts: ``finalize_model_grads`` must also sum the gradients of
expert adapters that are replicated across expert-parallel ranks (``peft_enabled`` when Bridge installs
the model-config hooks), ``forward_only(disable_adapters=True)`` runs under ``PEFT.disable_adapter``,
and checkpoints carry the LoRA object so Bridge saves and resumes the adapters (and MoE router expert
biases) only.
"""

from __future__ import annotations

from collections.abc import Callable

from megatron.core.transformer import MegatronModule

from megatron.bridge.engine.checkpoint import load_pretrained_weights
from megatron.bridge.engine.datum import LORA_EXPERT_ADAPTER_LAYOUTS, LoraConfig
from megatron.bridge.engine.recipes.base import TrainingRecipe
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.utils import create_peft_hook


PreWrapHook = Callable[[list[MegatronModule]], list[MegatronModule]]


def build_lora(config: LoraConfig, recipe: TrainingRecipe) -> LoRA:
    """Lower a ``LoraConfig`` to Bridge's ``LoRA``.

    ``config.target_modules=None`` takes ``recipe.lora_target_modules``. ``config.expert_adapters`` picks
    Bridge's layout for grouped MoE expert linears (``share_expert_adapters`` and
    ``experts_shared_outer_loras``); whatever a layout replicates across expert-parallel ranks is summed
    there by the gradient sync the engine installs. Bridge's other LoRA settings keep their defaults.

    Args:
        config: The user's adapter settings.
        recipe: The engine's recipe; supplies the default target modules.

    Returns:
        A fresh ``LoRA`` owned by the engine.

    Raises:
        ValueError: For a non-positive rank or alpha, a dropout outside ``[0, 1)``, an empty target list or
            an unknown ``expert_adapters`` layout.
    """
    targets = list(config.target_modules) if config.target_modules is not None else list(recipe.lora_target_modules)
    problems: list[str] = []
    if not isinstance(config.rank, int) or config.rank <= 0:
        problems.append(f"rank must be a positive integer, got {config.rank!r}")
    if not config.alpha > 0:
        problems.append(f"alpha must be positive, got {config.alpha!r}")
    if not 0.0 <= config.dropout < 1.0:
        problems.append(f"dropout must be in [0, 1), got {config.dropout!r}")
    if not targets:
        problems.append("target_modules is empty")
    if config.expert_adapters not in LORA_EXPERT_ADAPTER_LAYOUTS:
        problems.append(
            f"expert_adapters must be one of {sorted(LORA_EXPERT_ADAPTER_LAYOUTS)}, got {config.expert_adapters!r}"
        )
    if problems:
        raise ValueError("LoraConfig is not supported by TrainingEngine:\n  - " + "\n  - ".join(problems))
    share_expert_adapters, experts_shared_outer_loras = LORA_EXPERT_ADAPTER_LAYOUTS[config.expert_adapters]
    return LoRA(
        target_modules=targets,
        dim=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        share_expert_adapters=share_expert_adapters,
        experts_shared_outer_loras=experts_shared_outer_loras,
    )


def make_lora_hook(recipe: TrainingRecipe, peft: LoRA, checkpoint_path: str) -> PreWrapHook:
    """Pre-wrap hook: load the base weights into the bare model, then apply LoRA with Bridge's PEFT hook.

    ``create_peft_hook`` wraps the targets, freezes the base and records the adapter parameters it saves
    (``set_params_to_save``). The base load runs with no PEFT in the checkpoint view, so every base tensor
    is read.

    Args:
        recipe: The engine's recipe (checkpoint loading reads its model, optimizer and DDP sections).
        peft: The engine's ``LoRA``, from ``build_lora``.
        checkpoint_path: Megatron or HF checkpoint holding the pretrained base weights.

    Returns:
        The hook to register on ``recipe.model``.
    """
    apply_lora = create_peft_hook(peft)

    def hook(model: list[MegatronModule]) -> list[MegatronModule]:
        load_pretrained_weights(recipe, model, None, checkpoint_path)
        return apply_lora(model)

    return hook
