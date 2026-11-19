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

The engine only converts a ``LoraConfig`` into Bridge's ``LoRA`` (``build_lora``) and registers Bridge's PEFT
hook (``peft.utils.create_peft_hook``) as a pre-wrap hook, so targets match, the base is frozen and
checkpoints keep the adapters exactly as in Bridge's own LoRA training.

The hook runs on the bare model, before DDP and the optimizer are built, so both hold the adapters alone. The
base weights are loaded after the build by the same call as a full-weight engine's, with the adapters hidden
for the load (``peft.adapter_wrapper.hide_adapters``) so the base checkpoint maps onto the base weights alone.
LoRA can go on before the base is loaded because its adapters do not read the base weights when they are
built.

The rest of the engine needs three LoRA facts: ``finalize_model_grads`` must also sum the gradients of
expert adapters that are replicated across expert-parallel ranks (``peft_enabled`` when Bridge installs
the model-config hooks), ``forward_only(disable_adapters=True)`` runs under ``PEFT.disable_adapter``,
and checkpoints carry the LoRA object so Bridge saves and resumes the adapters (and MoE router expert
biases) only.
"""

from __future__ import annotations

from megatron.bridge.engine.datum import LORA_EXPERT_ADAPTER_LAYOUTS, LoraConfig
from megatron.bridge.engine.recipes.base import TrainingRecipe
from megatron.bridge.peft.lora import LoRA


def build_lora(config: LoraConfig, recipe: TrainingRecipe) -> LoRA:
    """Lower a ``LoraConfig`` to Bridge's ``LoRA``.

    ``config.target_modules=None`` takes ``recipe.lora_target_modules``. ``config.expert_adapters`` picks
    Bridge's layout for grouped MoE expert linears (``share_expert_adapters`` and
    ``experts_shared_outer_loras``); whatever a layout replicates across expert-parallel ranks is summed
    there by the gradient sync the engine installs. ``config.freeze_routing`` and
    ``config.skip_frozen_modules`` become Bridge LoRA's options of the same names. Bridge's other LoRA
    settings keep their defaults.

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
        freeze_routing=config.freeze_routing,
        skip_frozen_modules=config.skip_frozen_modules,
    )
