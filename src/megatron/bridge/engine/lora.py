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

Construction follows Bridge's own PEFT setup (``_create_peft_pre_wrap_hook`` in ``training/setup.py``):
the pretrained base weights are loaded into the bare model, LoRA then wraps the target modules and
freezes every base parameter, and only afterwards are DDP and the optimizer built, so both hold the
adapters alone. A LoRA engine therefore loads ``checkpoint_path`` inside its pre-wrap hook, as the value
head does, instead of after the build.

Routing stays the base model's. No router is a LoRA target, which is checked against the modules the
target names and wildcards resolve to, and the router state that ``finalize_model_grads`` updates even
with the router frozen is held at the base values: the expert bias is frozen and quantile balancing,
whose ``qb_beta`` has no such switch, is rejected. ``forward_only(disable_adapters=True)`` therefore
scores the base model exactly, and the adapters alone hold what training changed.

The rest of the engine needs three LoRA facts: ``finalize_model_grads`` must also sum the gradients of
expert adapters that are replicated across expert-parallel ranks (``peft_enabled`` when Bridge installs
the model-config hooks), ``forward_only(disable_adapters=True)`` runs under ``PEFT.disable_adapter``,
and checkpoints carry the LoRA object so Bridge saves and resumes the adapters only.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.moe.router import Router

from megatron.bridge.engine.checkpoint import load_pretrained_weights
from megatron.bridge.engine.datum import LORA_EXPERT_ADAPTER_LAYOUTS, LoraConfig
from megatron.bridge.engine.recipes.base import TrainingRecipe, is_multimodal
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.walk_utils import walk
from megatron.bridge.training.setup import _apply_peft_transformation


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
        ValueError: For a non-positive rank or alpha, a dropout outside ``[0, 1)``, an empty target list, a
            target naming the MoE router (routing stays the base model's, which router replay relies on),
            or an unknown ``expert_adapters`` layout. Wildcards that resolve to a router are caught by the
            pre-wrap hook, once the model exists.
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
    router_targets = [name for name in targets if "router" in name]
    if router_targets:
        problems.append(f"the MoE router cannot be a LoRA target: {router_targets}")
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
    """Pre-wrap hook: check the routers, load the base weights into the bare model, then apply LoRA.

    On a vision-language model, plain target names are first kept off the parts the recipe freezes
    (``_scope_to_trained_parts``). The router checks then run on the bare model: no router may match a
    target pattern, and quantile balancing is rejected. The base load runs with no PEFT in the checkpoint
    view, so every base tensor is read; LoRA then wraps the targets, freezes the base and records the
    adapter parameters it saves (``set_params_to_save``). Last, the routers' expert-bias updates are
    switched off.

    Args:
        recipe: The engine's recipe (checkpoint loading reads its model, optimizer and DDP sections).
        peft: The engine's ``LoRA``, from ``build_lora``.
        checkpoint_path: Megatron or HF checkpoint holding the pretrained base weights.

    Returns:
        The hook to register on ``recipe.model``.
    """

    def hook(model: list[MegatronModule]) -> list[MegatronModule]:
        _scope_to_trained_parts(peft, recipe, model)
        _check_routers(peft, model)
        load_pretrained_weights(recipe, model, None, checkpoint_path)
        model = _apply_peft_transformation(peft, model)
        _freeze_expert_bias(model)
        return model

    return hook


# The parts of Bridge's vision-language model (``Qwen3VLModel``) that the provider's freeze flags cover,
# as module-name prefixes in the bare model. The split is Bridge's own (``Qwen3VLModel.freeze``): the
# projection is the patch merger plus the deepstack mergers.
_VISION_LANGUAGE_PARTS = (
    ("freeze_language_model", ("language_model.",)),
    ("freeze_vision_model", ("vision_model.decoder.layers.",)),
    ("freeze_vision_projection", ("vision_model.merger.", "vision_model.decoder.deepstack_merger_list.")),
)


def _scope_to_trained_parts(peft: LoRA, recipe: TrainingRecipe, model: list[MegatronModule]) -> None:
    """On a vision-language recipe that freezes part of the model, keep plain target names off that part.

    A plain name (``linear_qkv``) matches its module anywhere, and the vision tower and the patch mergers
    use the language model's layer names. When the recipe freezes some of those parts, each plain name
    becomes one pattern per trained part that has a module of that name (``language_model.*.linear_qkv``,
    or ``vision_model.merger.linear_fc1`` for a direct child), so ``peft.target_modules`` is rewritten.
    Names containing ``*`` or ``.`` are the caller's own patterns and are kept as given. Nothing changes on
    a text model or when the recipe trains every part.

    Raises:
        ValueError: If the recipe freezes every part the targets could reach.
    """
    if not is_multimodal(recipe):
        return
    frozen = [flag for flag, _ in _VISION_LANGUAGE_PARTS if getattr(recipe.model, flag)]
    if not frozen:
        return
    trained_prefixes = [
        prefix for flag, prefixes in _VISION_LANGUAGE_PARTS if flag not in frozen for prefix in prefixes
    ]
    plain = [target for target in peft.target_modules if "*" not in target and "." not in target]
    scoped = [target for target in peft.target_modules if target not in plain]
    module_names = [name for chunk in model for name, _ in chunk.named_modules()]
    for prefix in trained_prefixes:
        relative = [name[len(prefix) :] for name in module_names if name.startswith(prefix)]
        for target in plain:
            if target in relative:  # a direct child of the part
                scoped.append(f"{prefix}{target}")
            if any(rel.endswith(f".{target}") for rel in relative):
                scoped.append(f"{prefix}*.{target}")
    if not scoped:
        raise ValueError(
            f"the recipe freezes every part LoRA targets {list(peft.target_modules)} could reach "
            f"({', '.join(frozen)}): nothing would be trained."
        )
    peft.target_modules = scoped


def _check_routers(peft: LoRA, model: list[MegatronModule]) -> None:
    """Raise if a target pattern resolves to an MoE router, or if a router uses quantile balancing.

    Matching goes through Bridge's own walk and ``LoRA.match``, so names and wildcards resolve exactly as
    they will when LoRA is applied (``PEFT.__call__`` resets the match bookkeeping this walk records).
    """
    matched: list[str] = []
    quantile_balanced: list[str] = []

    def check(module: torch.nn.Module, name: str | None = None, prefix: str | None = None) -> torch.nn.Module:
        if isinstance(module, Router):
            full_name = f"{prefix}.{name}" if prefix else name
            if peft.match(module, name, prefix) is not None:
                matched.append(full_name)
            if getattr(module, "qb_beta", None) is not None:
                quantile_balanced.append(full_name)
        return module

    for chunk in model:
        walk(chunk, check)
    if matched:
        raise ValueError(
            f"the MoE router cannot be a LoRA target, but target_modules={list(peft.target_modules)} matches "
            f"{len(matched)} router(s), e.g. {matched[0]}."
        )
    if quantile_balanced:
        raise NotImplementedError(
            "LoRA with quantile-balanced routing is not supported: finalize_model_grads updates the routers' "
            f"qb_beta during training (e.g. {quantile_balanced[0]}), so routing would drift from the base model."
        )


def _freeze_expert_bias(model: list[MegatronModule]) -> None:
    """Stop ``finalize_model_grads`` from updating the routers' expert bias (Megatron-Core's switch).

    The update runs whenever ``moe_router_enable_expert_bias`` is set, router weights frozen or not, so
    without this the policy's routing would drift from the base model's: ``disable_adapters`` would no
    longer score the base, and an adapter-only export would leave out the moved biases.
    """
    for chunk in model:
        for module in chunk.modules():
            if hasattr(module, "frozen_expert_bias"):
                module.frozen_expert_bias = True
