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

"""Resolve recipe-owned forward-step defaults without importing model code eagerly."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from collections.abc import Callable

    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.forward_step_func_types import ForwardStepCallable

logger = logging.getLogger(__name__)


def get_forward_step_reference(config: ConfigContainer) -> str | None:
    """Return the recipe default for the effective dataset, after config overrides."""
    reference = getattr(config, "forward_step", None)
    if reference is None:
        return None
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("config.forward_step must be a nonempty module:attribute reference or None")
    # Native Energon packing emits THD metadata consumed by the generic VLM step.
    # Preserve the runner's existing late selection when callers replace the dataset.
    if (
        reference == "megatron.bridge.models.qwen_vl.qwen3_vl_step:forward_step"
        and getattr(config.dataset, "packing_buffer_size", None) is not None
    ):
        return "megatron.bridge.training.vlm_step:forward_step"
    return reference


def load_forward_step_reference(reference: str, *, mode: str = "pretrain") -> ForwardStepCallable:
    """Load a function or construct a step class using the existing target policy.

    Config references use ``module:attribute``. Custom targets must be allowed by
    Bridge's existing config target policy; callers can also pass a Python callable
    directly to pretrain/finetune. Class-based steps retain mode-aware construction.
    """
    module, separator, attribute = reference.partition(":")
    if not separator or not module or not attribute or ":" in attribute:
        raise ValueError(f"Invalid config.forward_step {reference!r}; expected module:attribute")

    from megatron.bridge.utils.instantiate_utils import _resolve_target

    step = _resolve_target(f"{module}.{attribute}", full_key="forward_step")
    if inspect.isclass(step):
        # Constructor signatures are checked at runtime; classes narrow to type[object] statically.
        factory = cast("Callable[..., ForwardStepCallable]", step)
        step = factory(mode=mode) if "mode" in inspect.signature(step.__init__).parameters else factory()
    if not callable(step):
        raise TypeError(f"config.forward_step {reference!r} must resolve to a callable step")
    return cast("ForwardStepCallable", step)


def resolve_forward_step(
    config: ConfigContainer,
    forward_step_func: ForwardStepCallable | None = None,
    *,
    mode: str = "pretrain",
) -> ForwardStepCallable:
    """Use an explicit callable, otherwise resolve and record the recipe default."""
    if forward_step_func is not None:
        if not callable(forward_step_func):
            raise TypeError("forward_step_func must be callable")
        return forward_step_func
    reference = get_forward_step_reference(config)
    if reference is None:
        raise ValueError("No forward step configured; set config.forward_step or pass forward_step_func")
    step = load_forward_step_reference(reference, mode=mode)
    config.forward_step = reference
    logger.info("Using recipe forward step %s", reference)
    return step
