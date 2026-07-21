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

"""NVFP4 training with FP8 healing.

This module provides :class:`NVFP4HealingCallback`, which implements the FP8-healing
technique for NVFP4 training: run most training steps in NVFP4 (see the
``bf16_with_nvfp4_mixed`` recipe in ``megatron.bridge.training.mixed_precision``), then
switch the quantization recipe to FP8 at a configurable step so the final phase of
training "heals" model quality. This is the technique used in NVIDIA's MLPerf
Llama-2 70B LoRA submission.

Optionally, when base weights are frozen (PEFT/LoRA), the callback pre-quantizes them:
NVFP4 tensors replace the BF16 weights on device for the NVFP4 phase (saving memory),
while FP8 copies are stashed in pinned host memory (or on device) and swapped in when
healing starts.

The recipe switch works by replacing ``megatron.core.fp4_utils.get_fp4_recipe`` with a
function returning the FP8 healing recipe. Megatron-Core resolves that function at call
time, so the change takes effect on the next forward pass (and on CUDA-graph re-capture,
which the callback triggers by resetting ``FullCudaGraphWrapper`` state). The original
function is restored in ``on_train_end``. ``config.fp4`` stays truthy throughout, so the
model keeps using the same quantization autocast code path it was built with.

Example:
    ```python
    from megatron.bridge.training.nvfp4_healing import NVFP4HealingCallback, NVFP4HealingConfig

    cfg.mixed_precision = "bf16_with_nvfp4_mixed"
    healing = NVFP4HealingCallback(
        NVFP4HealingConfig(
            healing_iter=350,
            healing_recipe="mxfp8",
            pre_quantize_base_weights=True,
        )
    )
    finetune(cfg, forward_step, callbacks=[healing])
    ```
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.utils.common_utils import print_rank_0


if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


logger: logging.Logger = logging.getLogger(__name__)

VALID_HEALING_RECIPES: tuple[str, ...] = ("delayed", "mxfp8")


@dataclass(kw_only=True)
class NVFP4HealingConfig:
    """Configuration for :class:`NVFP4HealingCallback`.

    Attributes:
        healing_iter: Number of training steps to run in NVFP4 before switching to FP8.
            Steps with index >= ``healing_iter`` (0-based) run with the FP8 healing
            recipe. Must be >= 1.
        healing_recipe: FP8 recipe used for healing. ``"delayed"`` selects Transformer
            Engine ``DelayedScaling``; ``"mxfp8"`` selects ``MXFP8BlockScaling``
            (Blackwell only). Naming matches ``MixedPrecisionConfig.fp8_recipe``.
        pre_quantize_base_weights: Pre-quantize base weights of Transformer Engine
            ``Linear``/``LayerNormLinear`` modules at startup: NVFP4 replaces the
            weights on device, FP8 copies are stashed for healing. Requires the target
            weights to be frozen (e.g. PEFT/LoRA fine-tuning).
        store_quantized_params_on_gpu: Keep the stashed FP8 weights on GPU instead of
            pinned host memory (trades GPU memory for a zero-copy healing switch).
        fp8_amax_history_len: ``DelayedScaling`` amax history length.
        fp8_amax_compute_algo: ``DelayedScaling`` amax compute algorithm.
        reduce_amax: ``DelayedScaling`` amax reduction across the amax group.
        reset_cuda_graph_warmup: After healing, also reset the CUDA-graph warmup
            counters so warmup steps run again before graph re-capture.
    """

    healing_iter: int
    healing_recipe: str
    pre_quantize_base_weights: bool = False
    store_quantized_params_on_gpu: bool = False
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "max"
    reduce_amax: bool = True
    reset_cuda_graph_warmup: bool = False

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.healing_recipe not in VALID_HEALING_RECIPES:
            raise ValueError(f"healing_recipe must be one of {VALID_HEALING_RECIPES}, got {self.healing_recipe!r}")
        if self.healing_iter < 1:
            raise ValueError(f"healing_iter must be >= 1, got {self.healing_iter}")


def _require_te() -> None:
    """Raise a descriptive error if Transformer Engine with NVFP4 support is unavailable."""
    from megatron.core.utils import is_te_min_version

    try:
        import transformer_engine  # noqa: F401
    except ImportError as err:
        raise RuntimeError(
            "NVFP4 healing requires Transformer Engine >= 2.7.0.dev0, which is not installed."
        ) from err
    if not is_te_min_version("2.7.0.dev0"):
        raise RuntimeError("NVFP4 healing requires Transformer Engine >= 2.7.0.dev0 for NVFP4BlockScaling support.")


def _unwrap_model_chunk(chunk: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/Float16Module-style wrappers by following ``.module`` attributes."""
    current = chunk
    while True:
        child = getattr(current, "module", None)
        if child is None:
            return current
        current = child


class NVFP4HealingCallback(Callback):
    """Training callback implementing NVFP4 training with FP8 healing.

    See the module docstring for the technique overview and a usage example. The
    callback overrides three events:

    - ``on_data_init_start``: optional base-weight pre-quantization.
    - ``on_train_step_end``: applies the healing switch once, at ``healing_iter``.
    - ``on_train_end``: restores the original Megatron-Core FP4 recipe function.
    """

    def __init__(self, config: NVFP4HealingConfig) -> None:
        """Initialize the callback with a validated config."""
        self.config = config
        self._healed: bool = False
        self._patched: bool = False
        self._original_get_fp4_recipe: Any = None
        self._fp8_stash: list[Any] = []

    @property
    def healed(self) -> bool:
        """Whether the FP8 healing switch has been applied."""
        return self._healed

    # ------------------------------------------------------------------ hooks

    def on_data_init_start(self, context: CallbackContext) -> None:
        """Pre-quantize frozen base weights if ``pre_quantize_base_weights`` is set."""
        if not self.config.pre_quantize_base_weights:
            return
        gc.collect()
        torch.cuda.empty_cache()
        self._pre_quantize(context.model)

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Apply the FP8 healing switch once, when ``healing_iter`` is reached."""
        if self._healed:
            return
        if context.state.train_state.step + 1 != self.config.healing_iter:
            return
        print_rank_0(f"NVFP4 healing: switching to FP8 ('{self.config.healing_recipe}') recipe...")
        self._apply_healing(context)
        self._healed = True

    def on_train_end(self, context: CallbackContext) -> None:
        """Restore the original Megatron-Core FP4 recipe function."""
        self._restore_fp4_recipe()

    # ---------------------------------------------------------------- healing

    def _apply_healing(self, context: CallbackContext) -> None:
        raise NotImplementedError  # implemented in a later task

    def _restore_fp4_recipe(self) -> None:
        raise NotImplementedError  # implemented in a later task

    def _pre_quantize(self, model: list[torch.nn.Module]) -> None:
        raise NotImplementedError  # implemented in a later task
