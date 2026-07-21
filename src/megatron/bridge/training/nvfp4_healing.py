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
from typing import TYPE_CHECKING, Any, cast

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
        raise RuntimeError("NVFP4 healing requires Transformer Engine >= 2.7.0.dev0, which is not installed.") from err
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
        """Switch training to the FP8 healing recipe (and pre-quantized FP8 weights)."""
        model = context.model
        model_config = self._get_model_config(model)
        self._reset_cuda_graphs()
        if self.config.pre_quantize_base_weights:
            if not self._fp8_stash:
                raise RuntimeError("FP8 healing weight stash is empty; pre-quantization did not run.")
            self._swap_in_fp8_weights(model)
        self._patch_fp4_recipe(self._build_healing_recipe(model_config))

    def _build_healing_recipe(self, model_config: TransformerConfig) -> Any:
        """Construct the FP8 recipe object used for the healing phase."""
        _require_te()
        from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling

        if self.config.healing_recipe == "delayed":
            return DelayedScaling(
                amax_history_len=self.config.fp8_amax_history_len,
                amax_compute_algo=self.config.fp8_amax_compute_algo,
                reduce_amax=self.config.reduce_amax,
                fp8_dpa=model_config.fp8_dot_product_attention,
            )
        return MXFP8BlockScaling()

    def _patch_fp4_recipe(self, recipe: Any) -> None:
        """Make ``megatron.core.fp4_utils.get_fp4_recipe`` return ``recipe``."""
        import megatron.core.fp4_utils as fp4_utils

        if not self._patched:
            self._original_get_fp4_recipe = fp4_utils.get_fp4_recipe
            self._patched = True

        def healing_get_fp4_recipe(config: Any) -> Any:
            return recipe

        fp4_utils.get_fp4_recipe = healing_get_fp4_recipe

    def _restore_fp4_recipe(self) -> None:
        """Restore the original ``get_fp4_recipe`` if it was patched."""
        if not self._patched:
            return
        import megatron.core.fp4_utils as fp4_utils

        fp4_utils.get_fp4_recipe = self._original_get_fp4_recipe
        self._patched = False

    def _reset_cuda_graphs(self) -> None:
        """Clear captured full-iteration CUDA graphs so they re-capture with the new recipe."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        try:
            from megatron.core.full_cuda_graph import FullCudaGraphWrapper
        except ImportError:
            return
        if not hasattr(FullCudaGraphWrapper, "cuda_graph"):
            return
        for stage in ("training", "validation"):
            FullCudaGraphWrapper.cuda_graph[stage] = None
            FullCudaGraphWrapper.result[stage] = None
            if self.config.reset_cuda_graph_warmup:
                FullCudaGraphWrapper.curr_iteration[stage] = 0

    # ------------------------------------------------------- pre-quantization

    def _build_quantizers(self) -> tuple[Any, Any]:
        """Build the NVFP4 quantizer and the FP8 quantizer matching ``healing_recipe``."""
        _require_te()
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

        nvfp4_quantizer = NVFP4Quantizer()
        if self.config.healing_recipe == "delayed":
            device = torch.cuda.current_device()
            fp8_quantizer = Float8Quantizer(
                scale=torch.ones(1, dtype=torch.float32, device=device),
                amax=torch.zeros(1, dtype=torch.float32, device=device),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
        else:
            fp8_quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        return nvfp4_quantizer, fp8_quantizer

    def _validate_quantized(self, qparam: Any, kind: str) -> None:
        """Check that a quantized tensor carries the storages its kind requires."""
        if kind in ("nvfp4", "mxfp8"):
            if qparam._rowwise_data is None or qparam._columnwise_data is None:
                raise RuntimeError(f"Quantized '{kind}' tensor is missing rowwise/columnwise data.")
        else:
            if qparam._data is None:
                raise RuntimeError("Quantized FP8 tensor is missing data.")

    def _stash_fp8_copy(self, qparam: Any) -> None:
        """Clone a quantized FP8 tensor and stash it (pinned host memory by default)."""
        qparam = qparam.clone()
        if not self.config.store_quantized_params_on_gpu:
            # Pinned host memory enables fast async H2D transfer at healing time.
            if self.config.healing_recipe == "mxfp8":
                qparam._rowwise_data = qparam._rowwise_data.cpu().pin_memory()
                qparam._columnwise_data = qparam._columnwise_data.cpu().pin_memory()
            else:
                qparam._data = qparam._data.cpu().pin_memory()
                if getattr(qparam, "_transpose", None) is not None:
                    qparam._transpose = qparam._transpose.cpu().pin_memory()
        self._fp8_stash.append(qparam)

    def _pre_quantize(self, model: list[torch.nn.Module]) -> None:
        """Replace frozen base weights with NVFP4 tensors and stash FP8 copies for healing."""
        nvfp4_quantizer, fp8_quantizer = self._build_quantizers()
        fp8_kind = self.config.healing_recipe
        count = 0
        with torch.no_grad():
            for _, module in self._iter_quantizable_modules(model):
                weight: Any = module.weight
                if weight.requires_grad:
                    raise ValueError(
                        "pre_quantize_base_weights=True requires frozen base weights "
                        f"(e.g. PEFT/LoRA fine-tuning); found a trainable weight on {type(module).__name__}."
                    )
                fp8_param = fp8_quantizer(weight.detach())
                self._validate_quantized(fp8_param, fp8_kind)
                self._stash_fp8_copy(fp8_param)

                nvfp4_param = nvfp4_quantizer(weight.detach())
                self._validate_quantized(nvfp4_param, "nvfp4")
                setattr(module, "weight", torch.nn.Parameter(nvfp4_param, requires_grad=False))
                count += 1
        print_rank_0(f"NVFP4 healing: pre-quantized {count} weights (NVFP4 on device, FP8 stashed).")

    # ------------------------------------------------------------ weight swap

    def _move_stash_entry_to_device(self, weight: Any, device: Any) -> None:
        """Asynchronously move a stashed quantized tensor's storages to ``device``."""
        if self.config.healing_recipe == "mxfp8":
            weight._rowwise_data = weight._rowwise_data.to(device, non_blocking=True)
            weight._columnwise_data = weight._columnwise_data.to(device, non_blocking=True)
        else:
            weight._data = weight._data.to(device, non_blocking=True)
            if getattr(weight, "_transpose", None) is not None:
                weight._transpose = weight._transpose.to(device, non_blocking=True)

    def _swap_in_fp8_weights(self, model: list[torch.nn.Module]) -> None:
        """Replace pre-quantized NVFP4 weights with the stashed FP8 weights."""
        modules = list(self._iter_quantizable_modules(model))
        if len(modules) != len(self._fp8_stash):
            raise RuntimeError(
                f"FP8 weight stash size ({len(self._fp8_stash)}) does not match the quantizable module "
                f"count ({len(modules)}); the model structure changed after pre-quantization."
            )
        model_config = self._get_model_config(model)
        device = torch.cuda.current_device()
        transfer_stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        swaps: list[tuple[torch.nn.Module, Any, Any]] = []
        fuser_layers: dict[int, Any] = {}

        with torch.no_grad():
            # Batch all H2D transfers on a dedicated stream, synchronize once, then swap
            # pointers - avoids per-weight synchronization stalls.
            with torch.cuda.stream(transfer_stream):
                for (layer, module), new_weight in zip(modules, self._fp8_stash):
                    if not self.config.store_quantized_params_on_gpu:
                        self._move_stash_entry_to_device(new_weight, device)
                    swaps.append((module, module.weight, new_weight))
                    if model_config.use_transformer_engine_op_fuser:
                        fuser_layers[id(layer)] = layer
            transfer_stream.synchronize()

            for module, old_weight, new_weight in swaps:
                setattr(module, "weight", torch.nn.Parameter(new_weight, requires_grad=False))
                if hasattr(old_weight, "clear"):
                    old_weight.clear()

            for layer in fuser_layers.values():
                layer.mlp._fused_impl = (layer.mlp._make_fused_impl(),)
                layer.self_attention.linear_proj._fused_branches = (
                    layer.self_attention.linear_proj._make_fused_branches()
                )
                layer.self_attention.linear_qkv._fused_branches = (
                    layer.self_attention.linear_qkv._make_fused_branches()
                )

        self._fp8_stash = []

    # ------------------------------------------------------------- traversal

    @staticmethod
    def _get_model_config(model: list[torch.nn.Module]) -> TransformerConfig:
        """Return the Megatron-Core ``TransformerConfig`` of the (unwrapped) first chunk."""
        unwrapped: Any = _unwrap_model_chunk(model[0])
        return cast("TransformerConfig", unwrapped.config)

    @staticmethod
    def _is_target_module(module: torch.nn.Module) -> bool:
        """Whether ``module`` is a quantizable Transformer Engine linear with a weight."""
        import transformer_engine.pytorch as te

        is_te_linear = bool(isinstance(module, (te.Linear, te.LayerNormLinear)))
        return is_te_linear and getattr(module, "weight", None) is not None

    @staticmethod
    def _keep_layer_in_bf16(global_layer_idx: int, model_config: TransformerConfig) -> bool:
        """Mirror Megatron-Core's first/last-layers-in-BF16 selection (see ``get_fp4_context``)."""
        if not model_config.first_last_layers_bf16:
            return False
        if global_layer_idx < model_config.num_layers_at_start_in_bf16:
            return True
        return bool(global_layer_idx >= model_config.num_layers - model_config.num_layers_at_end_in_bf16)

    def _iter_quantizable_modules(
        self, model: list[torch.nn.Module]
    ) -> Iterator[tuple[torch.nn.Module, torch.nn.Module]]:
        """Yield ``(layer, module)`` for every quantizable TE linear, in deterministic order.

        Iterates all model chunks (virtual pipeline stages) and uses each layer's global
        ``layer_number`` (1-based) so BF16 first/last-layer selection is correct under
        pipeline parallelism. Layers Megatron-Core keeps in BF16 are skipped.
        """
        model_config = self._get_model_config(model)
        for chunk in model:
            unwrapped: Any = _unwrap_model_chunk(chunk)
            try:
                layers = unwrapped.decoder.layers
            except AttributeError as err:
                raise RuntimeError(
                    "NVFP4HealingCallback requires a Megatron-Core GPT-style model exposing "
                    f"`decoder.layers`; got {type(unwrapped).__name__}."
                ) from err
            for local_idx, layer in enumerate(layers):
                global_idx = getattr(layer, "layer_number", local_idx + 1) - 1
                if self._keep_layer_in_bf16(global_idx, model_config):
                    continue
                for _, module in layer.named_modules():
                    if self._is_target_module(module):
                        yield layer, module
