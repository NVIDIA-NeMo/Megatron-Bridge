# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Backward gradient debug hooks for VLM models.

Usage:
    from megatron.bridge.utils.grad_debug_hooks import GradDebugHooks

    # After model is constructed:
    hooks = GradDebugHooks(model, log_every_n_steps=1, verbose=True)
    hooks.register()

    # During training loop, after loss.backward():
    hooks.print_summary()

    # To remove all hooks:
    hooks.remove()
"""

import logging
from collections import OrderedDict
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn


logger = logging.getLogger(__name__)


def _grad_stats(grad: Tensor) -> dict:
    """Compute gradient norm for a single tensor."""
    return {"norm": grad.float().norm().item()}


def _format_stats(name: str, stats: dict) -> str:
    """Format gradient norm into a readable string."""
    return f"  {name:70s} | norm={stats['norm']:.6e}"


class GradDebugHooks:
    """Register backward hooks on model components for gradient debugging.

    This registers hooks on:
      - Vision model: patch_embed, decoder layers, deepstack mergers, final merger
      - Language model: embedding, decoder layers, output layer
      - Key intermediate tensors via tensor hooks

    Args:
        model: The model instance (e.g. Qwen3VLModel, Ministral3Model).
        log_every_n_steps: Only log every N backward passes. Default 1 (every step).
        verbose: If True, log per-layer stats. If False, only log summary.
        rank_filter: If set, only print console output on this distributed rank.
            None means print on all ranks. Does NOT affect wandb logging or
            gradient collection.
        module_name_filter: If set, only hook modules whose name contains this substring.
    """

    def __init__(
        self,
        model: nn.Module,
        log_every_n_steps: int = 1,
        verbose: bool = True,
        rank_filter: Optional[int] = 0,
        module_name_filter: Optional[str] = None,
    ):
        self.model = model
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose
        self.rank_filter = rank_filter
        self.module_name_filter = module_name_filter

        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._grad_records: OrderedDict[str, dict] = OrderedDict()
        self._step_count = 0
        self._registered = False

    def _is_collection_step(self) -> bool:
        """Check if we should collect gradient stats on this step (all ranks)."""
        return self._step_count % self.log_every_n_steps == 0

    def _is_print_rank(self) -> bool:
        """Check if this rank should print console output."""
        if self.rank_filter is None:
            return True
        try:
            if dist.is_initialized() and dist.get_rank() != self.rank_filter:
                return False
        except RuntimeError:
            pass
        return True

    def _make_module_backward_hook(self, name: str):
        """Create a full backward hook for a module."""

        def hook(module, grad_input, grad_output):
            if not self._is_collection_step():
                return

            # Record grad_output stats (gradient w.r.t. module output)
            for i, g in enumerate(grad_output):
                if g is not None:
                    key = f"{name}.grad_output[{i}]"
                    self._grad_records[key] = _grad_stats(g)

            # Record grad_input stats (gradient w.r.t. module input)
            for i, g in enumerate(grad_input):
                if g is not None:
                    key = f"{name}.grad_input[{i}]"
                    self._grad_records[key] = _grad_stats(g)

        return hook

    def _make_tensor_hook(self, name: str):
        """Create a hook for a tensor's gradient."""

        def hook(grad):
            if not self._is_collection_step():
                return
            self._grad_records[name] = _grad_stats(grad)

        return hook

    def register(self):
        """Register backward hooks on all relevant model components."""
        if self._registered:
            logger.warning("GradDebugHooks already registered. Call remove() first.")
            return

        self._register_module_hooks()
        self._registered = True
        logger.info(f"GradDebugHooks: registered {len(self._handles)} backward hooks")

    def _should_hook_module(self, name: str) -> bool:
        """Check if a module should be hooked based on the name filter."""
        if self.module_name_filter is None:
            return True
        return self.module_name_filter in name

    def _register_module_hooks(self):
        """Register hooks on named modules throughout the model."""
        model = self.model

        # Hook all named modules with full backward hooks
        # This covers vision model, language model, all layers, mergers, etc.
        for name, module in model.named_modules():
            if not self._should_hook_module(name):
                continue

            # Skip container modules that just hold children — hook leaf-ish modules
            # that actually do computation
            if self._is_hookable_module(module):
                handle = module.register_full_backward_hook(
                    self._make_module_backward_hook(name)
                )
                self._handles.append(handle)

    def _is_hookable_module(self, module: nn.Module) -> bool:
        """Determine if a module is worth hooking (has parameters or is a leaf computation)."""
        # Hook modules that have parameters directly (linear, conv, norm, embedding, etc.)
        has_own_params = any(True for _ in module.parameters(recurse=False))
        if has_own_params:
            return True

        # Also hook specific container modules that are architecturally significant
        hookable_types = (
            # Qwen3 VL
            "Qwen3VLVisionPatchEmbed",
            "Qwen3VLVisionPatchMerger",
            "Qwen3VLVisionModel",
            "Qwen3VLVisionTransformerBlock",
            "Qwen3VLGPTModel",
            "Qwen3VLTransformerBlock",
            "Qwen3VLSelfAttention",
            # Ministral 3
            "Ministral3Model",
            "Mistral3MultiModalProjector",
            # Megatron core
            "TransformerLayer",
        )
        return type(module).__name__ in hookable_types

    def register_tensor_hook(self, tensor: Tensor, name: str):
        """Register a gradient hook on a specific tensor for tracking.

        Call this on intermediate tensors you want to monitor, e.g.:
            hooks.register_tensor_hook(combined_embeddings, "combined_embeddings")

        Args:
            tensor: The tensor to hook (must have requires_grad=True).
            name: A descriptive name for logging.
        """
        if tensor is not None and tensor.requires_grad:
            handle = tensor.register_hook(self._make_tensor_hook(name))
            self._handles.append(handle)

    def step(self):
        """Call after loss.backward() to increment the step counter.

        This must be called each training step for log_every_n_steps to work correctly.
        """
        self._step_count += 1

    def get_wandb_metrics(self, prefix: str = "grad_debug/") -> dict[str, float]:
        """Return gradient norms as a flat dict suitable for wandb.log().

        Each recorded gradient produces a metric:
            grad_debug/<module_name>/norm

        Args:
            prefix: Prefix for all metric keys. Default "grad_debug/".

        Returns:
            Flat dict of {metric_name: float} ready for wandb.log().
        """
        metrics = {}
        for name, stats in self._grad_records.items():
            key = f"{prefix}{name}"
            metrics[f"{key}/norm"] = stats["norm"]
        return metrics

    def log_to_wandb(self, wandb_writer, iteration: int, prefix: str = "grad_debug/"):
        """Log gradient stats directly to wandb.

        This is NOT gated by rank_filter — wandb_writer is already None on
        non-wandb ranks, so only the rank that owns wandb will actually log.

        Args:
            wandb_writer: The wandb module (from global_state.wandb_logger).
            iteration: Current training iteration for the x-axis.
            prefix: Prefix for all metric keys.
        """
        if wandb_writer is None:
            return

        metrics = self.get_wandb_metrics(prefix=prefix)
        if metrics:
            wandb_writer.log(metrics, iteration)

    def print_summary(self, wandb_writer=None, iteration: int = 0):
        """Print a summary of all recorded gradient statistics and optionally log to wandb.

        Call this after loss.backward() and before optimizer.step().

        Wandb logging happens on whatever rank has a non-None wandb_writer
        (typically rank N-1). Console printing is controlled by rank_filter
        (typically rank 0). Both are gated by log_every_n_steps.

        Args:
            wandb_writer: Optional wandb module for logging metrics. If provided,
                gradient stats are also logged to wandb.
            iteration: Current training iteration (used for wandb x-axis).
        """
        if not self._is_collection_step():
            self._grad_records.clear()
            self.step()
            return

        # Log to wandb — not rank-filtered; wandb_writer is already None on non-wandb ranks
        if wandb_writer is not None:
            self.log_to_wandb(wandb_writer, iteration)

        # Console print — only on rank_filter rank to avoid noisy output
        if self._is_print_rank():
            rank_str = ""
            try:
                if dist.is_initialized():
                    rank_str = f"[rank {dist.get_rank()}] "
            except RuntimeError:
                pass

            header = f"\n{'='*120}\n{rank_str}GRADIENT DEBUG SUMMARY (step {self._step_count})\n{'='*120}"

            lines = [header]

            if self._grad_records:
                if self.verbose:
                    lines.append(f"\n{'-'*60} Gradient Norms {'-'*60}")
                    for name, stats in self._grad_records.items():
                        lines.append(_format_stats(name, stats))
            else:
                lines.append("  (no gradients recorded — model might not have backward pass or hooks are on wrong modules)")

            lines.append(f"{'='*120}\n")
            logger.info("\n".join(lines))

        self._grad_records.clear()
        self.step()

    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._grad_records.clear()
        self._registered = False
        logger.info("GradDebugHooks: all hooks removed")
