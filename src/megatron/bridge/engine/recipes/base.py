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
"""TrainingRecipe: the model and execution configuration a TrainingEngine is built from.

A recipe holds only what defines how a model runs on the target hardware: the model provider
(parallelism, MoE, MTP, recompute), optimizer defaults, DDP settings, precision, distributed-init
and RNG settings. Data, learning-rate schedule, logging and checkpoint paths belong to the
application loop and are deliberately absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import _enable_safe_hybridep_dispatch
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    DistributedDataParallelConfig,
    DistributedInitConfig,
    OptimizerConfig,
    RNGConfig,
    _validate_and_sync_distributed_optimizer_settings,
    _validate_fine_grained_activation_offloading,
    _validate_mixed_precision_consistency,
)
from megatron.bridge.training.flex_dispatcher_backend import validate_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, get_mixed_precision_config
from megatron.bridge.utils.common_utils import get_world_size_safe
from megatron.bridge.utils.cuda_graph import validate_cuda_graph_configuration
from megatron.bridge.utils.gdn_utils import validate_cudnn_gdn_stack


@dataclass
class TrainingRecipe:
    """Validated model and execution configuration for one supported model on one hardware target.

    Attributes:
        model: Model provider; it is also the model's ``TransformerConfig`` (parallel sizes, MoE, MTP,
            recompute, fused kernels, ``seq_length`` and ``vocab_size``).
        optimizer: Optimizer type and defaults (lr, betas, eps, weight decay, clipping, distributed
            optimizer). Per-step values are overridden through ``AdamParams`` at ``optim_step``.
        ddp: Gradient-buffer, sharding and overlap settings.
        mixed_precision: Precision recipe (``bf16_mixed()`` etc.) or its registered name.
        dist: Distributed initialization settings (timeouts, process-group options).
        rng: Seeds and RNG tracker options.
        comm_overlap: Optional communication-overlap settings.
        env_vars: Environment defaults applied before initialization; existing values win.
        batch_layout: How the engine lays out Datums for the kernels. ``"thd"`` packs several Datums
            along the sequence axis (batch 1, ``PackedSeqParams`` boundaries); ``"bshd"`` runs one Datum
            per forward pass as a padded ``[1, S]`` row with dense causal attention.
        pad_to_max_length: THD only. ``False`` pads each pack just to the CP/TP multiple, so microbatch
            widths vary with their content (what RL rollouts normally do). ``True`` pads every pack to
            ``model.seq_length`` so kernels always see one shape; wastes compute on padding but avoids
            per-shape kernel autotuning, which can dominate step time on some stacks.
    """

    model: ModelProviderMixin
    optimizer: OptimizerConfig
    ddp: DistributedDataParallelConfig
    mixed_precision: MixedPrecisionConfig | str
    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng: RNGConfig = field(default_factory=RNGConfig)
    comm_overlap: CommOverlapConfig | None = None
    env_vars: dict[str, str | int | float | bool] = field(default_factory=dict)
    batch_layout: Literal["thd", "bshd"] = "thd"
    pad_to_max_length: bool = False


SUPPORTED_BATCH_LAYOUTS = ("thd", "bshd")


def data_parallel_size(recipe: TrainingRecipe) -> int:
    """Data-parallel degree implied by the recipe's model parallelism and the job's world size."""
    model = recipe.model
    model_parallel_size = (
        model.tensor_model_parallel_size * model.pipeline_model_parallel_size * (model.context_parallel_size or 1)
    )
    world_size = get_world_size_safe()
    if world_size % model_parallel_size:
        raise ValueError(f"world size {world_size} is not divisible by TP x PP x CP = {model_parallel_size}.")
    return world_size // model_parallel_size


def check_supported(recipe: TrainingRecipe) -> None:
    """Reject execution modes the engine does not support, before anything is initialized."""
    problems: list[str] = []
    if getattr(recipe.dist, "lazy_mpu_init", False):
        problems.append("dist.lazy_mpu_init is not supported")
    # TODO(mfsdp-v2): support Megatron FSDP v2. Accumulating gradients across forward_backward calls
    # already works with plain FSDP sharding, because v2 reduces and adds each backward into a
    # persistent gradient buffer without rescaling the total. Two things are missing:
    #   - engine side: zero grads through the optimizer, scale them after finish_grad_sync, and release
    #     the FSDP hooks in close();
    #   - FSDP side (hybrid sharding only): v2 reduces the cross-replica axis only inside the backward
    #     it is told is the last one (is_last_microbatch=True). The engine cannot know that during
    #     forward_backward, so it needs a method to run that reduction on demand at optim_step (before
    #     finish_grad_sync).
    if (
        getattr(recipe.dist, "use_megatron_fsdp", False)
        or getattr(recipe.dist, "use_torch_fsdp2", False)
        or getattr(recipe.ddp, "use_megatron_fsdp", False)
    ):
        problems.append("FSDP wrappers are not supported; use DDP with the distributed optimizer")
    if getattr(recipe.model, "vocab_size", None) is None:
        problems.append("model.vocab_size must be set (the engine does not build a tokenizer)")
    if getattr(recipe.model, "seq_length", None) is None:
        problems.append("model.seq_length must be set (packed token budget per microbatch)")
    if getattr(recipe, "batch_layout", "thd") not in SUPPORTED_BATCH_LAYOUTS:
        problems.append(f"batch_layout must be one of {SUPPORTED_BATCH_LAYOUTS}")
    if problems:
        raise ValueError("Recipe is not supported by TrainingEngine:\n  - " + "\n  - ".join(problems))


def finalize_recipe(recipe: TrainingRecipe) -> None:
    """Resolve derived settings, apply engine conventions, and validate the recipe in place.

    Mirrors the parts of Bridge's ``runtime_config_update`` and ``ConfigContainer.validate`` that
    concern the model, optimizer, DDP, precision and distributed sections. Engine conventions are
    set here: per-token loss sums (finalization deferred to ``optim_step``), no DP averaging in
    the collective, and HybridEP padding for packed (THD) input.
    """
    check_supported(recipe)
    model = recipe.model

    for name, value in recipe.env_vars.items():
        os.environ.setdefault(name, str(value))

    if isinstance(recipe.mixed_precision, str):
        recipe.mixed_precision = get_mixed_precision_config(recipe.mixed_precision)
    recipe.mixed_precision.finalize()
    recipe.mixed_precision.setup(model, recipe.optimizer, recipe.ddp)

    if recipe.comm_overlap is not None:
        recipe.comm_overlap.data_parallel_size = data_parallel_size(recipe)
        recipe.comm_overlap.finalize()
        recipe.comm_overlap.setup(model, recipe.optimizer, recipe.ddp)

    model.calculate_per_token_loss = True
    recipe.ddp.average_in_collective = False
    if recipe.batch_layout == "thd":
        _enable_safe_hybridep_dispatch(model, uses_thd=True)
    else:
        # TODO(bshd): validate dense-layout requirements (CP alignment per Datum, no PackedSeqParams consumers).
        pass

    if hasattr(recipe.ddp, "finalize"):
        recipe.ddp.finalize()
    if hasattr(recipe.optimizer, "finalize"):
        recipe.optimizer.finalize()
    validate_cuda_graph_configuration(model)
    model.finalize()
    if recipe.rng.te_rng_tracker or model.use_te_rng_tracker:
        model.use_te_rng_tracker = recipe.rng.te_rng_tracker = True

    cp = model.context_parallel_size or 1
    if cp > 1 and model.seq_length % (2 * cp):
        raise ValueError(f"model.seq_length ({model.seq_length}) must be divisible by 2 x CP ({2 * cp}).")
    _validate_and_sync_distributed_optimizer_settings(recipe)
    _validate_mixed_precision_consistency(recipe)
    _validate_fine_grained_activation_offloading(recipe)
    validate_cudnn_gdn_stack(model)
    validate_flex_dispatcher_backend(model)
