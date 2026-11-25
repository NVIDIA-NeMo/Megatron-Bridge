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

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from megatron.training.models.base import ModelConfig

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
from megatron.bridge.training.gtp import is_gtp_remat_active
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, get_mixed_precision_config
from megatron.bridge.utils.common_utils import get_world_size_safe
from megatron.bridge.utils.cuda_graph import validate_cuda_graph_configuration
from megatron.bridge.utils.gdn_utils import validate_cudnn_gdn_stack


# Bridge's ``LoRA.target_modules`` default: attention QKV and output projections, MLP (and expert) FC1/FC2.
BRIDGE_LORA_TARGET_MODULES = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")


@dataclass
class TrainingRecipe:
    """Validated model and execution configuration for one supported model on one hardware target.

    Attributes:
        model: Builder-backed model config (``AutoBridge.get_model_config()``) or, for families still on
            the provider path, a model provider. Either exposes the ``TransformerConfig`` fields
            (parallel sizes, MoE, MTP, recompute, fused kernels) and ``seq_length`` and ``vocab_size``.
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
        mtp_target_scope: Positions the MTP auxiliary loss trains on, when ``model.mtp_num_layers`` is
            set. ``"primary"`` uses the primary objective's positions (``loss_mask & ~padding_mask``, the
            SFT baseline). ``"all_valid"`` uses every non-padded position, as in pretraining; the MTP
            term then carries more total weight relative to the primary loss (it is still divided by the
            primary denominator). Padding is excluded either way; the user's loss methods are unaffected.
        hf_path: HF checkpoint directory or model ID matching ``model``; ``export`` reads its config,
            tokenizer files and weight layout from it. ``None`` disables ``export``. A text-only model
            needs the text-only causal-LM checkpoint; a multimodal wrapper (one with ``text_config``)
            is accepted only for a multimodal model.
        hf_revision: HF revision of ``hf_path`` when it is a model ID; ``None`` for a local directory.
        router_replay: Rollout routing replay for MoE models. ``True`` builds the routers with Megatron's
            ``moe_enable_routing_replay`` and turns the fused router off (the fused top-k ignores
            replay); the engine then replays ``routed_experts`` whenever a prepared microbatch carries
            it (the default pipeline fills it from ``Datum.routed_experts``). ``False`` rejects the field.
        lora_target_modules: Megatron module names a ``LoraConfig`` without ``target_modules`` attaches
            adapters to. Bridge's four default linears, extended by a recipe whose model has other
            linears worth adapting (Qwen3.5 adds its GatedDeltaNet ``in_proj`` and ``out_proj``).
        router_replay_layer_index: Maps the zero-based Megatron decoder layer index of an MoE layer to the
            layer index a generation backend records its routing under (the HF layer). ``None`` when the
            two are the same; a model whose Megatron layout splits one HF layer into several sets it.
    """

    model: ModelConfig | ModelProviderMixin
    optimizer: OptimizerConfig
    ddp: DistributedDataParallelConfig
    mixed_precision: MixedPrecisionConfig | str
    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    rng: RNGConfig = field(default_factory=RNGConfig)
    comm_overlap: CommOverlapConfig | None = None
    env_vars: dict[str, str | int | float | bool] = field(default_factory=dict)
    batch_layout: Literal["thd", "bshd"] = "thd"
    pad_to_max_length: bool = False
    mtp_target_scope: Literal["primary", "all_valid"] = "primary"
    hf_path: str | None = None
    hf_revision: str | None = None
    router_replay: bool = False
    lora_target_modules: list[str] = field(default_factory=lambda: list(BRIDGE_LORA_TARGET_MODULES))
    router_replay_layer_index: Callable[[int], int] | None = None


SUPPORTED_BATCH_LAYOUTS = ("thd", "bshd")
SUPPORTED_MTP_TARGET_SCOPES = ("primary", "all_valid")
GTP_NOT_SUPPORTED = (
    "GTP weight rematerialization is not supported (tensor_parallel_num_weight_shards and "
    "expert_tensor_parallel_num_weight_shards must equal the tensor-parallel sizes)"
)


def _model_parallel_sizes(recipe: TrainingRecipe) -> tuple[int, int]:
    """(TP x PP x CP, ETP x EP x PP): the two rank grids Megatron's ``initialize_model_parallel`` needs
    the world size to be divisible by."""
    model = recipe.model
    tp = model.tensor_model_parallel_size
    pp = model.pipeline_model_parallel_size
    cp = model.context_parallel_size or 1
    ep = getattr(model, "expert_model_parallel_size", 1) or 1
    etp = getattr(model, "expert_tensor_parallel_size", None) or tp
    return tp * pp * cp, etp * ep * pp


def min_gpus(recipe: TrainingRecipe) -> int:
    """Smallest world size the recipe runs on (one data-parallel replica of both rank grids).

    Any multiple of it is accepted; the extra GPUs become data parallelism.
    """
    dense, expert = _model_parallel_sizes(recipe)
    return math.lcm(dense, expert)


def data_parallel_size(recipe: TrainingRecipe) -> int:
    """Data-parallel degree implied by the recipe's model parallelism and the job's world size."""
    dense, expert = _model_parallel_sizes(recipe)
    unit = min_gpus(recipe)
    world_size = get_world_size_safe()
    if world_size % unit:
        raise ValueError(
            f"world size {world_size} is not a multiple of the recipe's minimum GPU count {unit} "
            f"(TP x PP x CP = {dense}, ETP x EP x PP = {expert})."
        )
    return world_size // dense


def is_multimodal(recipe: TrainingRecipe) -> bool:
    """Whether the recipe builds a multimodal model (its model config carries a vision config)."""
    return getattr(recipe.model, "vision_config", None) is not None


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
    # TODO(gtp): support GTP weight rematerialization. It has never been run through the engine, and
    # its setup is process-global while the engine builds a policy and a frozen model in one process.
    if is_gtp_remat_active(recipe.model):
        problems.append(GTP_NOT_SUPPORTED)
    if (getattr(recipe.model, "pipeline_model_parallel_size", 1) or 1) > 1:
        # Megatron runs microbatch_group_size_per_vp_stage microbatches through a virtual stage at a time
        # and needs at least one per pipeline stage in a group.
        group_size = getattr(recipe.model, "microbatch_group_size_per_vp_stage", None)
        if (
            getattr(recipe.model, "virtual_pipeline_model_parallel_size", None) is not None
            and group_size is not None
            and group_size < recipe.model.pipeline_model_parallel_size
        ):
            problems.append(
                f"model.microbatch_group_size_per_vp_stage ({group_size}) must be at least the pipeline size "
                f"({recipe.model.pipeline_model_parallel_size})"
            )
        # Stages exchange each pack's hidden states. Megatron sizes the receiving buffers once per call
        # from one pack length, unless model.variable_seq_lengths has every transfer announce its shape.
        same_length = getattr(recipe, "batch_layout", "thd") != "thd" or getattr(recipe, "pad_to_max_length", False)
        if not same_length and not getattr(recipe.model, "variable_seq_lengths", False):
            problems.append(
                "pipeline parallelism with content-sized packs needs model.variable_seq_lengths=True "
                "(or pad_to_max_length=True)"
            )
    if getattr(recipe.model, "vocab_size", None) is None:
        problems.append("model.vocab_size must be set (the engine does not build a tokenizer)")
    if getattr(recipe.model, "seq_length", None) is None:
        problems.append("model.seq_length must be set (packed token budget per microbatch)")
    if getattr(recipe, "batch_layout", "thd") not in SUPPORTED_BATCH_LAYOUTS:
        problems.append(f"batch_layout must be one of {SUPPORTED_BATCH_LAYOUTS}")
    if getattr(recipe, "mtp_target_scope", "primary") not in SUPPORTED_MTP_TARGET_SCOPES:
        problems.append(f"mtp_target_scope must be one of {SUPPORTED_MTP_TARGET_SCOPES}")
    # model.mtp_detach_heads is accepted: the head trains on MCore's per-token MTP loss divided by the
    # primary count; see TODO(detached-mtp) in TrainingEngine.optim_step for the exact pooled mean.
    if getattr(recipe, "router_replay", False):
        if not getattr(recipe.model, "num_moe_experts", None):
            problems.append("router_replay needs an MoE model (model.num_moe_experts)")
        if getattr(recipe.model, "moe_router_load_balancing_type", "aux_loss") in ("sinkhorn",):
            problems.append("router_replay is not supported with sinkhorn routing (it bypasses the replay hook)")
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
    if hasattr(model, "max_seq_length_per_sample") and model.max_seq_length_per_sample is None:
        # As Megatron's launcher sets it from --seq-length: the NCCL EP dispatcher sizes its buffers from it.
        model.max_seq_length_per_sample = model.seq_length
    recipe.ddp.average_in_collective = False
    if recipe.router_replay:
        model.moe_enable_routing_replay = True
        model.moe_router_fusion = False  # the fused top-k path never consults the replay hook
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
    # finalize derives the GTP sizes from the weight-shard counts, which a recipe may set instead.
    if is_gtp_remat_active(model):
        raise ValueError(f"Recipe is not supported by TrainingEngine:\n  - {GTP_NOT_SUPPORTED}")
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
