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

import logging
from typing import Any

from omegaconf import OmegaConf

from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_delayed_scaling_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
)

from .common import get_perf_matrix_overrides


logger = logging.getLogger(__name__)


COMM_OVERLAP_CONFIG_MAP = {
    "llama3_70b": {
        "h100": {
            "bf16": userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        },
    },
    "llama31_405b": {
        "h100": {
            "bf16": userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
    },
}


def set_megatron_fsdp_overrides(recipe: Any, perf_overrides: Any) -> None:
    """Set the mcore fsdp overrides from the performance matrix."""
    use_megatron_fsdp = perf_overrides.get("use_megatron_fsdp", False)
    if use_megatron_fsdp:
        recipe.ddp.use_megatron_fsdp = True
        recipe.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        recipe.ddp.keep_fp8_transpose_cache = False
        # average_in_collective is not supported with Megatron FSDP
        recipe.ddp.average_in_collective = False

        recipe.model.init_model_with_meta_device = True
        recipe.model.gradient_accumulation_fusion = True

        if recipe.comm_overlap is not None and isinstance(recipe.comm_overlap, CommOverlapConfig):
            if recipe.comm_overlap.defer_embedding_wgrad_compute:
                logger.warning(
                    "Disabling deferring embedding wgrad compute because it cannot work with FSDP together."
                )
                recipe.comm_overlap.defer_embedding_wgrad_compute = False

        if recipe.optimizer.use_precision_aware_optimizer:
            recipe.optimizer.use_precision_aware_optimizer = False
            logger.warning("Disabling precision aware optimizer because it cannot work with FSDP together.")


def get_precision_config(compute_dtype: str, fp8_recipe: str):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    if compute_dtype == "fp8":
        if fp8_recipe == "ds":
            return bf16_with_fp8_delayed_scaling_mixed()
        elif fp8_recipe == "cs":
            current_scaling_cfg = bf16_with_fp8_current_scaling_mixed()
            # Disable BF16 Transformer layers in the performance config
            current_scaling_cfg.first_last_layers_bf16 = False
            return current_scaling_cfg
        elif fp8_recipe == "mx":
            return bf16_with_mxfp8_mixed()
        elif fp8_recipe == "ss":
            return bf16_with_fp8_subchannel_scaling_mixed()
        else:
            raise ValueError(f"Invalid FP8 recipe: {fp8_recipe}")
    elif compute_dtype == "bf16":
        return bf16_mixed()
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype}")


def set_cuda_graph_overrides(recipe: Any, perf_overrides: Any) -> None:
    """Set the CUDA graph overrides from the performance matrix."""
    enable_cuda_graph = perf_overrides.get("cuda_graphs", False)

    recipe.model.enable_cuda_graph = enable_cuda_graph
    recipe.model.use_te_rng_tracker = enable_cuda_graph
    recipe.rng.te_rng_tracker = enable_cuda_graph


def set_recompute_overrides(recipe: Any, perf_overrides: Any) -> None:
    """Set the recompute num layers overrides from the performance matrix."""
    recompute_num_layers = perf_overrides.get("recompute_num_layers", None)
    if recompute_num_layers is not None:
        recipe.model.recompute_granularity = "full"
        recipe.model.recompute_method = "block"
        recipe.model.recompute_num_layers = recompute_num_layers

    cpu_offloading_num_layers = perf_overrides.get("cpu_offloading_num_layers", 0)
    if cpu_offloading_num_layers > 0:
        recipe.model.cpu_offloading = True
        recipe.model.cpu_offloading_weights = False
        recipe.model.cpu_offloading_num_layers = cpu_offloading_num_layers


def apply_perf_matrix_overrides(yaml_root: Any, recipe: Any, args: Any, excluded_fields: dict[str, Any]) -> None:
    """Apply GPU/precision-specific overrides from a unified YAML's perf_matrix."""
    preset = get_perf_matrix_overrides(yaml_root, args)
    if not preset:
        num_gpus_yaml_key = f"num_gpus_{args.num_gpus or args.gpus_per_node}"
        logger.debug(f"No preset found for {args.gpu}.{num_gpus_yaml_key} in perf_matrix; skipping perf overrides")
        return

    common = preset.get("common") or {}
    compute_dtype = args.compute_dtype if args.compute_dtype == "bf16" else f"{args.compute_dtype}_{args.fp8_recipe}"
    dtype_cfg = preset.get(compute_dtype) if compute_dtype in preset else None

    # Deep-merge so dtype-specific values override common
    merged_perf = OmegaConf.merge(OmegaConf.create(common), OmegaConf.create(dtype_cfg or {}))
    perf_overrides: dict[str, Any] = OmegaConf.to_container(merged_perf, resolve=True)  # type: ignore

    recipe.train.micro_batch_size = perf_overrides.get("mbs", recipe.train.micro_batch_size)
    recipe.train.global_batch_size = perf_overrides.get("gbs", recipe.train.global_batch_size)
    recipe.dataset.sequence_length = perf_overrides.get("seq_length", recipe.dataset.sequence_length)

    recipe.model.tensor_model_parallel_size = perf_overrides.get("tp", 1)
    recipe.model.pipeline_model_parallel_size = perf_overrides.get("pp", 1)
    recipe.model.virtual_pipeline_model_parallel_size = perf_overrides.get("vp", None)
    recipe.model.context_parallel_size = perf_overrides.get("cp", 1)
    recipe.model.expert_model_parallel_size = perf_overrides.get("ep", 1)
    recipe.model.expert_tensor_parallel_size = perf_overrides.get("etp", None)

    set_megatron_fsdp_overrides(recipe, perf_overrides)
    set_cuda_graph_overrides(recipe, perf_overrides)
    set_recompute_overrides(recipe, perf_overrides)

    recipe.model.sequence_parallel = bool(recipe.model.tensor_model_parallel_size > 1)
