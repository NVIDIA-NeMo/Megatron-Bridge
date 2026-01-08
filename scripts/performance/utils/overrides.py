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

import argparse
import logging
from typing import List, Optional

from omegaconf import OmegaConf

from megatron.bridge.recipes.deepseek.deepseek_v3 import set_deepseek_v3_pipeline_model_parallel_layout
from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from utils.datasets import (
    create_mock_dataset_config,
    create_rp2_dataset_config,
    create_squad_dataset_config,
)
from utils.utils import WorkloadBaseConfig, get_workload_base_config


logger = logging.getLogger(__name__)


def _set_common_perf_overrides(recipe: ConfigContainer) -> ConfigContainer:
    """Set the common performance overrides."""
    recipe.train.train_iters = 50
    recipe.train.eval_iters = 0

    recipe.checkpoint.save = None

    recipe.logger.log_interval = 1
    recipe.logger.tensorboard_dir = None
    recipe.logger.save_config_filepath = "/nemo_run/configs/ConfigContainer.yaml"

    recipe.ddp.check_for_nan_in_grad = False
    recipe.ddp.check_for_large_grads = False

    recipe.rerun_state_machine.check_for_nan_in_loss = False

    recipe.scheduler.lr_decay_iters = recipe.train.train_iters
    recipe.scheduler.lr_warmup_iters = 10

    if hasattr(recipe.model, "use_transformer_engine_op_fuser") and recipe.model.use_transformer_engine_op_fuser:
        recipe.model.use_transformer_engine_op_fuser = False
    recipe.model.apply_rope_fusion = True
    recipe.model.cross_entropy_fusion_impl = "te"

    # TODO: This needs to be adjusted when overlapping HybridEP with computation or
    # the number of SMs for HybridEP is reduced.
    if recipe.model.moe_flex_dispatcher_backend == "hybridep":
        recipe.model.moe_hybridep_num_sms = 32

    return recipe


def _set_megatron_fsdp_overrides(
    recipe: ConfigContainer, use_megatron_fsdp: bool = False, nccl_ub: bool = False
) -> ConfigContainer:
    """Set the Megatron FSDP overrides."""
    if not use_megatron_fsdp:
        return

    recipe.ddp.use_megatron_fsdp = True
    recipe.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    recipe.ddp.keep_fp8_transpose_cache = False
    # average_in_collective is not supported with Megatron FSDP
    recipe.ddp.average_in_collective = False

    if nccl_ub:
        recipe.ddp.nccl_ub = True
        recipe.ddp.fsdp_manual_registration = True

    recipe.model.init_model_with_meta_device = True
    recipe.model.gradient_accumulation_fusion = True

    if recipe.comm_overlap is not None and isinstance(recipe.comm_overlap, CommOverlapConfig):
        if recipe.comm_overlap.defer_embedding_wgrad_compute:
            logger.warning("Disabling deferring embedding wgrad compute because it cannot work with FSDP together.")
            recipe.comm_overlap.defer_embedding_wgrad_compute = False

    if recipe.optimizer.use_precision_aware_optimizer:
        recipe.optimizer.use_precision_aware_optimizer = False
        logger.warning("Disabling precision aware optimizer because it cannot work with FSDP together.")

    recipe.checkpoint.load = None
    return recipe


def _set_cuda_graph_overrides(
    recipe: ConfigContainer, cuda_graph_impl: Optional[str] = None, cuda_graph_scope: Optional[str | List[str]] = None
) -> ConfigContainer:
    """Set the CUDA graph overrides."""
    if isinstance(cuda_graph_scope, str):
        cuda_graph_scope = [cuda_graph_scope]
    if cuda_graph_impl is not None:
        recipe.model.cuda_graph_impl = cuda_graph_impl
        if cuda_graph_impl != "none":
            recipe.rng.te_rng_tracker = recipe.model.use_te_rng_tracker = True
        else:  # this condition ensures we unset in case of user override to "none" from default
            recipe.rng.te_rng_tracker = recipe.model.use_te_rng_tracker = False

        if cuda_graph_impl == "transformer_engine":
            valid_te_scopes = ["attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"]
            assert all(scope in valid_te_scopes for scope in cuda_graph_scope), (
                f"Invalid cuda graph scope: {cuda_graph_scope}. Valid options are: {valid_te_scopes}"
            )

    if cuda_graph_scope is not None:
        recipe.model.cuda_graph_scope = cuda_graph_scope

    return recipe


def _set_recompute_overrides(
    recipe: ConfigContainer,
    cpu_offloading_num_layers: Optional[int] = None,
    recompute_num_layers: Optional[int] = None,
    recompute_modules: Optional[List[str]] = None,
) -> ConfigContainer:
    """Set the recompute and CPU offloading overrides."""
    if cpu_offloading_num_layers is not None:
        recipe.model.cpu_offloading = True
        recipe.model.cpu_offloading_weights = False
        recipe.model.cpu_offloading_num_layers = cpu_offloading_num_layers
    if recompute_num_layers is not None:
        recipe.model.recompute_granularity = "full"
        recipe.model.recompute_method = "block"
        recipe.model.recompute_num_layers = recompute_num_layers
    if recompute_modules is not None:
        recipe.model.recompute_modules = recompute_modules
        recipe.model.recompute_granularity = "selective"

    return recipe


def _set_moe_a2a_overlap_overrides(recipe: ConfigContainer, moe_a2a_overlap: bool = False) -> ConfigContainer:
    """Tune configuration for MoE A2A communication overlap."""
    if moe_a2a_overlap:
        recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
        recipe.comm_overlap.delay_wgrad_compute = True
        recipe.model.moe_shared_expert_overlap = False

    return recipe


def set_workload_base_configs(cfg: ConfigContainer, settings: WorkloadBaseConfig) -> ConfigContainer:
    """Set workload base configs."""
    cfg.model.tensor_model_parallel_size = settings.tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = settings.pipeline_model_parallel_size
    cfg.model.context_parallel_size = settings.context_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = settings.virtual_pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = settings.expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = settings.expert_tensor_parallel_size
    cfg.model.sequence_parallel = settings.sequence_parallel
    cfg.train.global_batch_size = settings.global_batch_size
    cfg.train.micro_batch_size = settings.micro_batch_size

    _set_megatron_fsdp_overrides(cfg, use_megatron_fsdp=settings.use_megatron_fsdp, nccl_ub=settings.nccl_ub)
    _set_cuda_graph_overrides(
        cfg,
        cuda_graph_impl=settings.cuda_graph_impl,
        cuda_graph_scope=settings.cuda_graph_scope,
    )
    _set_moe_a2a_overlap_overrides(cfg, moe_a2a_overlap=settings.moe_a2a_overlap)
    _set_recompute_overrides(
        cfg,
        recompute_modules=settings.recompute_modules,
        cpu_offloading_num_layers=settings.cpu_offloading_num_layers,
        recompute_num_layers=settings.recompute_num_layers,
    )
    _set_common_perf_overrides(cfg)

    return cfg


def set_cli_overrides(recipe: ConfigContainer, cli_overrides: List[str]) -> ConfigContainer:
    """Set Hydra-style CLI overrides."""
    if not cli_overrides:
        return recipe

    # OmegaConf can't serialize problematic callable objects (functions/methods/partials, etc.),
    # so those fields are "excluded_fields" in the temporary OmegaConf conversion and then restored after overrides
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(recipe)
    logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
    merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
    logger.debug("Hydra-style command-line overrides applied successfully.")
    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving "excluded_fields"
    apply_overrides(recipe, final_overrides_as_dict, excluded_fields)

    return recipe


def set_user_overrides(recipe: ConfigContainer, args: argparse.Namespace) -> ConfigContainer:
    """Set the user overrides."""
    _set_megatron_fsdp_overrides(recipe, use_megatron_fsdp=args.use_megatron_fsdp, nccl_ub=args.nccl_ub)
    _set_cuda_graph_overrides(
        recipe,
        cuda_graph_impl=args.cuda_graph_impl,
        cuda_graph_scope=args.cuda_graph_scope,
    )
    _set_recompute_overrides(
        recipe,
        recompute_num_layers=args.recompute_num_layers,
        cpu_offloading_num_layers=args.activation_offload_layers,
        recompute_modules=args.recompute_modules,
    )
    _set_moe_a2a_overlap_overrides(recipe, moe_a2a_overlap=args.moe_a2a_overlap)

    if args.use_tokendrop is True:
        recipe.model = apply_moe_token_drop(recipe.model)
        recipe.model.moe_router_force_load_balancing = False
    if args.use_tokendrop is False:
        recipe.model = apply_moe_token_drop(
            recipe.model, moe_expert_capacity_factor=-1.0, moe_pad_expert_input_to_capacity=False
        )
        recipe.model.moe_router_force_load_balancing = True
    if args.wandb_key is not None:
        recipe.logger.wandb_project = args.wandb_project_name
        recipe.logger.wandb_exp_name = args.wandb_experiment_name
        recipe.logger.wandb_entity = args.wandb_entity_name
        recipe.logger.wandb_save_dir = "/nemo_run/wandb"
    if args.max_steps is not None:
        recipe.train.train_iters = args.max_steps
    if args.tensor_model_parallel_size is not None:
        recipe.model.tensor_model_parallel_size = args.tensor_model_parallel_size
        recipe.model.sequence_parallel = bool(args.tensor_model_parallel_size > 1)
    if args.pipeline_model_parallel_size is not None:
        recipe.model.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    if args.context_parallel_size is not None:
        recipe.model.context_parallel_size = args.context_parallel_size
    if args.virtual_pipeline_model_parallel_size != -1:
        recipe.model.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    if args.expert_model_parallel_size is not None:
        recipe.model.expert_model_parallel_size = args.expert_model_parallel_size
    if args.expert_tensor_parallel_size is not None:
        recipe.model.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    if args.global_batch_size is not None:
        recipe.train.global_batch_size = args.global_batch_size
    if args.micro_batch_size is not None:
        recipe.train.micro_batch_size = args.micro_batch_size
    if args.pretrained_checkpoint is not None:
        recipe.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    if args.tokenizer_type == "NullTokenizer":
        recipe.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=args.vocab_size)
    elif args.tokenizer_type == "HuggingFaceTokenizer":
        if not args.tokenizer_model:
            raise ValueError("--tokenizer-model is required when using HuggingFaceTokenizer")
        tokenizer_model = args.tokenizer_model
        recipe.tokenizer = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=tokenizer_model)
    elif args.tokenizer_type == "SentencePieceTokenizer":
        if not args.tokenizer_model:
            raise ValueError("--tokenizer-model is required for SentencePieceTokenizer")
        recipe.tokenizer = TokenizerConfig(
            tokenizer_type="SentencePieceTokenizer", tokenizer_model=args.tokenizer_model
        )
    # Create dataset configuration based on type
    if args.data == "mock":
        recipe.dataset = create_mock_dataset_config(seq_length=args.seq_length or recipe.model.seq_length)
    elif args.data == "rp2":
        if not args.dataset_paths or not args.index_mapping_dir:
            raise ValueError("--dataset-paths and --index-mapping-dir are required for rp2 dataset")
        recipe.dataset = create_rp2_dataset_config(
            dataset_paths=args.dataset_paths,
            seq_length=recipe.dataset.sequence_length,
            index_mapping_dir=args.index_mapping_dir,
        )
    elif args.data == "squad":
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for squad dataset")
        recipe.dataset = create_squad_dataset_config(
            dataset_root=args.dataset_root, seq_length=args.seq_length or recipe.model.seq_length, packed=False
        )
    elif args.data == "squad_packed":
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for squad_packed dataset")
        recipe.dataset = create_squad_dataset_config(
            dataset_root=args.dataset_root, seq_length=args.seq_length or recipe.model.seq_length, packed=True
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.data}")

    # Reconfigure the DeepSeek-V3 pipeline model parallel layout
    # if the user has provided a custom PP and VP sizes
    model_recipe_name = args.model_recipe_name
    pp_size = args.pipeline_model_parallel_size
    vp_size = args.virtual_pipeline_model_parallel_size
    if model_recipe_name == "deepseek_v3_pretrain_config" and pp_size is not None and vp_size != -1:
        set_deepseek_v3_pipeline_model_parallel_layout(recipe.model, (pp_size, vp_size))

    if args.pytorch_profiler:
        recipe.logger.tensorboard_dir = "/nemo_run/pytorch_profile"

    return recipe


def set_post_overrides(
    recipe: ConfigContainer,
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    num_gpus: int,
    compute_dtype: str,
    task: str,
    user_gbs: Optional[int] = None,
) -> ConfigContainer:
    """Set the post overrides."""
    workload_base_config = get_workload_base_config(model_family_name, model_recipe_name, gpu, compute_dtype, task)

    if compute_dtype == "bf16":
        recipe.optimizer.use_precision_aware_optimizer = True

    tp = recipe.model.tensor_model_parallel_size
    pp = recipe.model.pipeline_model_parallel_size
    cp = recipe.model.context_parallel_size
    vp = recipe.model.virtual_pipeline_model_parallel_size or 1

    dp = int(num_gpus / (tp * pp * cp))
    logger.info(f"DP: {dp}; TP: {tp}; PP: {pp}; CP: {cp}; VP: {vp}")
    if dp > 1 and pp > 1 and vp > 1:
        recipe.optimizer.overlap_param_gather_with_optimizer_step = True
        if hasattr(recipe, "comm_overlap") and isinstance(recipe.comm_overlap, CommOverlapConfig):
            recipe.comm_overlap.overlap_param_gather_with_optimizer_step = True

    default_num_gpus = workload_base_config.num_gpus
    if user_gbs is None:
        if num_gpus != default_num_gpus:
            new_gbs = int(workload_base_config.gbs_scaling_factor * num_gpus)
            recipe.train.global_batch_size = new_gbs
            logger.info(
                f"Scaled global batch size from {workload_base_config.global_batch_size} to {new_gbs} based on {num_gpus} GPUs."
            )

    return recipe
