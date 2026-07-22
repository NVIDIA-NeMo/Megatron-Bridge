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
import os
from typing import List, Optional

from omegaconf import OmegaConf

from megatron.bridge.recipes.deepseek.deepseek_v3 import set_deepseek_v3_pipeline_model_parallel_layout
from megatron.bridge.recipes.kimi.kimi_k2 import _get_kimi_k2_pipeline_layout
from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides
from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.cuda_graph import (
    cuda_graph_module_names,
    is_full_iteration_cuda_graph,
    set_cuda_graph_modules,
    set_full_iteration_cuda_graph,
    validate_cuda_graph_configuration,
)
from utils.datasets import (
    create_c4_dataset_config,
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
    recipe.train.manual_gc = True
    recipe.train.manual_gc_interval = 100

    # Checkpoint save is disabled by default for performance benchmarks
    # Users can enable it via command-line arguments
    recipe.checkpoint.save = None

    recipe.logger.log_interval = 1
    recipe.logger.tensorboard_dir = None

    recipe.ddp.check_for_nan_in_grad = False
    recipe.ddp.check_for_large_grads = False

    recipe.rerun_state_machine.check_for_nan_in_loss = False

    recipe.scheduler.lr_decay_iters = recipe.train.train_iters
    recipe.scheduler.lr_warmup_iters = 10

    if hasattr(recipe.model, "apply_rope_fusion"):
        recipe.model.apply_rope_fusion = True
    if hasattr(recipe.model, "cross_entropy_fusion_impl"):
        recipe.model.cross_entropy_fusion_impl = "te"

    # TODO: This needs to be adjusted when overlapping HybridEP with computation or
    # the number of SMs for HybridEP is reduced.
    if hasattr(recipe.model, "moe_flex_dispatcher_backend") and recipe.model.moe_flex_dispatcher_backend == "hybridep":
        recipe.model.moe_hybridep_num_sms = 32

    return recipe


def _set_megatron_fsdp_overrides(recipe: ConfigContainer, use_megatron_fsdp: bool = False) -> ConfigContainer:
    """Set the Megatron FSDP overrides."""
    if not use_megatron_fsdp:
        return

    recipe.ddp.use_megatron_fsdp = True
    recipe.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    recipe.ddp.keep_fp8_transpose_cache = False
    # average_in_collective is not supported with Megatron FSDP
    recipe.ddp.average_in_collective = False

    recipe.model.init_model_with_meta_device = True
    recipe.model.gradient_accumulation_fusion = True

    if recipe.comm_overlap is not None and isinstance(recipe.comm_overlap, CommOverlapConfig):
        if recipe.comm_overlap.defer_embedding_wgrad_compute:
            logger.warning("Disabling deferring embedding wgrad compute because it cannot work with FSDP together.")
            recipe.comm_overlap.defer_embedding_wgrad_compute = False

    recipe.checkpoint.load = None
    return recipe


def _set_cuda_graph_overrides(
    recipe: ConfigContainer,
    cuda_graph_impl: Optional[str] = None,
    cuda_graph_scope: Optional[str | List[str]] = None,
) -> ConfigContainer:
    """Set the CUDA graph overrides."""
    if isinstance(cuda_graph_scope, str):
        cuda_graph_scope = [cuda_graph_scope]
    if cuda_graph_impl is not None:
        recipe.model.cuda_graph_impl = cuda_graph_impl
        if cuda_graph_impl != "none":
            recipe.rng.te_rng_tracker = recipe.model.use_te_rng_tracker = True

    if recipe.model.cuda_graph_impl == "none":
        recipe.rng.te_rng_tracker = recipe.model.use_te_rng_tracker = False
        # Normalize to MCore's current disabled representation: an empty
        # cuda_graph_modules list and no deprecated cuda_graph_scope value.
        validate_cuda_graph_configuration(recipe.model, config_name="model")
        return recipe

    if cuda_graph_scope is not None:
        if "full_iteration" in cuda_graph_scope:
            if cuda_graph_scope != ["full_iteration"]:
                raise ValueError("full_iteration CUDA graph scope cannot be combined with other scopes.")
            if recipe.model.cuda_graph_impl == "transformer_engine":
                raise ValueError("full_iteration CUDA graph scope cannot use cuda_graph_impl=transformer_engine.")
            set_full_iteration_cuda_graph(recipe.model)
        else:
            set_cuda_graph_modules(recipe.model, cuda_graph_scope)

    # Validate post-override state so we check the effective recipe value,
    # not the raw CLI input (which may be None when the recipe default is used).
    if recipe.model.cuda_graph_impl == "transformer_engine":
        valid_te_scopes = ["attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"]
        effective_scope = cuda_graph_module_names(recipe.model)
        assert all(scope in valid_te_scopes for scope in effective_scope), (
            f"Invalid cuda graph scope: {effective_scope}. Valid options are: {valid_te_scopes}"
        )
    validate_cuda_graph_configuration(recipe.model, config_name="model")

    if is_full_iteration_cuda_graph(recipe.model):
        recipe.rerun_state_machine.check_for_nan_in_loss = False

    return recipe


def _set_recompute_overrides(
    recipe: ConfigContainer,
    cpu_offloading_num_layers: Optional[int] = None,
    recompute_num_layers: Optional[int] = None,
    recompute_modules: Optional[List[str]] = None,
) -> ConfigContainer:
    """Set the recompute and CPU offloading overrides."""
    if cpu_offloading_num_layers is not None and cpu_offloading_num_layers > 0:
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
    # No else: if the caller has no recompute configuration to apply, leave
    # whatever the recipe already has in place. Callers that want to *disable*
    # recompute should set granularity=None / modules=[] at the recipe or via
    # Hydra overrides explicitly.

    return recipe


def _set_moe_a2a_overlap_overrides(recipe: ConfigContainer, moe_a2a_overlap: bool = False) -> ConfigContainer:
    """Tune configuration for MoE A2A communication overlap."""
    if moe_a2a_overlap:
        if recipe.comm_overlap is None:
            tp_comm_overlap = bool(recipe.model.tensor_model_parallel_size > 1)
            recipe.comm_overlap = CommOverlapConfig(tp_comm_overlap=tp_comm_overlap)
        recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
        recipe.comm_overlap.delay_wgrad_compute = True
        recipe.model.moe_shared_expert_overlap = False

    return recipe


def _set_recipe_env(
    recipe: ConfigContainer,
    name: str,
    value: str | int | float | bool,
    protected_env_names: set[str],
) -> None:
    if name not in protected_env_names:
        recipe.env_vars[name] = value


def _remove_recipe_env(recipe: ConfigContainer, name: str, protected_env_names: set[str]) -> None:
    if name not in protected_env_names:
        recipe.env_vars.pop(name, None)


def _apply_flat_cli_environment_compatibility(
    recipe: ConfigContainer,
    args: argparse.Namespace,
    *,
    base_dispatcher_backend: str | None,
    base_moe_a2a_overlap: bool,
    protected_env_names: set[str] | None = None,
) -> ConfigContainer:
    """Apply legacy argparse-to-environment coupling for a flat perf recipe.

    This is an internal ``run_script.py`` compatibility helper, not a general
    recipe environment API. Flat recipes own their default ``env_vars``, but
    the removed launcher plugin also changed a small set of environment values
    when users explicitly overrode TP, PP, CP, EP, MoE A2A overlap, or NCCL UB
    through the legacy argparse interface. Call this after applying both Hydra
    and argparse config overrides so those old commands retain their effective
    process environment.

    With no relevant argparse override, the recipe environment is unchanged.
    Names explicitly overridden through Hydra ``env_vars`` are protected and
    remain final. Existing shell or launcher values retain higher precedence
    later, when the recipe environment is installed with ``os.environ.setdefault``.
    """
    protected = protected_env_names or set()
    model = recipe.model
    tp_size = getattr(model, "tensor_model_parallel_size", 1) or 1
    pp_size = getattr(model, "pipeline_model_parallel_size", 1) or 1
    cp_size = getattr(model, "context_parallel_size", 1) or 1
    ep_size = getattr(model, "expert_model_parallel_size", 1) or 1
    gpu = args.gpu.lower()

    connection_override = any(
        value is not None
        for value in (
            args.tensor_model_parallel_size,
            args.context_parallel_size,
            args.moe_a2a_overlap,
        )
    )
    if connection_override:
        moe_a2a_overlap = args.moe_a2a_overlap if args.moe_a2a_overlap is not None else base_moe_a2a_overlap
        max_connections = 32 if base_dispatcher_backend in {"deepep", "hybridep"} else 8
        if gpu in {"b200", "b300", "gb200", "gb300"}:
            max_connections = 32
        elif (tp_size > 1 or cp_size > 1) and not moe_a2a_overlap:
            max_connections = 1
        _set_recipe_env(recipe, "CUDA_DEVICE_MAX_CONNECTIONS", max_connections, protected)

    if args.expert_model_parallel_size is not None and base_dispatcher_backend == "hybridep":
        if ep_size <= 0:
            raise ValueError("HybridEP expert parallel size must be positive.")
        if gpu in {"h100", "b200", "b300"}:
            domain_size = 8
            ranks_per_domain = min(ep_size, domain_size)
        elif gpu in {"gb200", "gb300", "vr200", "r100"}:
            domain_size = 72
            if ep_size > domain_size:
                raise ValueError("HybridEP expert parallel size must not exceed the 72-rank NVLink domain.")
            ranks_per_domain = ep_size
        else:
            raise ValueError(f"Unsupported GPU type for HybridEP topology: {args.gpu!r}.")
        _set_recipe_env(
            recipe,
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
            ranks_per_domain,
            protected,
        )

    if args.pipeline_model_parallel_size is not None:
        use_large_pp_chunk = pp_size > 1 and (
            (
                args.model_family_name == "llama"
                and args.model_recipe_name in {"llama3_70b", "llama31_405b"}
                and args.task == "pretrain"
            )
            or (args.model_family_name == "llama" and args.task == "sft")
        )
        if use_large_pp_chunk:
            _set_recipe_env(recipe, "NCCL_P2P_NET_CHUNKSIZE", 2097152, protected)
        else:
            _remove_recipe_env(recipe, "NCCL_P2P_NET_CHUNKSIZE", protected)

    if args.nccl_ub is True:
        _set_recipe_env(recipe, "NCCL_NVLS_ENABLE", 1, protected)
        _set_recipe_env(recipe, "NCCL_CTA_POLICY", 1, protected)

    return recipe


def _set_checkpoint_overrides(recipe: ConfigContainer, args: argparse.Namespace) -> ConfigContainer:
    """Set checkpoint save/load configuration."""
    # When save_interval is provided, enable checkpointing
    if args.save_interval is not None:
        recipe.checkpoint.save_interval = args.save_interval
        logger.info(f"Checkpoint save interval set to: {args.save_interval} iterations")

        # Set save directory (use provided or default)
        if args.save_dir is not None:
            recipe.checkpoint.save = args.save_dir
            logger.info(f"Checkpoint save directory set to: {args.save_dir}")
        else:
            recipe.checkpoint.save = "/nemo_run/checkpoints"
            logger.info("Checkpoint save directory defaulting to: /nemo_run/checkpoints")

    # If only save_dir is provided without save_interval, still enable checkpointing
    elif args.save_dir is not None:
        recipe.checkpoint.save = args.save_dir
        logger.info(f"Checkpoint save directory set to: {args.save_dir}")
        # Default save_interval to train_iters
        recipe.checkpoint.save_interval = recipe.train.train_iters
        logger.info(f"Checkpoint save interval defaulting to train_iters: {recipe.train.train_iters}")

    if args.load_dir is not None:
        recipe.checkpoint.load = args.load_dir
        logger.info(f"Checkpoint load directory set to: {args.load_dir}")

    if args.most_recent_k is not None:
        recipe.checkpoint.most_recent_k = args.most_recent_k
        logger.info(f"Keeping {args.most_recent_k} most recent checkpoints")

    if args.save_config_filepath is not None:
        recipe.logger.save_config_filepath = args.save_config_filepath
        # maybe_log_and_save_config calls cfg.to_yaml() during training startup; that uses
        # plain open(..., "w") and fails if the parent dir doesn't exist. Ensure it does.
        os.makedirs(os.path.dirname(os.path.abspath(args.save_config_filepath)), exist_ok=True)

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

    _set_megatron_fsdp_overrides(cfg, use_megatron_fsdp=settings.use_megatron_fsdp)
    _set_nccl_ub_overrides(cfg, nccl_ub=settings.nccl_ub)
    _set_cuda_graph_overrides(
        cfg,
        cuda_graph_impl=settings.cuda_graph_impl,
        cuda_graph_scope=settings.cuda_graph_scope,
    )
    _set_moe_a2a_overlap_overrides(cfg, moe_a2a_overlap=settings.moe_a2a_overlap)
    if settings.cutedsl_fused_grouped_mlp:
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.model.moe_mlp_glu_interleave_size = 32
        if settings.moe_a2a_overlap:
            cfg.model.high_priority_a2a_comm_stream = True
            cfg.model.moe_hybridep_num_sms_preprocessing = 32
    if settings.fp8_dot_product_attention is not None:
        cfg.mixed_precision.fp8_dot_product_attention = settings.fp8_dot_product_attention
    _set_recompute_overrides(
        cfg,
        recompute_modules=settings.recompute_modules,
        cpu_offloading_num_layers=settings.cpu_offloading_num_layers,
        recompute_num_layers=settings.recompute_num_layers,
    )
    if settings.te_precision_config_file is not None:
        from megatron.core.quantization.utils import load_quantization_recipe

        cfg.model.quant_recipe = load_quantization_recipe(settings.te_precision_config_file)
    _set_common_perf_overrides(cfg)

    if settings.fine_grained_activation_offloading is not None:
        cfg.model.fine_grained_activation_offloading = settings.fine_grained_activation_offloading
    if settings.offload_modules is not None:
        cfg.model.offload_modules = settings.offload_modules

    if settings.outer_dp_sharding_strategy is not None:
        cfg.ddp.outer_dp_sharding_strategy = settings.outer_dp_sharding_strategy
    if settings.num_distributed_optimizer_instances is not None:
        cfg.ddp.num_distributed_optimizer_instances = settings.num_distributed_optimizer_instances

    if settings.moe_flex_dispatcher_backend is not None:
        apply_flex_dispatcher_backend(cfg.model, settings.moe_flex_dispatcher_backend)
    elif hasattr(cfg.model, "moe_token_dispatcher_type"):
        cfg.model.moe_token_dispatcher_type = "alltoall"

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


def _set_nccl_ub_overrides(recipe: ConfigContainer, nccl_ub: bool = False) -> ConfigContainer:
    """Set the NCCL UB overrides."""
    if nccl_ub:
        recipe.ddp.nccl_ub = True
        # The current version of NCCL does not support the AVG operation for reductions with symmetric kernels.
        # To enable symmetric kernels, average_in_collective must be disabled.
        recipe.ddp.average_in_collective = False

    if recipe.ddp.use_megatron_fsdp and recipe.ddp.nccl_ub:
        recipe.ddp.fsdp_manual_registration = True

    return recipe


def set_user_overrides(recipe: ConfigContainer, args: argparse.Namespace) -> ConfigContainer:
    """Set the user overrides."""
    _set_megatron_fsdp_overrides(recipe, use_megatron_fsdp=args.use_megatron_fsdp)
    _set_nccl_ub_overrides(recipe, nccl_ub=args.nccl_ub)
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

    recipe.logger.save_config_filepath = args.save_config_filepath or "/nemo_run/configs/ConfigContainer.yaml"
    # maybe_log_and_save_config calls cfg.to_yaml() during training startup; that uses
    # plain open(..., "w") and fails if the parent dir doesn't exist. Ensure it does.
    os.makedirs(os.path.dirname(os.path.abspath(recipe.logger.save_config_filepath)), exist_ok=True)

    if args.max_steps is not None:
        recipe.train.train_iters = args.max_steps
    if args.tensor_model_parallel_size is not None:
        recipe.model.tensor_model_parallel_size = args.tensor_model_parallel_size
        recipe.model.sequence_parallel = bool(args.tensor_model_parallel_size > 1)
    if args.pipeline_model_parallel_size is not None:
        recipe.model.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    if args.context_parallel_size is not None:
        recipe.model.context_parallel_size = args.context_parallel_size
    # VP special case: -1 means "not specified, use default config", but None is a valid user override
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

    # Handle checkpoint configuration
    _set_checkpoint_overrides(recipe, args)

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

    if args.seq_length is not None:
        recipe.model.seq_length = args.seq_length

    # Create dataset configuration based on type
    if args.data == "mock":
        if args.domain == "llm":
            # Override the dataset configuration for LLM models.
            # For vlm models, use the default dataset configuration in model recipe,
            # becuase preprocess of dataset is different for each vlm model.
            recipe.dataset = create_mock_dataset_config(
                seq_length=recipe.model.seq_length,
                num_workers=recipe.dataset.num_workers,
                pin_memory=recipe.dataset.pin_memory,
                persistent_workers=recipe.dataset.persistent_workers,
            )
    elif args.data == "rp2":
        if not args.dataset_paths or not args.index_mapping_dir:
            raise ValueError("--dataset-paths and --index-mapping-dir are required for rp2 dataset")
        recipe.dataset = create_rp2_dataset_config(
            dataset_paths=args.dataset_paths,
            seq_length=args.seq_length if args.seq_length is not None else recipe.dataset.seq_length,
            index_mapping_dir=args.index_mapping_dir,
            num_workers=recipe.dataset.num_workers,
            pin_memory=recipe.dataset.pin_memory,
            persistent_workers=recipe.dataset.persistent_workers,
        )
    elif args.data == "c4":
        if not args.c4_root:
            raise ValueError("--c4_root is required for c4 dataset")
        recipe.dataset = create_c4_dataset_config(
            seq_length=args.seq_length if args.seq_length is not None else recipe.dataset.seq_length,
            c4_root=args.c4_root,
            train_shards=tuple(args.c4_train_shards),
            index_mapping_dir=args.index_mapping_dir,
            num_workers=recipe.dataset.num_workers,
            pin_memory=recipe.dataset.pin_memory,
            persistent_workers=recipe.dataset.persistent_workers,
        )
    elif args.data == "squad":
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for squad dataset")
        cp_size = getattr(recipe.model, "context_parallel_size", 1) or 1
        pad_seq_to_mult = cp_size * 2 if cp_size > 1 else 1
        recipe.dataset = create_squad_dataset_config(
            dataset_root=args.dataset_root,
            seq_length=recipe.model.seq_length,
            packed=False,
            pad_seq_to_mult=pad_seq_to_mult,
            num_workers=recipe.dataset.num_workers,
            pin_memory=recipe.dataset.pin_memory,
            persistent_workers=recipe.dataset.persistent_workers,
        )
    elif args.data == "squad_packed":
        if not args.dataset_root:
            raise ValueError("--dataset-root is required for squad_packed dataset")
        cp_size = getattr(recipe.model, "context_parallel_size", 1) or 1
        pad_seq_to_mult = cp_size * 2 if cp_size > 1 else 1
        recipe.dataset = create_squad_dataset_config(
            dataset_root=args.dataset_root,
            seq_length=recipe.model.seq_length,
            packed=True,
            pad_seq_to_mult=pad_seq_to_mult,
            num_workers=recipe.dataset.num_workers,
            pin_memory=recipe.dataset.pin_memory,
            persistent_workers=recipe.dataset.persistent_workers,
        )
        if recipe.model.cuda_graph_impl != "none":
            recipe.dataset.offline_packing_specs.pad_cu_seqlens = True
        recipe.dataset.dataset_kwargs = {"pad_to_max_length": True}
    else:
        raise ValueError(f"Unknown dataset type: {args.data}")
    if args.hidden_size is not None:
        recipe.model.hidden_size = args.hidden_size
    if args.num_layers is not None or args.first_k_dense_replace is not None:
        if args.first_k_dense_replace is not None:
            num_dense_layers = args.first_k_dense_replace
        else:
            num_dense_layers = recipe.model.moe_layer_freq.count(0)
        if args.num_layers is not None:
            recipe.model.num_layers = args.num_layers
        recipe.model.moe_layer_freq = [0] * num_dense_layers + [1] * (recipe.model.num_layers - num_dense_layers)
    if args.num_moe_experts is not None:
        recipe.model.num_moe_experts = args.num_moe_experts
    if args.pipeline_model_parallel_layout is not None:
        recipe.model.pipeline_model_parallel_layout = args.pipeline_model_parallel_layout

    # Reconfigure the DeepSeek-V3 pipeline model parallel layout
    # if the user has provided a custom PP and VP sizes
    model_recipe_name = args.model_recipe_name
    pp_size = args.pipeline_model_parallel_size
    vp_size = args.virtual_pipeline_model_parallel_size
    pipeline_model_parallel_layout = args.pipeline_model_parallel_layout
    if model_recipe_name == "deepseek_v3" and (
        pp_size is not None or vp_size != -1 or pipeline_model_parallel_layout is not None
    ):
        set_deepseek_v3_pipeline_model_parallel_layout(recipe.model, layout=pipeline_model_parallel_layout)
    if model_recipe_name == "kimi_k2":
        if pp_size is not None or vp_size != -1:
            if not isinstance(recipe.model.pipeline_model_parallel_layout, str):
                try:
                    layout = _get_kimi_k2_pipeline_layout(
                        recipe.model.pipeline_model_parallel_size, recipe.model.virtual_pipeline_model_parallel_size
                    )
                    recipe.model.pipeline_model_parallel_layout = layout
                except ValueError:
                    logger.warning(
                        f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout for Kimi-K2. Using default layout."
                    )
                    recipe.model.pipeline_model_parallel_layout = None
        if pipeline_model_parallel_layout is not None:
            recipe.model.pipeline_model_parallel_layout = pipeline_model_parallel_layout

    if args.pytorch_profiler:
        recipe.logger.tensorboard_dir = "/nemo_run/pytorch_profile"

    if args.moe_flex_dispatcher_backend is None:
        recipe.model.moe_flex_dispatcher_backend = None
        recipe.model.moe_token_dispatcher_type = "alltoall"
    elif args.moe_flex_dispatcher_backend != -1:
        apply_flex_dispatcher_backend(recipe.model, args.moe_flex_dispatcher_backend)

    pp_size = getattr(recipe.model, "pipeline_model_parallel_size", 1) or 1
    if args.task == "peft" and pp_size > 1 and not recipe.ddp.use_megatron_fsdp:
        recipe.dist.use_tp_pp_dp_mapping = True

    if args.deterministic:
        apply_determinism_overrides(recipe)

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
    config_variant: str | None = None,
) -> ConfigContainer:
    """Set the post overrides."""
    workload_base_config = get_workload_base_config(
        model_family_name,
        model_recipe_name,
        gpu,
        compute_dtype,
        task,
        config_variant,
    )

    if compute_dtype == "bf16" and recipe.optimizer.optimizer == "adam":
        recipe.optimizer.use_precision_aware_optimizer = True

    tp = recipe.model.tensor_model_parallel_size
    pp = recipe.model.pipeline_model_parallel_size
    cp = recipe.model.context_parallel_size
    vp = recipe.model.virtual_pipeline_model_parallel_size or 1

    dp = int(num_gpus / (tp * pp * cp))
    logger.info(f"DP: {dp}; TP: {tp}; PP: {pp}; CP: {cp}; VP: {vp}")
    # NOTE: overlap_param_gather_with_optimizer_step causes NaN grad norm for fp8_mx. Disabling it until the issue is resolved.
    if dp > 1 and pp > 1 and vp > 1 and compute_dtype not in ("fp8_mx", "nvfp4"):
        # Do not enable overlap_param_gather_with_optimizer_step for muon optimizer.
        if recipe.optimizer.optimizer != "dist_muon":
            recipe.optimizer.overlap_param_gather_with_optimizer_step = True
            if hasattr(recipe, "comm_overlap") and isinstance(recipe.comm_overlap, CommOverlapConfig):
                recipe.comm_overlap.overlap_param_gather_with_optimizer_step = True

    default_num_gpus = workload_base_config.num_gpus
    if user_gbs is None:
        # Only auto-rescale if the recipe's current GBS still equals the
        # workload default — i.e., no one (Hydra or earlier step) has already
        # expressed an intent. Otherwise Hydra-set GBS gets silently stomped.
        if recipe.train.global_batch_size == workload_base_config.global_batch_size and num_gpus != default_num_gpus:
            new_gbs = int(workload_base_config.gbs_scaling_factor * num_gpus)
            recipe.train.global_batch_size = new_gbs
            logger.info(
                f"Scaled global batch size from {workload_base_config.global_batch_size} to {new_gbs} based on {num_gpus} GPUs."
            )

    return recipe
