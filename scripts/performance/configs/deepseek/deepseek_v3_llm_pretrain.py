import logging

from megatron.bridge.recipes.deepseek.deepseek_v3 import pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

from scripts.performance.utils.helpers import (get_precision_config, moe_a2a_1f1b_overrides, set_basic_perf_overrides)

logger = logging.getLogger(__name__)


def deepseek_v3_gb200_bf16_config(
    fp8_recipe = None,
    use_tokendrop: bool = True,
    enable_deepep: bool = False,
    a2a_1f1b: bool = False,
) -> ConfigContainer:
    """GB200, 4xGPU, BF16 baseline config."""
    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    cfg = pretrain_config(
        mock=True, 
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=4,
        virtual_pipeline_parallelism=4,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.overlap_grad_reduce = True

    if enable_deepep:
        cfg.model.moe_router_force_load_balancing = True
    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)

    if a2a_1f1b:
        cfg = moe_a2a_1f1b_overrides(cfg, perf_overrides={"a2a_1f1b": a2a_1f1b})

    if use_tokendrop:
        cfg.model.recompute_modules = ["mla_up_proj"]
    else:
        cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    return cfg


def deepseek_v3_gb200_fp8_config(
    fp8_recipe: str = "cs",
    use_tokendrop: bool = True,
    enable_deepep: bool = False,
    a2a_1f1b: bool = False,
) -> ConfigContainer:
    """GB200, 4xGPU, FP8 baseline config."""
    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    cfg = pretrain_config(
        mock=True, 
        precision_config=get_precision_config("fp8", fp8_recipe)
        pipeline_parallelism=4,
        virtual_pipeline_parallelism=4,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.overlap_grad_reduce = True

    if enable_deepep:
        cfg.model.moe_router_force_load_balancing = True
    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)

    if a2a_1f1b:
        cfg = moe_a2a_1f1b_overrides(cfg, perf_overrides={"a2a_1f1b": a2a_1f1b})

    if use_tokendrop:
        cfg.model.recompute_modules = ["mla_up_proj"]
    else:
        cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    return cfg

def deepseek_v3_b200_bf16_config(
    fp8_recipe = None,
    use_tokendrop: bool = True,
    enable_deepep: bool = False,
    a2a_1f1b: bool = False,
) -> ConfigContainer:
    """GB200, 4xGPU, BF16 baseline config."""
    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    cfg = pretrain_config(
        mock=True, 
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=16,
        virtual_pipeline_parallelism=1,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.overlap_grad_reduce = True

    if enable_deepep:
        cfg.model.moe_router_force_load_balancing = True
    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)

    if a2a_1f1b:
        cfg = moe_a2a_1f1b_overrides(cfg, perf_overrides={"a2a_1f1b": a2a_1f1b})

    cfg.model.recompute_modules = ["mla_up_proj"]

    return cfg


def deepseek_v3_b200_fp8_config(
    fp8_recipe: str = "cs",
    use_tokendrop: bool = False,
    enable_deepep: bool = True,
    a2a_1f1b: bool = True,
) -> ConfigContainer:
    """GB200, 4xGPU, FP8 baseline config."""
    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    cfg = pretrain_config(
        mock=True, 
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=16,
        virtual_pipeline_parallelism=1,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.comm_overlap.overlap_grad_reduce = True

    if enable_deepep:
        cfg.model.moe_router_force_load_balancing = True
    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)

    if a2a_1f1b:
        cfg = moe_a2a_1f1b_overrides(cfg, perf_overrides={"a2a_1f1b": a2a_1f1b})

    cfg.model.recompute_modules = ["mla_up_proj"]

    return cfg

def deepseek_v3_h100_bf16_config(
    fp8_recipe = None,
    use_tokendrop: bool = False,
    enable_deepep: bool = True,
    a2a_1f1b: bool = True,
) -> ConfigContainer:
    """GB200, 4xGPU, BF16 baseline config."""
    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    cfg = pretrain_config(
        mock=True, 
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=8,
        virtual_pipeline_parallelism=4,
        enable_deepep=enable_deepep,
        layout="Et|(tt|)*30mL",
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    if enable_deepep:
        cfg.model.moe_router_force_load_balancing = True
    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)

    if a2a_1f1b:
        cfg = moe_a2a_1f1b_overrides(cfg, perf_overrides={"a2a_1f1b": a2a_1f1b})

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg


def deepseek_v3_h100_fp8_config(
    fp8_recipe: str = "cs",
    use_tokendrop: bool = True,
    enable_deepep: bool = False,
    a2a_1f1b: bool = False,
) -> ConfigContainer:
    """GB200, 4xGPU, FP8 baseline config."""
    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    cfg = pretrain_config(
        mock=True, 
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=16,
        virtual_pipeline_parallelism=1,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    

    if enable_deepep:
        cfg.model.moe_router_force_load_balancing = True
    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)

    if a2a_1f1b:
        cfg = moe_a2a_1f1b_overrides(cfg, perf_overrides={"a2a_1f1b": a2a_1f1b})

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
