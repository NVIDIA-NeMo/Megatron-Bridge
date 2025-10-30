import logging

from utils.helpers import (
    get_precision_config,
    get_user_parallelism_and_batch_size_configs,
    moe_a2a_1f1b_overrides,
    set_basic_perf_overrides,
)

from megatron.bridge.recipes.deepseek.deepseek_v3 import deepseek_v3_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop


logger = logging.getLogger(__name__)


def deepseek_v3_gb200_256gpus_bf16_config(**kwargs) -> ConfigContainer:
    """GB200, 256xGPU, BF16 baseline config."""
    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    enable_deepep = False if kwargs.get("enable_deepep") is None else kwargs.get("enable_deepep")
    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")

    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=pp if pp is not None else 4,
        virtual_pipeline_parallelism=vp if vp is not None else 4,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 4 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 4 if vp is None else vp
    cfg.model.expert_model_parallel_size = 64 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)
    else:
        cfg.model.moe_router_force_load_balancing = True

    if A2A_1F1B:
        moe_a2a_1f1b_overrides(cfg)

    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_gb200_256gpus_fp8_config(**kwargs) -> ConfigContainer:
    """B200, 256xGPU, FP8 baseline config."""
    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    enable_deepep = False if kwargs.get("enable_deepep") is None else kwargs.get("enable_deepep")
    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")

    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=pp if pp is not None else 4,
        virtual_pipeline_parallelism=vp if vp is not None else 4,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 4 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 4 if vp is None else vp
    cfg.model.expert_model_parallel_size = 64 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)
    else:
        cfg.model.moe_router_force_load_balancing = True

    if A2A_1F1B:
        moe_a2a_1f1b_overrides(cfg)

    cfg.dataset.num_workers = 0
    cfg.dataset.pin_memory = False

    return cfg


def deepseek_v3_b200_256gpus_bf16_config(**kwargs) -> ConfigContainer:
    """B200, 256xGPU, BF16 baseline config."""
    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    enable_deepep = False if kwargs.get("enable_deepep") is None else kwargs.get("enable_deepep")
    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")

    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=pp if pp is not None else 16,
        virtual_pipeline_parallelism=vp if vp is not None else 1,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 16 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 8 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    if enable_deepep:
        cfg.model.moe_router_force_load_balancing = True
    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)
    else:
        cfg.model.moe_router_force_load_balancing = True

    if A2A_1F1B:
        moe_a2a_1f1b_overrides(cfg)

    return cfg


def deepseek_v3_b200_256gpus_fp8_config(**kwargs) -> ConfigContainer:
    """B200, 256xGPU, FP8 baseline config."""
    use_tokendrop = True if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    enable_deepep = False if kwargs.get("enable_deepep") is None else kwargs.get("enable_deepep")
    A2A_1F1B = False if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")

    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=pp if pp is not None else 16,
        virtual_pipeline_parallelism=vp if vp is not None else 1,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cfg.model.tensor_model_parallel_size = 1 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 16 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = None if vp is None else vp
    cfg.model.expert_model_parallel_size = 8 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 2048 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.comm_overlap.overlap_grad_reduce = True

    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)
    else:
        cfg.model.moe_router_force_load_balancing = True

    if A2A_1F1B:
        moe_a2a_1f1b_overrides(cfg)

    return cfg


def deepseek_v3_h100_1024gpus_bf16_config(**kwargs) -> ConfigContainer:
    """H100, 1024xGPU, BF16 baseline config."""
    use_tokendrop = False if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    enable_deepep = True if kwargs.get("enable_deepep") is None else kwargs.get("enable_deepep")
    A2A_1F1B = True if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")

    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("bf16"),
        pipeline_parallelism=pp if pp is not None else 8,
        virtual_pipeline_parallelism=vp if vp is not None else 4,
        enable_deepep=enable_deepep,
        layout="Et|(tt|)*30mL",
    )

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cfg.model.tensor_model_parallel_size = 2 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 4 if vp is None else vp
    cfg.model.expert_model_parallel_size = 64 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 8192 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)
    else:
        cfg.model.moe_router_force_load_balancing = True

    if A2A_1F1B:
        moe_a2a_1f1b_overrides(cfg)

    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg


def deepseek_v3_h100_1024gpus_fp8_config(**kwargs) -> ConfigContainer:
    """H100, 1024xGPU, FP8 baseline config."""
    use_tokendrop = False if kwargs.get("use_tokendrop") is None else kwargs.get("use_tokendrop")
    enable_deepep = True if kwargs.get("enable_deepep") is None else kwargs.get("enable_deepep")
    A2A_1F1B = True if kwargs.get("moe_a2a") is None else kwargs.get("moe_a2a")

    if use_tokendrop and enable_deepep:
        enable_deepep = False
        logger.info("Using token drop, disabling DeepEP")

    tp, pp, cp, vp, ep, etp, mbs, gbs = get_user_parallelism_and_batch_size_configs(kwargs)
    fp8_recipe = kwargs.get("fp8_recipe", "cs")

    cfg = pretrain_config(
        mock=True,
        precision_config=get_precision_config("fp8", fp8_recipe),
        pipeline_parallelism=pp if pp is not None else 16,
        virtual_pipeline_parallelism=vp if vp is not None else 1,
        enable_deepep=enable_deepep,
        layout=None,
    )

    set_basic_perf_overrides(cfg, max_steps=kwargs.get("max_steps"))

    cfg.model.tensor_model_parallel_size = 2 if tp is None else tp
    cfg.model.pipeline_model_parallel_size = 8 if pp is None else pp
    cfg.model.context_parallel_size = 1 if cp is None else cp
    cfg.model.virtual_pipeline_model_parallel_size = 4 if vp is None else vp
    cfg.model.expert_model_parallel_size = 64 if ep is None else ep
    cfg.model.expert_tensor_parallel_size = 1 if etp is None else etp
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 8192 if gbs is None else gbs
    cfg.train.micro_batch_size = 1 if mbs is None else mbs
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.model.recompute_modules = ["mla_up_proj", "mlp"]

    if use_tokendrop:
        cfg.model = apply_moe_token_drop(cfg.model)
    else:
        cfg.model.moe_router_force_load_balancing = True

    if A2A_1F1B:
        moe_a2a_1f1b_overrides(cfg)

    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
