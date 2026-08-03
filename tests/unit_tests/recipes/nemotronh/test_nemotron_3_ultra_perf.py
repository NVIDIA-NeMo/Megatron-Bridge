import pytest
import torch

import megatron.bridge.training.config as training_config
from megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh import (
    nemotron_3_ultra_pretrain_64gpu_gb200_fp8mx_config,
    nemotron_3_ultra_pretrain_128gpu_gb200_fp8mx_tp2_config,
    nemotron_3_ultra_pretrain_128gpu_gb200_fp8mx_tp2_ub_config,
    nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config,
    nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp2_config,
    nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp2_ub_config,
    nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp4_config,
)
from megatron.bridge.perf_recipes.nemotronh.h100.nemotronh import (
    nemotron_3_ultra_pretrain_128gpu_h100_bf16_fsdp_tp2_config,
    nemotron_3_ultra_pretrain_256gpu_h100_bf16_fsdp_config,
)
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("recipe_factory", "num_optimizer_instances", "outer_dp_sharding_strategy"),
    [
        (nemotron_3_ultra_pretrain_64gpu_gb200_fp8mx_config, 1, "no_shard"),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config, 1, "no_shard"),
    ],
)
def test_gb200_ultra_recipes_embed_performance_defaults(
    recipe_factory,
    num_optimizer_instances: int,
    outer_dp_sharding_strategy: str,
) -> None:
    cfg = recipe_factory()

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.sequence_parallel is True
    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.train.global_batch_size == 256
    assert cfg.train.micro_batch_size == 1

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_grouped_gemm is True
    assert cfg.model.moe_permute_fusion is True
    assert cfg.model.moe_router_fusion is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.use_fused_weighted_squared_relu is True
    assert cfg.model.moe_router_padding_for_quantization is True
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["fused_group_mlp"]
    assert cfg.model.min_offloaded_tensor_size == 250_000_000
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe_act"]

    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100
    assert cfg.dist.high_priority_stream_groups == ["ep"]
    assert cfg.dist.distributed_timeout_minutes == 30
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.ddp.num_buckets == 48
    assert cfg.logger.log_throughput is True

    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.average_in_collective is False
    assert cfg.ddp.num_distributed_optimizer_instances == num_optimizer_instances
    assert cfg.ddp.outer_dp_sharding_strategy == outer_dp_sharding_strategy
    assert cfg.ddp.megatron_fsdp_grad_comm_dtype == torch.bfloat16
    assert cfg.ddp.megatron_fsdp_main_params_dtype == torch.float32
    assert cfg.ddp.megatron_fsdp_main_grads_dtype == torch.bfloat16
    assert cfg.ddp.megatron_fsdp_use_decoupled_grad is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.optimizer.overlap_param_gather is True
    assert cfg.dist.use_megatron_fsdp is True
    assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag is False
    assert cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag is False
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.model.gradient_accumulation_fusion is False
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"
    assert cfg.checkpoint.async_save is False

    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 64
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "recipe_factory",
        "tensor_parallel_size",
        "virtual_pipeline_parallel_size",
        "global_batch_size",
        "num_optimizer_instances",
        "outer_dp_sharding_strategy",
        "nccl_user_buffers",
    ),
    [
        (nemotron_3_ultra_pretrain_128gpu_gb200_fp8mx_tp2_config, 2, None, 256, 1, "no_shard", False),
        (nemotron_3_ultra_pretrain_128gpu_gb200_fp8mx_tp2_ub_config, 2, None, 256, 1, "no_shard", True),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp2_config, 2, None, 512, 2, "optim", False),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp2_ub_config, 2, None, 512, 2, "optim", True),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp4_config, 4, None, 512, 1, "no_shard", False),
    ],
)
def test_gb200_ultra_no_offload_candidates(
    recipe_factory,
    tensor_parallel_size: int,
    virtual_pipeline_parallel_size: int | None,
    global_batch_size: int,
    num_optimizer_instances: int,
    outer_dp_sharding_strategy: str,
    nccl_user_buffers: bool,
) -> None:
    cfg = recipe_factory()

    assert cfg.model.tensor_model_parallel_size == tensor_parallel_size
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size == virtual_pipeline_parallel_size
    assert cfg.model.sequence_parallel is True
    assert cfg.train.global_batch_size == global_batch_size
    assert cfg.train.micro_batch_size == 1
    assert cfg.logger.log_throughput is True
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules == []
    assert cfg.ddp.num_distributed_optimizer_instances == num_optimizer_instances
    assert cfg.ddp.outer_dp_sharding_strategy == outer_dp_sharding_strategy
    assert cfg.ddp.megatron_fsdp_enable_fine_grained_param_gather is True
    assert cfg.ddp.nccl_ub is nccl_user_buffers
    assert cfg.ddp.megatron_fsdp_max_pool_double_buffer is nccl_user_buffers
    assert cfg.ddp.fsdp_manual_registration is nccl_user_buffers
    assert "NVTE_CPU_OFFLOAD_V1" not in cfg.env_vars
    assert ("PYTORCH_CUDA_ALLOC_CONF" not in cfg.env_vars) is nccl_user_buffers


@pytest.mark.unit
def test_gb200_ultra_recipe_environments_are_not_shared() -> None:
    cfg = nemotron_3_ultra_pretrain_64gpu_gb200_fp8mx_config()
    cfg.env_vars["sentinel"] = 1

    fresh_cfg = nemotron_3_ultra_pretrain_64gpu_gb200_fp8mx_config()
    assert "sentinel" not in fresh_cfg.env_vars


@pytest.mark.unit
@pytest.mark.parametrize(
    ("recipe_factory", "tensor_parallel_size", "virtual_pipeline_parallel_size", "global_batch_size"),
    [
        (nemotron_3_ultra_pretrain_128gpu_h100_bf16_fsdp_tp2_config, 2, None, 256),
        (nemotron_3_ultra_pretrain_256gpu_h100_bf16_fsdp_config, 4, None, 512),
    ],
)
def test_h100_ultra_fsdp_recipes_match_the_gb200_workloads(
    recipe_factory,
    tensor_parallel_size: int,
    virtual_pipeline_parallel_size: int | None,
    global_batch_size: int,
) -> None:
    cfg = recipe_factory()

    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp8 is None
    assert cfg.model.tensor_model_parallel_size == tensor_parallel_size
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size == virtual_pipeline_parallel_size
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.sequence_parallel is True
    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.train.global_batch_size == global_batch_size
    assert cfg.train.micro_batch_size == 1
    assert cfg.logger.log_throughput is True

    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.moe_flex_dispatcher_backend is None
    assert cfg.model.moe_router_padding_for_quantization is False
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.offload_modules == []
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe", "layernorm", "core_attn"]

    assert cfg.dist.use_megatron_fsdp is True
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.num_distributed_optimizer_instances == 1
    assert cfg.ddp.outer_dp_sharding_strategy == "no_shard"
    assert cfg.ddp.megatron_fsdp_enable_fine_grained_param_gather is True
    assert cfg.ddp.megatron_fsdp_use_decoupled_grad is True
    assert cfg.ddp.megatron_fsdp_grad_comm_dtype == torch.float32
    assert cfg.ddp.megatron_fsdp_main_grads_dtype == torch.float32
    assert cfg.model.gradient_accumulation_fusion is True
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.check_for_large_grads is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.optimizer.overlap_param_gather is True
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"

    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN" not in cfg.env_vars
    assert "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API" not in cfg.env_vars
    assert "NVLINK_DOMAIN_SIZE" not in cfg.env_vars
    assert "USE_MNNVL" not in cfg.env_vars
    assert "NVTE_CPU_OFFLOAD_V1" not in cfg.env_vars
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars


@pytest.mark.unit
@pytest.mark.parametrize(
    ("recipe_factory", "world_size"),
    [
        (nemotron_3_ultra_pretrain_64gpu_gb200_fp8mx_config, 64),
        (nemotron_3_ultra_pretrain_128gpu_gb200_fp8mx_tp2_config, 128),
        (nemotron_3_ultra_pretrain_128gpu_gb200_fp8mx_tp2_ub_config, 128),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config, 256),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp2_config, 256),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp2_ub_config, 256),
        (nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_tp4_config, 256),
        (nemotron_3_ultra_pretrain_128gpu_h100_bf16_fsdp_tp2_config, 128),
        (nemotron_3_ultra_pretrain_256gpu_h100_bf16_fsdp_config, 256),
    ],
)
def test_ultra_fsdp_recipes_validate_for_declared_world_size(
    monkeypatch: pytest.MonkeyPatch,
    recipe_factory,
    world_size: int,
) -> None:
    monkeypatch.setattr(training_config, "get_world_size_safe", lambda: world_size)
    monkeypatch.setattr(training_config, "validate_flex_dispatcher_backend", lambda _model: None)
    cfg = recipe_factory()

    training_config.runtime_config_update(cfg)

    assert cfg.model.pipeline_model_parallel_size == 1
    assert world_size == cfg.model.tensor_model_parallel_size * (world_size // cfg.model.tensor_model_parallel_size)
    expert_data_parallel_size = world_size // (
        cfg.model.pipeline_model_parallel_size
        * cfg.model.expert_tensor_parallel_size
        * cfg.model.expert_model_parallel_size
    )
    assert expert_data_parallel_size % cfg.ddp.num_distributed_optimizer_instances == 0
