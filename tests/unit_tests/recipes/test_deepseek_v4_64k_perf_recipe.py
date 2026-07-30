import json

import pytest
import torch

from megatron.bridge.perf_recipes.deepseek import (
    deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_config,
    deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config,
)
from megatron.bridge.training.config import MockVarlenDatasetConfig
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_deepseek_v4_pro_64k_thd_recipe() -> None:
    cfg = deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_config()

    assert cfg.model.num_layers == 15
    assert cfg.model.num_moe_experts == 384
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.context_parallel_size == 16
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False

    assert cfg.model.seq_length == 65_536
    assert cfg.model.max_position_embeddings == 65_536
    assert cfg.model.vocab_size == 129_280
    assert cfg.model.actual_vocab_size == 129_280
    assert cfg.model.q_lora_rank == 1536
    assert cfg.model.o_groups == 16
    assert cfg.model.o_lora_rank == 1024
    assert cfg.model.csa_compress_ratios == [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
    assert cfg.model.sequence_packing_scheduler == "dp_balanced"
    assert cfg.model.max_seqlen_per_dp_cp_rank == 4096
    assert cfg.model.pad_packed_seq_alignment == "max"
    assert cfg.model.thd_max_packed_sequences == 8
    assert cfg.model.cp_partition_mode == "contiguous"
    assert cfg.model.calculate_per_token_loss is True

    assert isinstance(cfg.dataset, MockVarlenDatasetConfig)
    assert cfg.dataset.seq_length == 65_536
    assert cfg.dataset.context_parallel_size == 16
    assert cfg.dataset.data_parallel_size == 4
    varlen_config = json.loads(cfg.dataset.varlen_mock_dataset_config_json)
    assert varlen_config["format"] == "thd"
    assert varlen_config["min_seq_len"] == 65_534
    assert varlen_config["max_seq_len"] == 65_534
    assert varlen_config["mean_seq_len"] == 65_534

    assert cfg.train.global_batch_size == 256
    assert cfg.train.train_iters is None
    assert cfg.train.train_samples == 585_937_500
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.exit_interval == 30
    assert cfg.train.manual_gc_interval == 10
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.cuda_graph_warmup_steps == 1
    assert cfg.model.fine_grained_activation_offloading is False
    assert cfg.model.moe_paged_stash is False

    assert cfg.optimizer.optimizer == "muon"
    assert cfg.optimizer.use_distributed_optimizer is False
    assert cfg.optimizer.use_layer_wise_distributed_optimizer is True
    assert cfg.optimizer.use_layer_wise_param_layout is False
    assert cfg.optimizer.overlap_param_gather is True
    assert cfg.optimizer.optimizer_offload_fraction == 1.0
    assert cfg.optimizer.barrier_with_L1_time is True
    assert cfg.scheduler.lr_decay_iters is None
    assert cfg.scheduler.lr_decay_samples == 584_765_624
    assert cfg.scheduler.lr_warmup_iters == 0
    assert cfg.scheduler.lr_warmup_samples == 1_536_000
    assert cfg.optimizer.muon_momentum == 0.9
    assert cfg.optimizer.muon_nesterov is False
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.comm_overlap.overlap_param_gather is True

    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_router_force_load_balancing is True
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.apply_dsa_kernel_fusion is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.recompute_modules == ["mla_up_proj", "mhc"]


def test_deepseek_v4_pro_full_model_uses_64k_thd_recipe() -> None:
    cfg = deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config()

    assert cfg.model.num_layers == 61
    assert cfg.model.moe_layer_freq == [1] * 61
    assert cfg.model.csa_compress_ratios == [128, 128, 4, *([128, 4] * 29), 0]
    assert cfg.model.moe_n_hash_layers == 3
    assert cfg.model.activation_func_clamp_value == 10.0
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.virtual_pipeline_model_parallel_size == 4
    assert cfg.model.context_parallel_size == 16
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.pipeline_model_parallel_layout == "Et*4|(t*4|)*14tmL"

    assert cfg.model.seq_length == 65_536
    assert isinstance(cfg.dataset, MockVarlenDatasetConfig)
    assert cfg.dataset.seq_length == 65_536
    assert cfg.dataset.context_parallel_size == 16
    assert cfg.dataset.data_parallel_size == 4
    assert cfg.model.sequence_packing_scheduler == "dp_balanced"
    assert cfg.model.max_seqlen_per_dp_cp_rank == 4096
    assert cfg.train.global_batch_size == 256
    assert cfg.train.micro_batch_size == 1

    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.moe_paged_stash is False
    assert cfg.model.fine_grained_activation_offloading is True
    assert cfg.model.offload_modules == ["core_attn", "attn_proj"]
    assert cfg.model.fine_grained_offloading_max_inflight_offloads == 2
    assert cfg.model.recompute_modules == ["mla_up_proj", "mhc"]
    assert cfg.optimizer.optimizer == "muon"
    assert cfg.optimizer.use_layer_wise_distributed_optimizer is True
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 64
    assert cfg.env_vars["NVTE_CPU_OFFLOAD_V1"] == 1
