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

import pytest
import torch

from megatron.bridge.models.kimi import KimiK2Provider
from megatron.bridge.recipes.kimi.kimi_k2 import _get_kimi_k2_pipeline_layout, kimi_k2_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class TestKimiK2PipelineLayout:
    """Test cases for _get_kimi_k2_pipeline_layout function."""

    def test_pipeline_layout_pp1_vp1(self):
        """Test pipeline layout for PP=1, VP=1."""
        layout = _get_kimi_k2_pipeline_layout(1, 1)
        assert layout is None

    def test_pipeline_layout_pp16_vp1(self):
        """Test pipeline layout for PP=16, VP=1."""
        layout = _get_kimi_k2_pipeline_layout(16, 1)
        expected_layout = [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]]
        assert layout == expected_layout

    def test_pipeline_layout_pp8_vp2(self):
        """Test pipeline layout for PP=8, VP=2."""
        layout = _get_kimi_k2_pipeline_layout(8, 2)
        expected_layout = [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]]
        assert layout == expected_layout

    def test_pipeline_layout_invalid_pp_vp_combination(self):
        """Test that invalid PP/VP combinations raise ValueError."""
        with pytest.raises(ValueError, match="Invalid PP and VP size"):
            _get_kimi_k2_pipeline_layout(3, 1)


class TestKimiK2PretrainConfig:
    """Test cases for kimi_k2_pretrain_config function."""

    def test_pretrain_config_basic_structure(self):
        """Test that kimi_k2_pretrain_config returns a valid ConfigContainer."""
        cfg = kimi_k2_pretrain_config()

        # Check it returns a ConfigContainer with all required components
        assert isinstance(cfg, ConfigContainer)
        assert isinstance(cfg.model, KimiK2Provider)
        assert cfg.train is not None
        assert cfg.optimizer is not None
        assert cfg.scheduler is not None
        assert cfg.dataset is not None
        assert cfg.tokenizer is not None
        assert cfg.checkpoint is not None
        assert cfg.comm_overlap is not None

    def test_pretrain_config_default_training_settings(self):
        """Test default training settings."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.train.train_iters == 1_000_000
        assert cfg.train.global_batch_size == 4096
        assert cfg.train.micro_batch_size == 1
        assert cfg.train.eval_interval == 2000
        assert cfg.train.manual_gc is True
        assert cfg.train.manual_gc_interval == 5
        assert cfg.train.manual_gc_eval == 5

    def test_pretrain_config_model_parallelism(self):
        """Test default parallelism configuration."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.model.tensor_model_parallel_size == 2
        assert cfg.model.pipeline_model_parallel_size == 16
        assert cfg.model.pipeline_dtype == torch.bfloat16
        assert cfg.model.virtual_pipeline_model_parallel_size is None
        assert cfg.model.context_parallel_size == 1
        assert cfg.model.expert_model_parallel_size == 32
        assert cfg.model.sequence_parallel is True
        assert cfg.model.expert_tensor_parallel_size == 1

    def test_pretrain_config_model_recomputation(self):
        """Test recomputation settings."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.model.recompute_granularity == "selective"
        assert cfg.model.recompute_modules is None
        assert cfg.model.recompute_method is None
        assert cfg.model.recompute_num_layers is None
        assert cfg.model.fine_grained_activation_offloading is False
        assert cfg.model.offload_modules is None

    def test_pretrain_config_pipeline_split_settings(self):
        """Test pipeline split settings."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.model.account_for_embedding_in_pipeline_split is False
        assert cfg.model.account_for_loss_in_pipeline_split is False
        assert cfg.model.num_layers_in_first_pipeline_stage is None
        assert cfg.model.num_layers_in_last_pipeline_stage is None

    def test_pretrain_config_ddp_settings_for_muon(self):
        """Test DDP settings configured for Muon optimizer."""
        cfg = kimi_k2_pretrain_config()

        # Muon requires these specific DDP settings
        assert cfg.ddp.overlap_grad_reduce is True
        assert cfg.ddp.overlap_param_gather is False  # Muon needs this to be False
        assert cfg.ddp.check_for_nan_in_grad is True
        assert cfg.ddp.use_distributed_optimizer is False  # Muon needs this to be False
        assert cfg.ddp.use_megatron_fsdp is False
        assert cfg.ddp.grad_reduce_in_fp32 is True
        assert cfg.ddp.average_in_collective is True
        assert cfg.ddp.data_parallel_sharding_strategy == "no_shard"

    def test_pretrain_config_dataset_configuration(self):
        """Test dataset configuration."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.dataset.sequence_length == 4096
        assert cfg.dataset.num_workers == 8
        assert cfg.dataset.data_sharding is True
        assert cfg.dataset.split == "9999,8,2"
        assert cfg.dataset.blend is None
        assert cfg.dataset.blend_per_split is None

    def test_pretrain_config_tokenizer_configuration(self):
        """Test tokenizer configuration."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        assert cfg.tokenizer.tokenizer_model is None
        assert cfg.tokenizer.vocab_size == cfg.model.vocab_size

    def test_pretrain_config_mixed_precision(self):
        """Test mixed precision configuration."""
        cfg = kimi_k2_pretrain_config()

        assert isinstance(cfg.mixed_precision, MixedPrecisionConfig)
        assert cfg.mixed_precision.bf16 is True
        assert cfg.mixed_precision.params_dtype == torch.bfloat16
        assert cfg.mixed_precision.pipeline_dtype == torch.bfloat16
        assert cfg.mixed_precision.autocast_enabled is False
        assert cfg.mixed_precision.grad_reduce_in_fp32 is True

    def test_pretrain_config_optimizer_precision(self):
        """Test optimizer precision settings."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.optimizer.use_precision_aware_optimizer is False
        assert cfg.optimizer.main_grads_dtype == torch.float32
        assert cfg.optimizer.main_params_dtype == torch.float32
        assert cfg.optimizer.exp_avg_dtype == torch.float32
        assert cfg.optimizer.exp_avg_sq_dtype == torch.float32

    def test_pretrain_config_moe_settings(self):
        """Test MoE-specific configuration."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.model.moe_token_dispatcher_type == "alltoall"
        assert cfg.model.moe_flex_dispatcher_backend == "deepep"
        assert cfg.model.moe_hybridep_num_sms == 16
        assert cfg.model.moe_router_fusion is False
        assert cfg.model.moe_permute_fusion is True
        assert cfg.model.moe_grouped_gemm is True
        assert cfg.model.moe_router_padding_for_fp8 is False
        assert cfg.model.moe_shared_expert_overlap is True
        assert cfg.model.moe_router_force_load_balancing is False

    def test_pretrain_config_transformer_engine_and_cuda_graph(self):
        """Test Transformer Engine and CUDA Graph settings."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.model.transformer_impl == "transformer_engine"
        assert cfg.model.cuda_graph_impl == "none"
        assert cfg.model.cuda_graph_scope == "full"
        assert cfg.model.cuda_graph_warmup_steps == 3

    def test_pretrain_config_kernel_selections(self):
        """Test kernel selection settings."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.model.attention_backend is None
        assert cfg.model.cross_entropy_loss_fusion is True
        assert cfg.model.cross_entropy_fusion_impl == "te"

    def test_pretrain_config_comm_overlap(self):
        """Test communication overlap configuration."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.comm_overlap.tp_comm_overlap is False
        assert cfg.comm_overlap.delay_wgrad_compute is False
        assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False

    def test_pretrain_config_checkpoint(self):
        """Test checkpoint configuration."""
        cfg = kimi_k2_pretrain_config()

        assert cfg.checkpoint.save_interval == 2000
        assert cfg.checkpoint.async_save is False

    def test_pretrain_config_pipeline_layout(self):
        """Test pipeline layout is configured."""
        cfg = kimi_k2_pretrain_config()

        # Default PP=16, VP=None (1), should have a layout
        expected_layout = _get_kimi_k2_pipeline_layout(16, 1)
        assert cfg.model.pipeline_model_parallel_layout == expected_layout
