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

from megatron.bridge.models.gpt_provider import GPTProvider175B
from megatron.bridge.recipes.gpt.gpt3_175b import gpt3_175b_pretrain_config
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestPretrainConfig:
    """Test cases for the pretrain_config function."""

    def test_pretrain_config_basic_structure(self):
        """Test pretrain_config returns a valid ConfigContainer."""
        config = gpt3_175b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, GPTProvider175B)
        assert config.train is not None
        assert config.optimizer is not None
        assert config.scheduler is not None
        assert config.dataset is not None
        assert config.tokenizer is not None
        assert config.checkpoint is not None
        assert config.comm_overlap is not None

    def test_pretrain_config_model_parallelism(self):
        """Test model parallelism configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 8
        assert config.model.pipeline_dtype == torch.bfloat16
        assert config.model.virtual_pipeline_model_parallel_size == 6
        assert config.model.context_parallel_size == 1
        assert config.model.sequence_parallel is True
        assert config.model.pipeline_model_parallel_layout is None

    def test_pretrain_config_training_settings(self):
        """Test training configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.train.train_iters == 1_168_251
        assert config.train.global_batch_size == 2048
        assert config.train.micro_batch_size == 2
        assert config.validation.eval_interval == 2000
        assert config.validation.eval_iters == 32
        assert config.train.manual_gc is True
        assert config.train.manual_gc_interval == 100
        assert config.train.manual_gc_eval == 100

    def test_pretrain_config_optimizer_configuration(self):
        """Test optimizer configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.optimizer.optimizer == "adam"
        assert config.optimizer.lr == 0.9e-4
        assert config.optimizer.min_lr == 0.9e-5
        assert config.optimizer.weight_decay == 0.1
        assert config.optimizer.bf16 is True
        assert config.optimizer.fp16 is False
        assert config.optimizer.use_precision_aware_optimizer is False
        assert config.optimizer.main_grads_dtype == torch.float32
        assert config.optimizer.main_params_dtype == torch.float32
        assert config.optimizer.exp_avg_dtype == torch.float32
        assert config.optimizer.exp_avg_sq_dtype == torch.float32

    def test_pretrain_config_dataset_configuration(self):
        """Test dataset configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.dataset.seq_length == 2048
        assert config.dataset.num_workers == 8
        assert config.dataset.split == "9999,8,2"
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None
        assert config.dataset.reset_attention_mask is False
        assert config.dataset.reset_position_ids is False
        assert config.dataset.eod_mask_loss is False
        assert config.dataset.num_dataset_builder_threads == 1
        assert config.dataset.data_sharding is True
        assert config.dataset.dataloader_type == "single"
        assert config.dataset.random_seed == 1234

    def test_pretrain_config_tokenizer_configuration(self):
        """Test tokenizer configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None
        assert config.tokenizer.vocab_size == DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    def test_pretrain_config_transformer_engine_and_cuda_graph(self):
        """Test Transformer Engine and CUDA Graph settings."""
        config = gpt3_175b_pretrain_config()

        assert config.model.transformer_impl == "transformer_engine"
        assert config.model.cuda_graph_impl == "none"
        assert config.model.cuda_graph_scope == "full"
        assert config.model.cuda_graph_warmup_steps == 3

    def test_pretrain_config_kernel_selections(self):
        """Test kernel selection settings."""
        config = gpt3_175b_pretrain_config()

        assert config.model.attention_backend is None
        assert config.model.cross_entropy_loss_fusion is True
        assert config.model.cross_entropy_fusion_impl == "native"

    def test_pretrain_config_recomputation_and_offloading(self):
        """Test recomputation and offloading settings."""
        config = gpt3_175b_pretrain_config()

        assert config.model.recompute_granularity is None
        assert config.model.recompute_modules is None
        assert config.model.fine_grained_activation_offloading is False
        assert config.model.offload_modules is None

    def test_pretrain_config_checkpoint_configuration(self):
        """Test checkpoint configuration in pretrain_config."""
        config = gpt3_175b_pretrain_config()

        assert config.checkpoint.save_interval == 2000
        assert config.checkpoint.ckpt_format == "torch_dist"
        assert config.checkpoint.fully_parallel_save is True

    def test_pretrain_config_ddp_configuration(self):
        """Test distributed data parallel configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is True
        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.use_distributed_optimizer is True
        assert config.ddp.use_megatron_fsdp is False
        assert config.ddp.grad_reduce_in_fp32 is True
        assert config.ddp.average_in_collective is True
        assert config.ddp.data_parallel_sharding_strategy == "no_shard"

    def test_pretrain_config_comm_overlap(self):
        """Test default CommOverlapConfig setup for GPT3 175B."""
        config = gpt3_175b_pretrain_config()

        # GPT3 175B should have advanced comm overlap enabled by default
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True
        assert config.comm_overlap.defer_embedding_wgrad_compute is True
        assert config.comm_overlap.wgrad_deferral_limit == 50
        assert config.comm_overlap.overlap_param_gather_with_optimizer_step is False

    def test_pretrain_config_scheduler_configuration(self):
        """Test scheduler configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.scheduler.start_weight_decay == 0.033
        assert config.scheduler.end_weight_decay == 0.033
        assert config.scheduler.weight_decay_incr_style == "constant"
        assert config.scheduler.lr_decay_style == "cosine"
        assert config.scheduler.lr_warmup_iters == 2000
        assert config.scheduler.lr_warmup_init == 0.0
        assert config.scheduler.lr_decay_iters is None  # Will be set to train_iters during validation
        assert config.scheduler.override_opt_param_scheduler is True

    def test_pretrain_config_rng_configuration(self):
        """Test RNG configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.rng.seed == 1234

    def test_pretrain_config_logger_configuration(self):
        """Test logger configuration."""
        config = gpt3_175b_pretrain_config()

        assert config.logger.log_interval == 10
        assert "tb_logs" in config.logger.tensorboard_dir
        assert config.logger.log_timers_to_tensorboard is True

    def test_pretrain_config_precision_configuration(self):
        """Test precision configuration for GPT3 175B."""
        config = gpt3_175b_pretrain_config()

        # Should have precision config
        assert config.mixed_precision is not None
        # Grad reduce should be optimized for performance
        assert config.mixed_precision.grad_reduce_in_fp32 is False
