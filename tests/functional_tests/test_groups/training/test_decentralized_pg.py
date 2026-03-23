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

"""
Functional tests for the use_decentralized_pg feature.

This feature enables using ProcessGroupCollection passed through functions instead
of relying on mcore's global parallel state (mpu) variables. When enabled, parallel
groups are obtained from the pg_collection object rather than the global
megatron.core.parallel_state module.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.initialize import destroy_global_state
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


@pytest.fixture(autouse=True)
def cleanup_megatron_state():
    """Cleanup Megatron global state after each test.

    This fixture ensures that global state is cleaned up even if a test fails,
    preventing state leakage between tests when running multiple tests in the
    same pytest session.
    """
    yield
    # Cleanup after test (runs even if test fails)
    try:
        destroy_global_state()
    except Exception:
        # Ignore errors during cleanup - state might not have been initialized
        pass


class TestDecentralizedPgPretrain:
    """
    Functional tests for pretraining with use_decentralized_pg enabled.
    """

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_decentralized_pg(self, tmp_path):
        """
        Test end to end training with use_decentralized_pg=True.

        This test verifies that training works correctly when parallel groups
        are passed through functions instead of using global mpu state.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 5

            model_cfg = GPTModelProvider(
                normalization="RMSNorm",
                activation_func=F.silu,
                gated_linear_unit=True,
                position_embedding_type="rope",
                add_bias_linear=False,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                bias_activation_fusion=True,
                masked_softmax_fusion=True,
                persist_layer_norm=True,
                bias_dropout_fusion=True,
                apply_rope_fusion=True,
                num_query_groups=8,
                init_method_std=0.02,
                layernorm_epsilon=1e-05,
                rotary_percent=1.0,
                rope_scaling=True,
                rope_scaling_factor=32.0,
                rotary_base=500_000,
                hidden_size=2048,
                ffn_hidden_size=8192,
                num_attention_heads=32,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
                num_layers=1,
                # Disable shared embeddings - not supported with decentralized PG
                share_embeddings_and_output_weights=False,
            )

            # Config Container with use_decentralized_pg=True
            cfg = ConfigContainer(
                model=model_cfg,
                train=TrainingConfig(
                    train_iters=total_iters,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                validation=ValidationConfig(
                    eval_interval=5,
                    eval_iters=0,
                ),
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    bf16=True,
                    fp16=False,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                    use_distributed_optimizer=True,
                    clip_grad=1.0,
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp=DistributedDataParallelConfig(
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                    use_distributed_optimizer=True,
                ),
                dist=DistributedInitConfig(
                    use_decentralized_pg=True,  # Enable the feature
                    use_gloo_process_groups=False,  # Gloo not supported with custom pg_collection
                ),
                dataset=MockGPTDatasetConfig(
                    random_seed=1234,
                    reset_attention_mask=False,
                    reset_position_ids=False,
                    eod_mask_loss=False,
                    seq_length=seq_length,
                    num_dataset_builder_threads=1,
                    data_sharding=True,
                    dataloader_type="single",
                    num_workers=1,
                ),
                logger=LoggerConfig(
                    log_interval=5,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=total_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify training completed
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_decentralized_pg_disabled(self, tmp_path):
        """
        Test end to end training with use_decentralized_pg=False (default).

        This test verifies that training works correctly with the default
        behavior using global mpu state.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 5

            model_cfg = GPTModelProvider(
                normalization="RMSNorm",
                activation_func=F.silu,
                gated_linear_unit=True,
                position_embedding_type="rope",
                add_bias_linear=False,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                bias_activation_fusion=True,
                masked_softmax_fusion=True,
                persist_layer_norm=True,
                bias_dropout_fusion=True,
                apply_rope_fusion=True,
                num_query_groups=8,
                init_method_std=0.02,
                layernorm_epsilon=1e-05,
                rotary_percent=1.0,
                rope_scaling=True,
                rope_scaling_factor=32.0,
                share_embeddings_and_output_weights=True,
                rotary_base=500_000,
                hidden_size=2048,
                ffn_hidden_size=8192,
                num_attention_heads=32,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
                num_layers=1,
            )

            # Config Container with use_decentralized_pg=False (default)
            cfg = ConfigContainer(
                model=model_cfg,
                train=TrainingConfig(
                    train_iters=total_iters,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                validation=ValidationConfig(
                    eval_interval=5,
                    eval_iters=0,
                ),
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    bf16=True,
                    fp16=False,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                    use_distributed_optimizer=True,
                    clip_grad=1.0,
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp=DistributedDataParallelConfig(
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                    use_distributed_optimizer=True,
                ),
                dist=DistributedInitConfig(
                    use_decentralized_pg=False,  # Explicitly disable (default)
                ),
                dataset=MockGPTDatasetConfig(
                    random_seed=1234,
                    reset_attention_mask=False,
                    reset_position_ids=False,
                    eod_mask_loss=False,
                    seq_length=seq_length,
                    num_dataset_builder_threads=1,
                    data_sharding=True,
                    dataloader_type="single",
                    num_workers=1,
                ),
                logger=LoggerConfig(
                    log_interval=5,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=total_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify training completed
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            clear_directories(tmp_path)

    #
    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_decentralized_pg_and_pp(self, tmp_path):
        """
        Test training with use_decentralized_pg=True and pipeline parallelism.

        This test verifies that the decentralized process groups feature works correctly
        with pipeline parallelism enabled.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        # Skip if world size is not at least 2 for PP
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            pytest.skip("This test requires at least 2 GPUs for PP=2")

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 5

            model_cfg = GPTModelProvider(
                normalization="RMSNorm",
                activation_func=F.silu,
                gated_linear_unit=True,
                position_embedding_type="rope",
                add_bias_linear=False,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                bias_activation_fusion=True,
                masked_softmax_fusion=True,
                persist_layer_norm=True,
                bias_dropout_fusion=True,
                apply_rope_fusion=True,
                num_query_groups=8,
                init_method_std=0.02,
                layernorm_epsilon=1e-05,
                rotary_percent=1.0,
                rope_scaling=True,
                rope_scaling_factor=32.0,
                rotary_base=500_000,
                hidden_size=2048,
                ffn_hidden_size=8192,
                num_attention_heads=32,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,  # Enable PP
                context_parallel_size=1,
                sequence_parallel=False,
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
                num_layers=2,  # Need at least 2 layers for PP=2
                # Disable shared embeddings - not supported with decentralized PG
                share_embeddings_and_output_weights=False,
            )

            # Config Container with use_decentralized_pg=True
            cfg = ConfigContainer(
                model=model_cfg,
                train=TrainingConfig(
                    train_iters=total_iters,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                validation=ValidationConfig(
                    eval_interval=5,
                    eval_iters=0,
                ),
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    bf16=True,
                    fp16=False,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                    use_distributed_optimizer=True,
                    clip_grad=1.0,
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp=DistributedDataParallelConfig(
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                    use_distributed_optimizer=True,
                ),
                dist=DistributedInitConfig(
                    use_decentralized_pg=True,  # Enable the feature
                    use_gloo_process_groups=False,  # Gloo not supported with custom pg_collection
                ),
                dataset=MockGPTDatasetConfig(
                    random_seed=1234,
                    reset_attention_mask=False,
                    reset_position_ids=False,
                    eod_mask_loss=False,
                    seq_length=seq_length,
                    num_dataset_builder_threads=1,
                    data_sharding=True,
                    dataloader_type="single",
                    num_workers=1,
                ),
                logger=LoggerConfig(
                    log_interval=5,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=total_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify training completed
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_decentralized_pg_and_cp(self, tmp_path):
        """
        Test training with use_decentralized_pg=True and context parallelism.

        This test verifies that the decentralized process groups feature works correctly
        with context parallelism enabled.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        # Skip if world size is not at least 2 for CP
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            pytest.skip("This test requires at least 2 GPUs for CP=2")

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 5

            model_cfg = GPTModelProvider(
                normalization="RMSNorm",
                activation_func=F.silu,
                gated_linear_unit=True,
                position_embedding_type="rope",
                add_bias_linear=False,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                bias_activation_fusion=True,
                masked_softmax_fusion=True,
                persist_layer_norm=True,
                bias_dropout_fusion=True,
                apply_rope_fusion=True,
                num_query_groups=8,
                init_method_std=0.02,
                layernorm_epsilon=1e-05,
                rotary_percent=1.0,
                rope_scaling=True,
                rope_scaling_factor=32.0,
                rotary_base=500_000,
                hidden_size=2048,
                ffn_hidden_size=8192,
                num_attention_heads=32,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=2,  # Enable CP
                sequence_parallel=False,
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
                num_layers=1,
                # Disable shared embeddings - not supported with decentralized PG
                share_embeddings_and_output_weights=False,
            )

            # Config Container with use_decentralized_pg=True
            cfg = ConfigContainer(
                model=model_cfg,
                train=TrainingConfig(
                    train_iters=total_iters,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                validation=ValidationConfig(
                    eval_interval=5,
                    eval_iters=0,
                ),
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    bf16=True,
                    fp16=False,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                    use_distributed_optimizer=True,
                    clip_grad=1.0,
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp=DistributedDataParallelConfig(
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                    use_distributed_optimizer=True,
                ),
                dist=DistributedInitConfig(
                    use_decentralized_pg=True,  # Enable the feature
                    use_gloo_process_groups=False,  # Gloo not supported with custom pg_collection
                ),
                dataset=MockGPTDatasetConfig(
                    random_seed=1234,
                    reset_attention_mask=False,
                    reset_position_ids=False,
                    eod_mask_loss=False,
                    seq_length=seq_length,
                    num_dataset_builder_threads=1,
                    data_sharding=True,
                    dataloader_type="single",
                    num_workers=1,
                ),
                logger=LoggerConfig(
                    log_interval=5,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=total_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify training completed
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_decentralized_pg_combined_parallelism(self, tmp_path):
        """
        Test training with use_decentralized_pg=True and combined TP+PP.

        This test verifies that the decentralized process groups feature works correctly
        with multiple forms of parallelism enabled simultaneously.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        # Skip if world size is not at least 4 for TP=2, PP=2
        world_size = torch.distributed.get_world_size()
        if world_size < 4:
            pytest.skip("This test requires at least 4 GPUs for TP=2, PP=2")

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 5

            model_cfg = GPTModelProvider(
                normalization="RMSNorm",
                activation_func=F.silu,
                gated_linear_unit=True,
                position_embedding_type="rope",
                add_bias_linear=False,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                bias_activation_fusion=True,
                masked_softmax_fusion=True,
                persist_layer_norm=True,
                bias_dropout_fusion=True,
                apply_rope_fusion=True,
                num_query_groups=8,
                init_method_std=0.02,
                layernorm_epsilon=1e-05,
                rotary_percent=1.0,
                rope_scaling=True,
                rope_scaling_factor=32.0,
                rotary_base=500_000,
                hidden_size=2048,
                ffn_hidden_size=8192,
                num_attention_heads=32,
                tensor_model_parallel_size=2,  # Enable TP
                pipeline_model_parallel_size=2,  # Enable PP
                context_parallel_size=1,
                sequence_parallel=True,  # Usually used with TP
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
                num_layers=2,  # Need at least 2 layers for PP=2
                # Disable shared embeddings - not supported with decentralized PG
                share_embeddings_and_output_weights=False,
            )

            # Config Container with use_decentralized_pg=True
            cfg = ConfigContainer(
                model=model_cfg,
                train=TrainingConfig(
                    train_iters=total_iters,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                validation=ValidationConfig(
                    eval_interval=5,
                    eval_iters=0,
                ),
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    bf16=True,
                    fp16=False,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                    use_distributed_optimizer=True,
                    clip_grad=1.0,
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp=DistributedDataParallelConfig(
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                    use_distributed_optimizer=True,
                ),
                dist=DistributedInitConfig(
                    use_decentralized_pg=True,  # Enable the feature
                    use_gloo_process_groups=False,  # Gloo not supported with custom pg_collection
                ),
                dataset=MockGPTDatasetConfig(
                    random_seed=1234,
                    reset_attention_mask=False,
                    reset_position_ids=False,
                    eod_mask_loss=False,
                    seq_length=seq_length,
                    num_dataset_builder_threads=1,
                    data_sharding=True,
                    dataloader_type="single",
                    num_workers=1,
                ),
                logger=LoggerConfig(
                    log_interval=5,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=total_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify training completed
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_decentralized_pg_and_tp(self, tmp_path):
        """
        Test training with use_decentralized_pg=True and tensor parallelism.

        This test verifies that the decentralized process groups feature works correctly
        with tensor parallelism enabled.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        # Skip if world size is not at least 2 for TP
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            pytest.skip("This test requires at least 2 GPUs for TP=2")

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 5

            model_cfg = GPTModelProvider(
                normalization="RMSNorm",
                activation_func=F.silu,
                gated_linear_unit=True,
                position_embedding_type="rope",
                add_bias_linear=False,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                bias_activation_fusion=True,
                masked_softmax_fusion=True,
                persist_layer_norm=True,
                bias_dropout_fusion=True,
                apply_rope_fusion=True,
                num_query_groups=8,
                init_method_std=0.02,
                layernorm_epsilon=1e-05,
                rotary_percent=1.0,
                rope_scaling=True,
                rope_scaling_factor=32.0,
                rotary_base=500_000,
                hidden_size=2048,
                ffn_hidden_size=8192,
                num_attention_heads=32,
                tensor_model_parallel_size=2,  # Enable TP
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=True,  # Usually used with TP
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
                num_layers=1,
                # Disable shared embeddings - not supported with decentralized PG
                share_embeddings_and_output_weights=False,
            )

            # Config Container with use_decentralized_pg=True
            cfg = ConfigContainer(
                model=model_cfg,
                train=TrainingConfig(
                    train_iters=total_iters,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                validation=ValidationConfig(
                    eval_interval=5,
                    eval_iters=0,
                ),
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    bf16=True,
                    fp16=False,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                    use_distributed_optimizer=True,
                    clip_grad=1.0,
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=1,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp=DistributedDataParallelConfig(
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                    use_distributed_optimizer=True,
                ),
                dist=DistributedInitConfig(
                    use_decentralized_pg=True,  # Enable the feature
                    use_gloo_process_groups=False,  # Gloo not supported with custom pg_collection
                ),
                dataset=MockGPTDatasetConfig(
                    random_seed=1234,
                    reset_attention_mask=False,
                    reset_position_ids=False,
                    eod_mask_loss=False,
                    seq_length=seq_length,
                    num_dataset_builder_threads=1,
                    data_sharding=True,
                    dataloader_type="single",
                    num_workers=1,
                ),
                logger=LoggerConfig(
                    log_interval=5,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=total_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify training completed
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            clear_directories(tmp_path)
