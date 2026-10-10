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
import os
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
from megatron.core import parallel_state


try:
    from megatron.core.tensor_parallel import gtp_api
except ImportError:
    gtp_api = None

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
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
from megatron.bridge.training.gtp import get_transformer_config
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


class Llama32TestModelProvider(GPTModelProvider):
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    num_query_groups: int = 8
    kv_channels: int = 64
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-05
    rotary_percent: float = 1.0
    rotary_base: int = 500_000
    rope_scaling: bool = True
    rope_scaling_factor: float = 32.0
    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    vocab_size: int | None = None


class GTPValidationCallback(Callback):
    """Validate GTP groups, parameters, and finite per-step training results.

    This callback is a GTP runtime smoke test, not an MLM numerical-parity or
    tensor-parity assertion.
    """

    def __init__(self) -> None:
        self.num_steps = 0

    def on_train_start(self, context: CallbackContext) -> None:
        transformer_config = get_transformer_config(context.state.cfg.model)
        gtp_size = transformer_config.gtp_weight_remat_size
        assert gtp_size == 2
        assert parallel_state.get_gtp_weight_remat_world_size() == gtp_size
        assert parallel_state.get_data_parallel_world_size(with_gtp_remat=False) == 2 // gtp_size
        assert parallel_state.get_data_parallel_world_size(with_gtp_remat=True) == 2
        gtp_params = [param for chunk in context.model for param in chunk.parameters() if gtp_api.is_gtp_param(param)]
        assert gtp_params
        assert all(param.chain_id is not None for param in gtp_params)

    def on_train_step_end(self, context: CallbackContext) -> None:
        assert context.skipped_iter == 0
        assert context.grad_norm is not None and torch.isfinite(torch.tensor(context.grad_norm))
        assert context.loss_dict
        assert all(torch.isfinite(loss).all() for loss in context.loss_dict.values())
        self.num_steps += 1

    def on_train_end(self, context: CallbackContext) -> None:
        assert self.num_steps == 3
        assert context.state.train_state.consumed_train_samples == 6


class TestPretrain:
    """
    Test end to end training with checkpoint functionality.
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.skipif(gtp_api is None or not gtp_api.HAVE_GTP, reason="GTP requires MCore GTP support")
    def test_pretrain_with_generalized_tensor_parallelism(self):
        """Train three finite BF16 steps with dense weights sharded over a two-rank GTP axis."""
        initialize_distributed()
        total_iters = 3
        seq_length = 64
        transformer_cfg = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            tensor_model_parallel_size=1,
            tensor_parallel_num_weight_shards=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            add_bias_linear=False,
            gradient_accumulation_fusion=False,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
        )
        model_cfg = BridgeGPTModelConfig(
            transformer=transformer_cfg,
            vocab_size=128,
            seq_length=seq_length,
            share_embeddings_and_output_weights=False,
        )
        cfg = ConfigContainer(
            model=model_cfg,
            train=TrainingConfig(
                train_iters=total_iters,
                global_batch_size=2,
                micro_batch_size=1,
            ),
            validation=ValidationConfig(eval_interval=100, eval_iters=0),
            optimizer=OptimizerConfig(
                optimizer="adam",
                bf16=True,
                fp16=False,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=False,
                clip_grad=1.0,
                lr=3e-4,
                min_lr=3e-5,
            ),
            scheduler=SchedulerConfig(
                start_weight_decay=0.01,
                end_weight_decay=0.01,
                weight_decay_incr_style="constant",
                lr_decay_style="cosine",
                lr_warmup_iters=1,
                lr_decay_iters=total_iters,
                override_opt_param_scheduler=True,
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                average_in_collective=False,
                use_distributed_optimizer=False,
            ),
            dataset=MockGPTDatasetConfig(
                random_seed=1234,
                seq_length=seq_length,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                dataloader_type="single",
                num_workers=0,
            ),
            logger=LoggerConfig(log_interval=1),
            tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=128),
            checkpoint=CheckpointConfig(save=None, load=None),
            rng=RNGConfig(seed=1234),
        )

        callback = GTPValidationCallback()
        pretrain(cfg, forward_step, callbacks=[callback])

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_checkpoint(self, tmp_path):
        """
        Test end to end training with checkpoint functionality.
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
            total_iters = 10

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

            # Config Container
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
                    eval_iters=2,
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
                    lr_warmup_iters=2,
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
                    save_interval=40,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify checkpoint files
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            # pytest's tmp_path fixture doesn't clean up immediately.
            # Clean up manually.
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_independent_eval_batch_size(self, tmp_path):
        """
        Test end to end training with eval batch sizes different from training batch sizes.

        Uses eval_global_batch_size=4 and eval_micro_batch_size=2, while training uses
        global_batch_size=8 and micro_batch_size=1. This verifies that the eval path
        correctly uses the independent eval batch configuration.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        global_batch_size = 8
        micro_batch_size = 1
        eval_global_batch_size = 4
        eval_micro_batch_size = 2
        seq_length = 512
        total_iters = 10

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
                eval_iters=2,
                eval_global_batch_size=eval_global_batch_size,
                eval_micro_batch_size=eval_micro_batch_size,
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
                lr_warmup_iters=2,
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
                ckpt_format="torch_dist",
            ),
            rng=RNGConfig(seed=1234),
        )

        # Run training — eval runs at iter 5 and 10 with independent batch sizes
        try:
            pretrain(cfg, forward_step)
        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.pleasefixme
    def test_pretrain_with_mup(self, tmp_path, caplog):
        """
        Test end to end training with μP (Maximal Update Parameterization) enabled.

        Verifies that use_mup=True flows through the full training stack: the model
        config's mup_width_mult is computed by finalize(), get_model_config() on the
        DDP-wrapped model still returns use_mup=True, and setup_optimizer applies the
        per-parameter-class LR overrides without error.

        Uses mup_base_hidden_size=1024 with hidden_size=2048 (width_mult=2.0) so that
        the LR scaling is non-trivial and any failure to apply overrides would be visible.
        """
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 5

            model_cfg = Llama32TestModelProvider(
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
                use_mup=True,
                mup_base_hidden_size=1024,  # width_mult = 2048/1024 = 2.0
            )

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
                    eval_iters=2,
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
                    save_interval=100,
                    ckpt_format="torch_dist",
                ),
                rng=RNGConfig(seed=1234),
            )

            with caplog.at_level(logging.INFO, logger="megatron.bridge.training.optim"):
                pretrain(cfg, forward_step)

            # Assert μP optimizer overrides were applied (not just a smoke test)
            mup_log_messages = [r.message for r in caplog.records if "μP enabled" in r.message]
            assert mup_log_messages, (
                "Expected μP optimizer override log message but found none. "
                "Check that use_mup=True flows through setup_optimizer."
            )
            assert "width_mult=2" in mup_log_messages[0], (
                f"Expected width_mult=2 in μP log, got: {mup_log_messages[0]}"
            )

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_vpp(self, tmp_path):
        """
        Test end to end training with virtual pipeline parallelism.
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
            total_iters = 10

            # Create model config with VPP
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
                num_layers=16,
                hidden_size=2048,
                ffn_hidden_size=8192,
                num_attention_heads=32,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                context_parallel_size=1,
                sequence_parallel=False,
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
                vocab_size=None,
            )

            # Create other configurations
            optimizer_cfg = OptimizerConfig(
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
            )

            ddp_cfg = DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=True,
                use_distributed_optimizer=True,
            )

            # Setup communication overlap for VPP
            from megatron.bridge.training.comm_overlap import CommOverlapConfig

            comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False,
            )

            # Create config container
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
                    eval_iters=2,
                ),
                optimizer=optimizer_cfg,
                scheduler=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=2,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp=ddp_cfg,
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
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                ),
                rng=RNGConfig(seed=1234),
                comm_overlap=comm_overlap,
            )

            # Run training
            pretrain(cfg, forward_step)

            # Verify checkpoint files
            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

        finally:
            # pytest's tmp_path fixture doesn't clean up immediately.
            # Clean up manually.
            clear_directories(tmp_path)
