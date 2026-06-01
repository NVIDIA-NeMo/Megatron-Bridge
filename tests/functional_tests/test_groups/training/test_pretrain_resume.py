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

import os
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
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
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


@dataclass
class Llama3TinyModelProvider(GPTModelProvider):
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
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-05
    rotary_percent: float = 1.0
    rotary_base: int = 500_000
    seq_length: int = 1024
    num_layers: int = 1
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int | None = None


class TestPretrainResume:
    """
    Test end to end training with checkpoint functionality.
    """

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_save_load(self, tmp_path):
        """
        Test end to end training with checkpoint saving and resuming functionality.
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
            checkpoint_iters = 10

            # First training run - train for 10 iterations and save checkpoint
            cfg_first = ConfigContainer(
                model=Llama3TinyModelProvider(seq_length=seq_length),
                train=TrainingConfig(
                    train_iters=checkpoint_iters,
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
                    save_interval=checkpoint_iters,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                    dist_ckpt_optim_fully_reshardable=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run first training job
            pretrain(cfg_first, forward_step)

            torch.distributed.barrier()

            # Verify checkpoint files from first run
            verify_checkpoint_files(
                checkpoint_dir,
                checkpoint_iters,
                ckpt_format=cfg_first.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg_first.checkpoint.storage_writers_per_rank,
            )

            torch.distributed.barrier()

            # Second training run - resume from checkpoint and train for remaining 10 iterations
            cfg_second = ConfigContainer(
                model=Llama3TinyModelProvider(seq_length=seq_length),
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
                    save_interval=checkpoint_iters,
                    save=checkpoint_dir,
                    load=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                    dist_ckpt_optim_fully_reshardable=True,
                ),
                rng=RNGConfig(seed=1234),
            )

            # Run second training job (resume from checkpoint)
            pretrain(cfg_second, forward_step)

            torch.distributed.barrier()

            # Verify checkpoint files from second run
            verify_checkpoint_files(
                checkpoint_dir,
                total_iters,
                ckpt_format=cfg_second.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg_second.checkpoint.storage_writers_per_rank,
            )

        finally:
            clear_directories(shared_base_dir)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_save_load_with_hf_sidecar(self, tmp_path, monkeypatch):
        """
        Test checkpoint save/resume when an additional HuggingFace sidecar is exported.

        The HF export itself is stubbed to keep the functional test lightweight; the
        test exercises the training checkpoint integration: Megatron async save is
        scheduled, the ``iter_*/hf/`` sidecar is written, and resume from the parent
        save directory still loads the native Megatron checkpoint.
        """

        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints_hf_sidecar")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard_hf_sidecar")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        def fake_save_hf_weights(state, model, iter_dir):
            if torch.distributed.get_rank() == 0:
                os.makedirs(iter_dir, exist_ok=True)
                with open(os.path.join(iter_dir, "hf_sidecar_marker.txt"), "w") as f:
                    f.write(f"step={state.train_state.step}\n")

        monkeypatch.setattr("megatron.bridge.training.checkpointing._save_hf_weights", fake_save_hf_weights)

        def make_config(train_iters, *, load=None):
            global_batch_size = 2
            micro_batch_size = 1
            seq_length = 128
            save_interval = 2

            return ConfigContainer(
                model=Llama3TinyModelProvider(seq_length=seq_length),
                train=TrainingConfig(
                    train_iters=train_iters,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                validation=ValidationConfig(
                    eval_interval=train_iters + 1,
                    eval_iters=1,
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
                    lr_decay_iters=4,
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
                    log_interval=1,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint=CheckpointConfig(
                    save_interval=save_interval,
                    save=checkpoint_dir,
                    load=load,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                    async_save=True,
                    dist_ckpt_optim_fully_reshardable=True,
                    also_save_hf_checkpoint=True,
                    hf_source_path="/mock/hf-source",
                ),
                rng=RNGConfig(seed=1234),
            )

        try:
            first_iters = 2
            total_iters = 4

            cfg_first = make_config(first_iters)
            pretrain(cfg_first, forward_step)

            torch.distributed.barrier()

            verify_checkpoint_files(
                checkpoint_dir,
                first_iters,
                ckpt_format=cfg_first.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg_first.checkpoint.storage_writers_per_rank,
            )

            first_hf_marker = os.path.join(checkpoint_dir, f"iter_{first_iters:07d}", "hf", "hf_sidecar_marker.txt")
            if torch.distributed.get_rank() == 0:
                assert os.path.exists(first_hf_marker)

            torch.distributed.barrier()

            cfg_second = make_config(total_iters, load=checkpoint_dir)
            pretrain(cfg_second, forward_step)

            torch.distributed.barrier()

            verify_checkpoint_files(
                checkpoint_dir,
                total_iters,
                ckpt_format=cfg_second.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg_second.checkpoint.storage_writers_per_rank,
            )

            second_hf_marker = os.path.join(checkpoint_dir, f"iter_{total_iters:07d}", "hf", "hf_sidecar_marker.txt")
            if torch.distributed.get_rank() == 0:
                assert os.path.exists(second_hf_marker)

        finally:
            clear_directories(shared_base_dir)
