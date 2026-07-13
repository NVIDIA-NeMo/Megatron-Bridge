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


def create_tiny_llama_hf_source(path: str, seq_length: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    config = LlamaConfig(
        vocab_size=10000,
        hidden_size=Llama3TinyModelProvider.hidden_size,
        intermediate_size=Llama3TinyModelProvider.ffn_hidden_size,
        num_hidden_layers=Llama3TinyModelProvider.num_layers,
        num_attention_heads=Llama3TinyModelProvider.num_attention_heads,
        num_key_value_heads=Llama3TinyModelProvider.num_query_groups,
        max_position_embeddings=seq_length,
        rms_norm_eps=1e-5,
        rope_theta=500000,
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(path, safe_serialization=True)

    vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3}
    vocab.update({f"token_{idx}": idx for idx in range(4, 10000)})
    tokenizer_model = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_model.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(path)


class TestPretrainResume:
    """
    Test end to end training with checkpoint functionality.
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("also_save_hf_checkpoint", [False, True])
    def test_pretrain_save_load(self, tmp_path, also_save_hf_checkpoint):
        """
        Test end to end training with checkpoint saving and resuming functionality.
        """

        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")
        hf_source_dir = os.path.join(shared_base_dir, "tiny_llama_hf_source")

        global_batch_size = 8
        micro_batch_size = 1
        seq_length = 512
        total_iters = 10
        checkpoint_iters = 5

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

            if also_save_hf_checkpoint:
                create_tiny_llama_hf_source(hf_source_dir, seq_length)

        torch.distributed.barrier()

        try:

            def verify_hf_sidecar(iteration: int) -> None:
                if also_save_hf_checkpoint and torch.distributed.get_rank() == 0:
                    hf_dir = os.path.join(checkpoint_dir, f"iter_{iteration:07d}", "hf")
                    assert os.path.exists(os.path.join(hf_dir, "config.json"))
                    assert any(name.endswith(".safetensors") for name in os.listdir(hf_dir))

            # First training run - train to checkpoint_iters and save checkpoint
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
                    also_save_hf_checkpoint=also_save_hf_checkpoint,
                    hf_source_path=hf_source_dir if also_save_hf_checkpoint else None,
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

            verify_hf_sidecar(checkpoint_iters)

            torch.distributed.barrier()

            # Second training run - resume from checkpoint and train to total_iters
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
                    also_save_hf_checkpoint=also_save_hf_checkpoint,
                    hf_source_path=hf_source_dir if also_save_hf_checkpoint else None,
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
            verify_hf_sidecar(total_iters)

        finally:
            clear_directories(shared_base_dir)
