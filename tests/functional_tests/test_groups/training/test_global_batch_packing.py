# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Two-GPU end-to-end runs of global-batch online packing (GPT-SFT) on the pinned Megatron-Core."""

import gc
import json
import os

import pytest
import torch

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.recipes.qwen.h100.qwen3 import (
    qwen3_600m_pretrain_1gpu_h100_bf16_config,
    qwen3_600m_sft_1gpu_h100_bf16_config,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


SEQ_LENGTH = 512


def _set_existing_attr(target: object, name: str, value: object) -> None:
    if not hasattr(target, name):
        raise ValueError(f"{type(target).__name__} has no field {name!r}")
    setattr(target, name, value)


def _make_model_small(model: object) -> None:
    for name, value in {
        "num_layers": 2,
        "hidden_size": 128,
        "ffn_hidden_size": 512,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "kv_channels": 32,
        "seq_length": SEQ_LENGTH,
    }.items():
        _set_existing_attr(model, name, value)


def _configure_global_batch_packing(cfg, *, context_parallel_size: int) -> None:
    """Small layout for global-batch packing: TP1 PP1, the rest of the two GPUs as CP or DP."""
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = context_parallel_size
    cfg.model.calculate_per_token_loss = True
    # One packed microbatch holds up to SEQ_LENGTH tokens across the CP group.
    cfg.model.max_seqlen_per_dp_cp_rank = SEQ_LENGTH // context_parallel_size
    cfg.ddp.average_in_collective = False
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 8
    cfg.validation.eval_interval = 1
    cfg.validation.eval_iters = 0
    cfg.scheduler.lr_warmup_iters = 0
    cfg.logger.log_interval = 1


class TestGlobalBatchPacking:
    """Global-batch online packing through pretrain and finetune on two GPUs."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("context_parallel_size", [1, 2], ids=["dp2", "cp2"])
    def test_finetune_packs_gpt_sft_rows(self, tmp_path, context_parallel_size):
        pytest.importorskip("transformer_engine_torch")
        initialize_distributed()
        if torch.distributed.get_world_size() < 2:
            pytest.skip("requires 2 GPUs")

        shared_dir = broadcast_path(tmp_path)
        pretrain_checkpoint_dir = os.path.join(shared_dir, "pretrain_checkpoints")
        sft_checkpoint_dir = os.path.join(shared_dir, "sft_checkpoints")
        tensorboard_dir = os.path.join(shared_dir, "tensorboard")
        dataset_root = os.path.join(shared_dir, "sft_data")
        if torch.distributed.get_rank() == 0:
            for directory in (pretrain_checkpoint_dir, sft_checkpoint_dir, tensorboard_dir, dataset_root):
                os.makedirs(directory, exist_ok=True)
            # Answers of very different lengths, so packs mix short and long rows.
            rows = [
                {"input": f"Question {idx}: repeat the word {idx}.", "output": " ".join([str(idx)] * (1 + 7 * idx))}
                for idx in range(32)
            ]
            with open(os.path.join(dataset_root, "training.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        torch.distributed.barrier()

        pretrain_cfg = qwen3_600m_pretrain_1gpu_h100_bf16_config()
        _make_model_small(pretrain_cfg.model)
        pretrain_cfg.model.tensor_model_parallel_size = 1
        pretrain_cfg.model.pipeline_model_parallel_size = 1
        pretrain_cfg.model.context_parallel_size = context_parallel_size
        pretrain_cfg.dataset.seq_length = SEQ_LENGTH
        pretrain_cfg.train.train_iters = 1
        pretrain_cfg.train.global_batch_size = 2
        pretrain_cfg.train.micro_batch_size = 1
        pretrain_cfg.validation.eval_interval = 1
        pretrain_cfg.validation.eval_iters = 0
        pretrain_cfg.scheduler.lr_warmup_iters = 0
        pretrain_cfg.logger.log_interval = 1
        pretrain_cfg.logger.tensorboard_dir = tensorboard_dir
        pretrain_cfg.checkpoint.save_interval = pretrain_cfg.train.train_iters
        pretrain_cfg.checkpoint.save = pretrain_checkpoint_dir
        pretrain_cfg.checkpoint.load = None

        cfg = qwen3_600m_sft_1gpu_h100_bf16_config()
        _make_model_small(cfg.model)
        _configure_global_batch_packing(cfg, context_parallel_size=context_parallel_size)
        cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
        cfg.tokenizer.tokenizer_model = "gpt2"
        cfg.ddp.grad_reduce_in_fp32 = False
        cfg.ddp.use_distributed_optimizer = False
        cfg.optimizer.use_distributed_optimizer = False
        cfg.train.train_iters = 2
        cfg.logger.tensorboard_dir = tensorboard_dir
        cfg.dataset = GPTSFTDatasetConfig(
            dataset_root=dataset_root,
            seq_length=SEQ_LENGTH,
            dataloader_type="single",
            num_workers=1,
            do_validation=False,
            do_test=False,
            max_train_samples=32,
            enable_global_batch_packing=True,
        )
        cfg.checkpoint.save_interval = cfg.train.train_iters
        cfg.checkpoint.save = sft_checkpoint_dir
        cfg.checkpoint.load = None
        cfg.checkpoint.pretrained_checkpoint = pretrain_checkpoint_dir

        try:
            pretrain(pretrain_cfg, forward_step)
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.barrier()

            finetune(cfg, forward_step)
            # Validation derived the scheduler and the CP alignment (2 * cp, or 1 without CP).
            assert cfg.model.sequence_packing_scheduler == "dp_balanced"
            assert cfg.dataset.global_batch_packing_pad_to_multiple_of == (
                2 * context_parallel_size if context_parallel_size > 1 else 1
            )
            verify_checkpoint_files(
                sft_checkpoint_dir,
                cfg.train.train_iters,
                ckpt_format=cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg.checkpoint.storage_writers_per_rank,
            )
        finally:
            clear_directories(shared_dir)
