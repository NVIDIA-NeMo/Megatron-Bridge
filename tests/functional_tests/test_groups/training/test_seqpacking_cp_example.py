# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import gc
import json
import os

import pytest
import torch

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.data.packing import PackedSequenceSpecs
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


def _set_existing_attr(target: object, name: str, value: object) -> None:
    if not hasattr(target, name):
        raise ValueError(f"{type(target).__name__} has no field {name!r}")
    setattr(target, name, value)


def _make_functional_test_model_small(model: object) -> None:
    # Keep this checkpoint-loading functional test far below runner memory limits.
    # The path under test is CP + sequence packing + pretrained checkpoint loading,
    # not the full Llama 3.2 1B model shape.
    for name, value in {
        "num_layers": 2,
        "hidden_size": 128,
        "ffn_hidden_size": 512,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "kv_channels": 32,
        "seq_length": 256,
    }.items():
        _set_existing_attr(model, name, value)


class TestPeftSftExample:
    """Run the PEFT SFT example as a functional test with packed sequences + CP."""

    @pytest.mark.run_only_on("GPU")
    def test_sft_example_runs_with_cp_and_packing(self, tmp_path):
        pytest.importorskip("transformer_engine_torch")
        initialize_distributed()

        if torch.distributed.get_world_size() < 2:
            pytest.skip("requires >=2 GPUs for context_parallel_size=2")

        shared_dir = broadcast_path(tmp_path)
        pretrain_checkpoint_dir = os.path.join(shared_dir, "pretrain_checkpoints")
        pretrain_tensorboard_dir = os.path.join(shared_dir, "pretrain_tensorboard")
        sft_checkpoint_dir = os.path.join(shared_dir, "sft_checkpoints")
        sft_tensorboard_dir = os.path.join(shared_dir, "sft_tensorboard")
        dataset_root = os.path.join(shared_dir, "sft_data")

        if torch.distributed.get_rank() == 0:
            os.makedirs(pretrain_checkpoint_dir, exist_ok=True)
            os.makedirs(pretrain_tensorboard_dir, exist_ok=True)
            os.makedirs(sft_checkpoint_dir, exist_ok=True)
            os.makedirs(sft_tensorboard_dir, exist_ok=True)
            os.makedirs(dataset_root, exist_ok=True)
            rows = [{"input": f"Question: {idx} + {idx}? Answer:", "output": str(idx + idx)} for idx in range(32)]
            with open(os.path.join(dataset_root, "training.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        torch.distributed.barrier()

        pretrain_cfg = qwen3_600m_pretrain_1gpu_h100_bf16_config()
        _make_functional_test_model_small(pretrain_cfg.model)
        pretrain_cfg.model.tensor_model_parallel_size = 1
        pretrain_cfg.model.pipeline_model_parallel_size = 1
        pretrain_cfg.model.context_parallel_size = 2
        pretrain_cfg.dataset.seq_length = 256
        pretrain_cfg.train.train_iters = 1
        pretrain_cfg.train.global_batch_size = 2
        pretrain_cfg.train.micro_batch_size = 1
        pretrain_cfg.validation.eval_interval = 1
        pretrain_cfg.validation.eval_iters = 0
        pretrain_cfg.scheduler.lr_warmup_iters = 0
        pretrain_cfg.logger.log_interval = 1
        pretrain_cfg.logger.tensorboard_dir = pretrain_tensorboard_dir
        pretrain_cfg.checkpoint.save_interval = pretrain_cfg.train.train_iters
        pretrain_cfg.checkpoint.save = pretrain_checkpoint_dir
        pretrain_cfg.checkpoint.load = None

        cfg = qwen3_600m_sft_1gpu_h100_bf16_config()
        _make_functional_test_model_small(cfg.model)
        cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
        cfg.tokenizer.tokenizer_model = "gpt2"
        cfg.model.calculate_per_token_loss = True
        cfg.ddp.average_in_collective = False
        cfg.ddp.grad_reduce_in_fp32 = False
        cfg.ddp.use_distributed_optimizer = False
        cfg.optimizer.use_distributed_optimizer = False

        # Keep the world-size math simple: tp=1, pp=1, cp=2 -> dp derived from env.
        cfg.model.tensor_model_parallel_size = 1
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.context_parallel_size = 2

        # Small, fast run
        cfg.train.train_iters = 2
        cfg.train.global_batch_size = 2
        cfg.train.micro_batch_size = 1
        cfg.validation.eval_interval = 1
        cfg.validation.eval_iters = 0
        cfg.scheduler.lr_warmup_iters = 0
        cfg.logger.log_interval = 1
        cfg.logger.tensorboard_dir = sft_tensorboard_dir

        # Use a small packed local SFT dataset to exercise THD/context-parallel slicing
        cfg.dataset = GPTSFTDatasetConfig(
            dataset_root=dataset_root,
            seq_length=256,
            dataloader_type="batch",
            num_workers=1,
            do_validation=False,
            do_test=False,
            dataset_kwargs={"pad_to_max_length": True},
            max_train_samples=16,
            enable_offline_packing=True,
            offline_packing_specs=PackedSequenceSpecs(
                packed_sequence_size=512,
                tokenizer_model_name="gpt2",
                num_tokenizer_workers=1,
                pad_seq_to_mult=cfg.model.context_parallel_size * 2,
            ),
        )

        cfg.checkpoint.save_interval = cfg.train.train_iters
        cfg.checkpoint.save = sft_checkpoint_dir
        cfg.checkpoint.load = None
        cfg.checkpoint.pretrained_checkpoint = pretrain_checkpoint_dir

        try:
            pretrain(pretrain_cfg, forward_step)
            verify_checkpoint_files(
                pretrain_checkpoint_dir,
                pretrain_cfg.train.train_iters,
                ckpt_format=pretrain_cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=pretrain_cfg.checkpoint.storage_writers_per_rank,
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.barrier()

            finetune(cfg, forward_step)
            verify_checkpoint_files(
                sft_checkpoint_dir,
                cfg.train.train_iters,
                ckpt_format=cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg.checkpoint.storage_writers_per_rank,
            )
        finally:
            clear_directories(shared_dir)
