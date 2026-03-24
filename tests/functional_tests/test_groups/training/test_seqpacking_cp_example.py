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

import os

import pytest
import torch

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.squad import process_squad_example
from megatron.bridge.recipes.llama.llama3 import llama32_1b_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


class TestPeftSftExample:
    """Run the PEFT SFT example as a functional test with packed sequences + CP."""

    @pytest.mark.run_only_on("GPU")
    def test_sft_example_runs_with_cp_and_packing(self, tmp_path):
        pytest.importorskip("transformer_engine_torch")
        initialize_distributed()

        if torch.distributed.get_world_size() < 2:
            pytest.skip("requires >=2 GPUs for context_parallel_size=2")

        shared_dir = broadcast_path(tmp_path)
        checkpoint_dir = os.path.join(shared_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)
        torch.distributed.barrier()

        cfg = llama32_1b_sft_config()
        cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
        cfg.tokenizer.tokenizer_model = "meta-llama/Llama-3.2-1B"
        cfg.model.calculate_per_token_loss = True
        cfg.ddp.average_in_collective = False

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
        cfg.logger.tensorboard_dir = tensorboard_dir

        # Use a small packed SQuAD dataset to exercise THD/context-parallel slicing
        cfg.dataset = HFDatasetConfig(
            dataset_name="squad",
            process_example_fn=process_squad_example,
            seq_length=256,
            dataloader_type="batch",
            num_workers=1,
            do_validation=False,
            do_test=False,
            val_proportion=None,
            dataset_kwargs={"pad_to_max_length": True},
            max_train_samples=16,
            packed_sequence_specs=PackedSequenceSpecs(
                packed_sequence_size=512,
                tokenizer_model_name="meta-llama/Llama-3.2-1B",
                pad_seq_to_mult=cfg.model.context_parallel_size * 2,
            ),
            rewrite=False,
        )

        cfg.model.seq_length = 256
        cfg.checkpoint.save_interval = cfg.train.train_iters
        cfg.checkpoint.save = checkpoint_dir
        cfg.checkpoint.pretrained_checkpoint = None

        try:
            finetune(cfg, forward_step)
            verify_checkpoint_files(
                checkpoint_dir,
                cfg.train.train_iters,
                ckpt_format=cfg.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg.checkpoint.storage_writers_per_rank,
            )
        finally:
            clear_directories(shared_dir)
