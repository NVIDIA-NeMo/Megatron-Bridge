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

"""Functional smoke tests for Qwen checkpointing."""

import os
import shutil
import sys

import pytest
from torch.distributed.run import main as torchrun_main

from megatron.bridge.recipes.qwen import qwen3_4b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


BASE_DIR = "/workspace/test_ckpts/qwen3_4b"
MBRIDGE_CKPT = f"{BASE_DIR}/mbridge"
MCORE_CKPT = f"{BASE_DIR}/mcore"
TB_DIR = f"{BASE_DIR}/tb"


class TestQwen3Ckpt:
    """Test class for Qwen checkpoint functional tests."""

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_4b_ckpt_mbridge(self):
        """Functional test for Qwen MBridge checkpoint."""

        config = qwen3_4b_pretrain_config()

        config.checkpoint.save = MBRIDGE_CKPT
        config.checkpoint.load = MCORE_CKPT if os.path.exists(MCORE_CKPT) else None
        config.checkpoint.load_optim = False

        config.train.train_iters = 10 if config.checkpoint.load else 5
        config.train.eval_iters = 5
        config.train.save_interval = 5
        config.train.global_batch_size = 4
        config.train.micro_batch_size = 1

        config.scheduler.lr_warmup_iters = 2

        config.logger.log_interval = 1

        pretrain(config=config, forward_step_func=forward_step)

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_4b_ckpt_mcore(self, monkeypatch):
        """Functional test for Qwen MCore checkpoint."""

        load_dir = MBRIDGE_CKPT if os.path.exists(MBRIDGE_CKPT) else None
        train_iters = 10 if load_dir else 5

        # Set environment variables
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")

        # Set MLM script
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "torchrun",
                "--nproc_per_node=2",
                "/opt/Megatron-Bridge/3rdparty/Megatron-LM/pretrain_gpt.py",
                "--init-method-std",
                "0.014",
                "--disable-bias-linear",
                "--use-rope-scaling",
                "--swiglu",
                "--qk-layernorm",
                "--rotary-percent",
                "1.0",
                "--rotary-base",
                "1000000",
                "--use-rotary-position-embeddings",
                "--num-layers",
                "36",
                "--hidden-size",
                "2560",
                "--num-attention-heads",
                "32",
                "--ffn-hidden-size",
                "9728",
                "--kv-channels",
                "128",
                "--group-query-attention",
                "--position-embedding-type",
                "rope",
                "--attention-backend",
                "fused",
                "--num-query-groups",
                "8",
                "--normalization",
                "RMSNorm",
                "--attention-dropout",
                "0.0",
                "--hidden-dropout",
                "0.0",
                "--tensor-model-parallel-size",
                "2",
                "--pipeline-model-parallel-size",
                "1",
                "--seq-length",
                "4096",
                "--max-position-embeddings",
                "4096",
                "--micro-batch-size",
                "1",
                "--global-batch-size",
                "4",
                "--mock-data",
                "--tokenizer-type",
                "NullTokenizer",
                "--vocab-size",
                "151936",
                "--train-iters",
                f"{train_iters}",
                "--save-interval",
                "5",
                "--eval-interval",
                "5",
                "--eval-iters",
                "4",
                "--load",
                load_dir,
                "--save",
                MCORE_CKPT,
                "--ckpt-format",
                "torch_dist",
                "--log-progress",
                "--bf16",
                "--lr",
                "4.5e-4",
                "--min-lr",
                "4.5e-5",
                "--num-workers",
                "2",
                "--tensorboard-dir",
                "/workspace/tb",
                "--log-interval",
                "1",
                "--log-throughput",
                "--no-load-optim",
                "--no-load-rng",
            ],
        )

        # Run MLM script
        torchrun_main()

    def test_remove_artifacts(self):
        """Removes model artifacts"""
        shutil.rmtree(BASE_DIR)

        assert not os.path.exists(BASE_DIR)
