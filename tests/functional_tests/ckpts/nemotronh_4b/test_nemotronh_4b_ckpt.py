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

"""Functional smoke tests for LLaMA recipe configurations."""

import os
import pytest
import shutil
import sys

from torch.distributed.run import main as torchrun_main

from megatron.bridge.recipes.nemotronh import nemotronh_4b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


BASE_DIR = "/workspace/test_ckpts/nemotronh_4b"
MBRIDGE_CKPT = f"{BASE_DIR}/mbridge"
MCORE_CKPT = f"{BASE_DIR}/mcore"
TB_DIR = f"{BASE_DIR}/tb"


class TestNemotronhCkpt:
    """Test class for Nempotron Hybrid checkpoint functional tests."""

    @pytest.mark.run_only_on("GPU")
    def test_nemotronh_4b_ckpt_mbridge(self):
        """Functional test for Nemotron Hybrid MBridge checkpoint."""

        config = nemotronh_4b_pretrain_config()

        config.model.num_layers = 26
        config.model.hybrid_override_pattern = "M-M-M-M*-M-M-M-M*-M-M-M-M*"

        config.train.train_iters = 5
        config.train.eval_iters = 5
        config.train.save_interval = 5
        config.train.global_batch_size = 4
        config.train.micro_batch_size = 1

        config.scheduler.lr_warmup_iters = 2

        config.logger.log_interval = 1

        config.checkpoint.save = MBRIDGE_CKPT

        pretrain(config=config, forward_step_func=forward_step)
    
    @pytest.mark.run_only_on("GPU")
    def test_nemotronh_4b_ckpt_mcore(self, monkeypatch):
        """Functional test for Nemotron Hybrid MCore checkpoint."""

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
                "/opt/Megatron-Bridge/3rdparty/Megatron-LM/pretrain_mamba.py",
                "--init-method-std", "0.014",
                "--disable-bias-linear",
                "--use-rope-scaling",
                "--squared-relu",
                "--qk-layernorm",
                "--rotary-percent", "1.0",
                "--rotary-base", "1000000",
                "--use-rotary-position-embeddings",
                "--hybrid-override-pattern", "M-M-M-M*-M-M-M-M*-M-M-M-M*",
                "--spec", "megatron.core.models.mamba.mamba_layer_specs", "mamba_stack_spec",
                "--num-layers", "26",
                "--hidden-size", "3072",
                "--num-attention-heads", "32",
                "--mamba-num-heads", "112",
                "--ffn-hidden-size", "12288",
                "--kv-channels", "128",
                "--group-query-attention",
                "--position-embedding-type", "none",
                "--attention-backend", "fused",
                "--num-query-groups", "8",
                "--normalization", "RMSNorm",
                "--attention-dropout", "0.0",
                "--hidden-dropout", "0.0",
                "--tensor-model-parallel-size", "2",
                "--pipeline-model-parallel-size", "1",
                "--seq-length", "8192",
                "--max-position-embeddings", "8192",
                "--micro-batch-size", "1",
                "--global-batch-size", "4",
                "--train-iters", "10",
                "--mock-data",
                "--tokenizer-type", "NullTokenizer",
                "--vocab-size", "151936",
                "--save-interval", "5",
                "--eval-interval", "5",
                "--eval-iters", "4",
                "--load", MBRIDGE_CKPT,
                "--save", MCORE_CKPT,
                "--ckpt-format", "torch_dist",
                "--log-progress",
                "--bf16",
                "--lr", "4.5e-4",
                "--min-lr", "4.5e-5",
                "--num-workers", "2",
                "--tensorboard-dir", TB_DIR,
                "--log-interval", "1",
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
