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

import pytest
import subprocess

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


class TestLlama3MBridgeCkpt:
    """Test class for LLaMA recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    def test_llama32_1B_ckpt_mbridge(self):
        """Functional test for LLaMA recipes with appropriate parallelism configurations."""

        config = llama32_1b_pretrain_config()

        config.model.seq_length = 8192

        config.train.train_iters = 5
        config.train.eval_iters = 5
        config.train.save_interval = 5
        config.train.global_batch_size = 8
        config.train.micro_batch_size = 1

        config.scheduler.lr_warmup_iters = 2

        config.logger.log_interval = 1

        config.checkpoint.save = "/workspace/test_ckpts/llama32_1b_mbridge"

        pretrain(config=config, forward_step_func=forward_step)

    @pytest.mark.run_only_on("GPU")
    def test_llama32_1B_ckpt_mcore(self):
        """Functional test for LLaMA recipes with appropriate parallelism configurations."""

        script_path = "test_llama3_1b_mcore.sh"
        process = subprocess.Popen(
            ["bash", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
