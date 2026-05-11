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

"""Functional smoke tests for Mcore WAN pretrain mock runs.

Uses the generic run_recipe.py entry point with wan_1_3b_pretrain_config and wan_step.
Mock/synthetic data is used when dataset.path is not set (no --mock flag).
"""

import os
import subprocess

import pytest


class TestMcoreWanPretrain:
    """Test class for Mcore WAN pretrain functional tests."""

    @pytest.mark.run_only_on("GPU")
    def test_wan_pretrain_mock(self, tmp_path):
        """
        Functional test for WAN pretrain recipe with mock data.

        This test verifies that the WAN pretrain recipe can run successfully
        via run_recipe.py with minimal configuration (mock data when no dataset.path), ensuring:
        1. The distributed training can start without errors
        2. Model initialization works correctly
        3. Forward/backward passes complete successfully
        4. The training loop executes without crashes
        """
        checkpoint_dir = os.path.join(tmp_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the command for the mock run
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "scripts/training/run_recipe.py",
            "--recipe",
            "wan_1_3b_pretrain_config",
            "--step_func",
            "wan_step",
            "model.tensor_model_parallel_size=1",
            "model.pipeline_model_parallel_size=1",
            "model.context_parallel_size=2",
            "model.crossattn_emb_size=1536",
            "model.hidden_size=1536",
            "model.ffn_hidden_size=8960",
            "model.num_attention_heads=12",
            "model.num_layers=3",
            "model.qkv_format=thd",
            f"checkpoint.save={checkpoint_dir}",
            f"checkpoint.load={checkpoint_dir}",
            "checkpoint.load_optim=false",
            "checkpoint.save_interval=200",
            "optimizer.lr=5e-6",
            "optimizer.min_lr=5e-6",
            "train.eval_iters=0",
            "train.train_iters=10",
            "scheduler.lr_decay_style=constant",
            "scheduler.lr_warmup_iters=0",
            "model.seq_length=2048",
            "dataset.seq_length=2048",
            "train.global_batch_size=2",
            "train.micro_batch_size=1",
            "dataset.global_batch_size=2",
            "dataset.micro_batch_size=1",
            "logger.log_interval=1",
        ]

        # Run the command with a timeout
        result = None
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                check=True,
            )

            # Basic verification that the run completed
            assert result.returncode == 0, f"Command failed with return code {result.returncode}"

        except subprocess.TimeoutExpired:
            pytest.fail("WAN pretrain mock run exceeded timeout of 1800 seconds (30 minutes)")
        except subprocess.CalledProcessError as e:
            result = e
            pytest.fail(f"WAN pretrain mock run failed with return code {e.returncode}. Check STDOUT/STDERR above.")
        finally:
            # Always print output for debugging
            if result is not None:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
