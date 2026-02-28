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

"""Functional smoke tests for Mcore FLUX pretrain mock runs."""

import os
import subprocess

import pytest


class TestMcoreFluxPretrain:
    """Test class for Mcore FLUX pretrain functional tests."""

    @pytest.mark.run_only_on("GPU")
    def test_flux_pretrain_mock(self, tmp_path):
        """
        Functional test for FLUX pretrain recipe with mock data.

        This test verifies that the FLUX pretrain recipe can run successfully
        in mock mode with minimal configuration, ensuring:
        1. The distributed training can start without errors
        2. Model initialization works correctly
        3. Forward/backward passes complete successfully
        4. The training loop executes without crashes
        """
        # Set up temporary directories for dataset and checkpoints
        dataset_path = os.path.join(tmp_path, "mock_dataset")
        checkpoint_dir = os.path.join(tmp_path, "checkpoints")
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build the command for the mock run
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/diffusion/recipes/flux/pretrain_flux.py",
            "--mock",
            "--timestep-sampling",
            "logit_normal",
            "--scheduler-steps",
            "1000",
            "model.tensor_model_parallel_size=1",
            "model.pipeline_model_parallel_size=1",
            "model.context_parallel_size=1",
            "model.num_joint_layers=1",
            "model.num_single_layers=2",
            "model.hidden_size=1024",
            "model.num_attention_heads=8",
            "model.ffn_hidden_size=4096",
            "model.in_channels=64",
            "model.context_dim=4096",
            "model.guidance_embed=false",
            f"checkpoint.save={checkpoint_dir}",
            f"checkpoint.load={checkpoint_dir}",
            "checkpoint.save_interval=200",
            "optimizer.lr=1e-4",
            "train.eval_iters=0",
            "train.train_iters=10",
            "train.global_batch_size=2",
            "train.micro_batch_size=1",
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
            pytest.fail("FLUX pretrain mock run exceeded timeout of 1800 seconds (30 minutes)")
        except subprocess.CalledProcessError as e:
            result = e
            pytest.fail(f"FLUX pretrain mock run failed with return code {e.returncode}")
        finally:
            # Always print output for debugging
            if result is not None:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
