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

"""Functional tests for QAT (Quantization Aware Training) workflow."""

import os
import subprocess
from pathlib import Path

import pytest

from megatron.bridge.training.utils.checkpoint_utils import (
    TRACKER_PREFIX,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_checkpoint_train_state_filename,
)
from tests.functional_tests.utils import clear_directories


# QAT workflow test configurations
# (recipe_name, parallelism_overrides)
QAT_WORKFLOW_CONFIGS = [
    ("llama32_1b", {}),  # Small model, use recipe defaults
]


class TestQATWorkflow:
    """
    Test complete QAT workflow: first quantize HuggingFace models using PTQ,
    then run pre-training from the quantized checkpoint.
    """

    def _run_quantization(self, base_dir, quant_cfg="fp8", tp=1, pp=1):
        """
        Helper method to run PTQ quantization step.

        Args:
            base_dir: Base directory to save the quantized checkpoint
            quant_cfg: Quantization configuration to use
            tp: Tensor parallelism size
            pp: Pipeline parallelism size

        Returns:
            tuple: (subprocess.CompletedProcess, actual_output_path)
        """
        # Create descriptive checkpoint name including configuration
        checkpoint_name = f"llama32_quantized_{quant_cfg}_tp{tp}_pp{pp}"
        output_dir = base_dir / checkpoint_name
        output_dir.mkdir(exist_ok=True)
        # Calculate total number of processes needed
        total_procs = max(tp * pp, 1)

        # Use venv python if available, otherwise system python
        import sys

        python_executable = sys.executable

        # Base command for PTQ quantization
        cmd = [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={total_procs}",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/quantization/quantize.py",
            "--hf-model-id",
            "meta-llama/Llama-3.2-1B",
            "--export-quant-cfg",
            quant_cfg,
            "--megatron-save-path",
            str(output_dir),
            "--disable-hf-datasets-file-lock",
        ]

        # Add parallelism arguments only if > 1
        if tp > 1:
            cmd.extend(["--tp", str(tp)])
        if pp > 1:
            cmd.extend(["--pp", str(pp)])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)
        return result, output_dir

    def _run_pretrain_from_quantized_checkpoint(
        self,
        quantized_checkpoint_path: str,
        checkpoint_save_dir: str,
        hf_model_id: str = "meta-llama/Llama-3.2-1B",
        tp: int = 1,
        pp: int = 1,
        cp: int = 2,
        train_iters: int = 10,
        save_interval: int = 10,
    ):
        """
        Run pre-training from a quantized checkpoint using subprocess.

        Args:
            quantized_checkpoint_path: Path to the quantized checkpoint
            checkpoint_save_dir: Directory to save checkpoints during training
            hf_model_id: HuggingFace model ID
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            cp: Context parallelism size (default: 2)
            train_iters: Number of training iterations
            save_interval: Interval for saving checkpoints

        Returns:
            tuple: (subprocess.CompletedProcess, final_iteration)
                   where final_iteration is the last checkpoint saved
        """
        # Calculate total number of processes needed (tp * pp * cp)
        total_procs = tp * pp * cp

        # Use venv python if available, otherwise system python
        import sys

        python_executable = sys.executable

        # Calculate the final iteration (last checkpoint that will be saved)
        # Checkpoints are saved at intervals, so the last one is at train_iters if it's a multiple of save_interval
        final_iteration = (train_iters // save_interval) * save_interval

        # Base command for pre-training from quantized checkpoint
        cmd = [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={total_procs}",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/quantization/pretrain_quantized_llama3_8b.py",
            "--hf-path",
            hf_model_id,
            "model.gradient_accumulation_fusion=False",
            f"checkpoint.pretrained_checkpoint={quantized_checkpoint_path}",
            f"checkpoint.save={checkpoint_save_dir}",
            f"checkpoint.save_interval={save_interval}",
            f"train.train_iters={train_iters}",
            "train.eval_interval=5",
            "train.eval_iters=2",
            "train.global_batch_size=8",
            "scheduler.lr_warmup_iters=2",
            f"scheduler.lr_decay_iters={train_iters}",
        ]

        # Always add parallelism arguments to override script defaults
        cmd.append(f"model.tensor_model_parallel_size={tp}")
        cmd.append(f"model.pipeline_model_parallel_size={pp}")
        cmd.append(f"model.context_parallel_size={cp}")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)
        return result, final_iteration

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("recipe_name,parallelism_overrides", QAT_WORKFLOW_CONFIGS)
    def test_qat_workflow(self, recipe_name, parallelism_overrides, tmp_path):
        """
        Test complete QAT workflow:
        1. Run PTQ quantization to create quantized checkpoint
        2. Run pre-training from the quantized checkpoint

        Args:
            recipe_name: Name of the recipe for logging/debugging
            parallelism_overrides: Dict with tensor/pipeline/context parallelism settings
            tmp_path: Pytest temporary path fixture
        """
        # Extract parallelism settings (None = use defaults)
        tensor_model_parallel_size = parallelism_overrides.get("tensor_model_parallel_size")
        pipeline_model_parallel_size = parallelism_overrides.get("pipeline_model_parallel_size")
        context_parallel_size = parallelism_overrides.get("context_parallel_size")

        quant_base_dir = tmp_path / "quantization"
        quant_base_dir.mkdir(exist_ok=True)

        checkpoint_save_dir = tmp_path / "checkpoints"
        checkpoint_save_dir.mkdir(exist_ok=True)

        try:
            print(f"=== STEP 1: Running PTQ quantization for {recipe_name} ===")
            # Step 1: Run PTQ quantization (use defaults if None)
            quantize_result, quantized_checkpoint_dir = self._run_quantization(
                quant_base_dir,
                quant_cfg="fp8",
                tp=tensor_model_parallel_size or 1,
                pp=pipeline_model_parallel_size or 1,
            )

            if quantize_result.returncode != 0:
                print(f"Quantization STDOUT: {quantize_result.stdout}")
                print(f"Quantization STDERR: {quantize_result.stderr}")
                assert False, f"PTQ quantization step failed with return code {quantize_result.returncode}"

            # Verify quantization succeeded
            assert "Quantizing the model with fp8 configuration" in quantize_result.stdout, (
                f"Quantization start message not found. Output: {quantize_result.stdout}"
            )
            assert quantized_checkpoint_dir.exists(), (
                f"Quantized checkpoint directory not found at {quantized_checkpoint_dir}"
            )

            checkpoint_contents = list(quantized_checkpoint_dir.iterdir())
            assert len(checkpoint_contents) > 0, f"Quantized checkpoint directory is empty: {quantized_checkpoint_dir}"

            print("✓ PTQ quantization completed successfully")
            print(f"  Checkpoint saved at: {quantized_checkpoint_dir}")
            print(f"  Checkpoint contents: {[item.name for item in checkpoint_contents]}")

            print(f"=== STEP 2: Running pre-training from quantized checkpoint for {recipe_name} ===")
            # Step 2: Run pre-training from the quantized checkpoint
            train_iters = 10
            save_interval = 10
            pretrain_result, expected_iteration = self._run_pretrain_from_quantized_checkpoint(
                quantized_checkpoint_path=str(quantized_checkpoint_dir),
                checkpoint_save_dir=str(checkpoint_save_dir),
                hf_model_id="meta-llama/Llama-3.2-1B",
                tp=tensor_model_parallel_size or 1,
                pp=pipeline_model_parallel_size or 1,
                cp=context_parallel_size or 2,  # Default context parallelism is 2
                train_iters=train_iters,
                save_interval=save_interval,
            )

            if pretrain_result.returncode != 0:
                print(f"Pre-training STDOUT: {pretrain_result.stdout}")
                print(f"Pre-training STDERR: {pretrain_result.stderr}")
                assert False, f"Pre-training step failed with return code {pretrain_result.returncode}"

            print("✓ Pre-training from quantized checkpoint completed successfully")
            print(f"  Training ran for {train_iters} iterations, saving every {save_interval} iterations")
            print(f"  Expected final checkpoint iteration: {expected_iteration}")

            # Verify checkpoint files were created with comprehensive checks
            # (adapted from verify_checkpoint_files but without requiring torch.distributed)
            assert checkpoint_save_dir.exists(), f"Checkpoint save directory not found at {checkpoint_save_dir}"

            # Verify Megatron-Bridge tracker file
            latest_tracker_file = get_checkpoint_train_state_filename(str(checkpoint_save_dir), prefix=TRACKER_PREFIX)
            assert os.path.exists(latest_tracker_file), (
                f"Latest checkpoint tracker file not found at {latest_tracker_file}"
            )
            print(f"✓ Megatron-Bridge tracker file found: {latest_tracker_file}")

            # Verify Megatron-LM compatibility tracker file
            megatron_lm_tracker = get_checkpoint_tracker_filename(str(checkpoint_save_dir))
            assert os.path.exists(megatron_lm_tracker), f"Megatron-LM tracker file not found at {megatron_lm_tracker}"
            print(f"✓ Megatron-LM tracker file found: {megatron_lm_tracker}")

            # Verify the tracker file contains the correct iteration
            with open(megatron_lm_tracker, "r") as f:
                saved_iteration = f.read().strip()
            assert saved_iteration == str(expected_iteration), (
                f"Megatron-LM tracker file contains '{saved_iteration}', expected '{expected_iteration}'"
            )
            print(f"✓ Tracker file contains correct iteration: {expected_iteration}")

            # Verify final checkpoint directory exists
            final_iter_dir = get_checkpoint_name(str(checkpoint_save_dir), expected_iteration, release=False)
            assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"
            print(f"✓ Final checkpoint directory found: {final_iter_dir}")

            # Verify metadata file exists
            metadata_file = os.path.join(final_iter_dir, ".metadata")
            assert os.path.exists(metadata_file), f"Checkpoint metadata file not found at {metadata_file}"
            print(f"✓ Metadata file found: {metadata_file}")

            # Verify .distcp files (torch.distributed.checkpoint format)
            distcp_files = [f for f in os.listdir(final_iter_dir) if f.endswith(".distcp")]

            # Calculate expected world size from parallelism settings
            tp = tensor_model_parallel_size or 1
            pp = pipeline_model_parallel_size or 1
            cp = context_parallel_size or 2
            world_size = tp * pp * cp

            # For torch_dist format, expect 2 * world_size .distcp files
            # (one for model state, one for optimizer state per rank)
            expected_distcp_files = 2 * world_size
            assert len(distcp_files) == expected_distcp_files, (
                f"Expected {expected_distcp_files} .distcp files (2 * {world_size} world_size), "
                f"found {len(distcp_files)}: {distcp_files}"
            )
            print(
                f"✓ Correct number of .distcp files: {len(distcp_files)} "
                f"(world_size={world_size}, tp={tp}, pp={pp}, cp={cp})"
            )

            print(f"SUCCESS: Complete QAT workflow test passed for {recipe_name}")

        except Exception as e:
            print(f"Error during QAT workflow test for {recipe_name}: {e}")
            raise
        finally:
            # Clean up all test directories
            clear_directories(quant_base_dir)
            clear_directories(checkpoint_save_dir)
