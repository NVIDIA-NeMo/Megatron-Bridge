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

"""
Functional smoke test for examples/inference/vlm/vlm_inference.py (Megatron VLM checkpoint).

Checkpoint path follows tests/functional_tests/training/test_load_model.py
(/home/TestData/megatron_bridge/checkpoints/...). Coverage invocation follows
tests/functional_tests/converter/test_checkpoint_conversion.py and
test_generate_vlm_from_hf.py (torch.distributed.run + coverage run).
"""

import subprocess
from pathlib import Path

import pytest


class TestVLMInferenceScript:
    """Megatron VLM inference via vlm_inference.py under coverage."""

    @pytest.mark.run_only_on("GPU")
    def test_vlm_inference_megatron_checkpoint(self):
        ckpt_path = "/home/TestData/megatron_bridge/checkpoints/qwen25-vl-3b"
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/inference/vlm/vlm_inference.py",
            "--megatron_model_path",
            ckpt_path,
            "--prompt",
            "Say hi in one short phrase.",
            "--max_new_tokens",
            "5",
            "--top_k",
            "1",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent,
                timeout=3600,
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                pytest.fail(f"vlm_inference.py failed with return code {result.returncode}")

            assert "GENERATED TEXT OUTPUT" in result.stdout, (
                f"Generation output header not found. Output: {result.stdout}"
            )
            assert "Prompt:" in result.stdout, result.stdout
            assert "Generated:" in result.stdout, result.stdout

        except Exception as e:
            print(f"Error during VLM inference test: {e}")
            raise
