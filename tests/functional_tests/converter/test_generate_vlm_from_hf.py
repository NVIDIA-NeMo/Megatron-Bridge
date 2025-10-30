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

"""
Functional test: Vision-Language generation from a HuggingFace model using the
examples/conversion/hf_to_megatron_generate_vlm.py entry point.
"""

import subprocess
from pathlib import Path

import pytest


class TestGenerateVLMFromHF:
    @pytest.mark.run_only_on("GPU")
    def test_generate_vlm(self):
        """
        Run distributed VLM generation on a small instruct model with an image URL.
        """
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
            "examples/conversion/hf_to_megatron_generate_vlm.py",
            "--hf_model_path",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "--image_path",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "--prompt",
            "Describe this image.",
            "--tp",
            "2",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        if result.returncode != 0:
            print("STDOUT:\n" + result.stdout)
            print("STDERR:\n" + result.stderr)
            assert False, f"VLM generation failed with return code {result.returncode}"

        # Basic sanity checks on output
        assert "GENERATED TEXT OUTPUT" in result.stdout, f"Generation output header not found. Output: {result.stdout}"
        assert "Prompt: Describe this image." in result.stdout, (
            f"Prompt line not found in output. Output: {result.stdout}"
        )
        assert "Generated:" in result.stdout, f"Generated text line not found in output. Output: {result.stdout}"
