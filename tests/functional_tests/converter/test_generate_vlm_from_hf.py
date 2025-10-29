"""
Functional test: Vision-Language generation from a HuggingFace model using the
examples/conversion/hf_to_megatron_generate_vlm.py entry point.
"""

import subprocess
from pathlib import Path

import pytest


class TestGenerateVLMFromHF:
    @pytest.mark.run_only_on("GPU")
    def test_generate_vlm_single_gpu(self):
        """
        Run 1-process distributed VLM generation on a small instruct model with an image URL.
        """
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--nnodes=2",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_to_megatron_generate_vlm.py",
            "--hf_model_path",
            "Qwen/Qwen2.5-3B-Instruct",
            "--image_path",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "--prompt",
            "Describe this image.",
            "--tp",
            "2"
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
