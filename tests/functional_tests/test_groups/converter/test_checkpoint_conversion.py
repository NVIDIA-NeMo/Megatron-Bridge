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

import subprocess
from pathlib import Path

import pytest


class TestCheckpointConversion:
    """
    Test checkpoint conversion between HuggingFace and Megatron formats.
    """

    @pytest.mark.run_only_on("GPU")
    def test_import_hf_to_megatron(self, tmp_path):
        """
        Test importing a HuggingFace model to Megatron checkpoint format.

        Args:
            tmp_path: Pytest temporary path fixture
        """
        # Create temporary output directory for Megatron checkpoint
        megatron_output_dir = tmp_path / "megatron_checkpoint"
        megatron_output_dir.mkdir(exist_ok=True)
        # Run the supported CPU conversion worker under coverage.
        cmd = [
            "python",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "scripts/conversion/run_conversion.py",
            "import",
            "--device",
            "cpu",
            "--hf-model",
            "meta-llama/Llama-3.2-1B",
            "--megatron-path",
            str(megatron_output_dir),
            "--torch-dtype",
            "bfloat16",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            # Check that the import completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Import failed with return code {result.returncode}"

            # Verify that the import succeeded based on output messages
            output = result.stdout + result.stderr
            assert "CPU import complete:" in output, f"Import success message not found in output. Output: {output}"

            # Verify that the checkpoint directory structure was created
            assert megatron_output_dir.exists(), f"Megatron checkpoint directory not found at {megatron_output_dir}"

            # Check for expected checkpoint files/directories
            checkpoint_contents = list(megatron_output_dir.iterdir())
            assert len(checkpoint_contents) > 0, f"Megatron checkpoint directory is empty: {megatron_output_dir}"

            print("SUCCESS: HF to Megatron import test completed successfully")
            print(f"Megatron checkpoint saved at: {megatron_output_dir}")
            print(f"Checkpoint contents: {[item.name for item in checkpoint_contents]}")

        except Exception as e:
            print(f"Error during HF to Megatron import test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    def test_convert_sh_cpu_roundtrip_preserves_weights(self, tmp_path):
        """Test the public launcher with a bit-exact HF-to-Megatron-to-HF roundtrip."""
        import torch
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        repo_root = Path(__file__).resolve().parents[4]
        hf_source_dir = tmp_path / "hf_source"
        megatron_checkpoint_dir = tmp_path / "megatron_checkpoint"
        hf_export_dir = tmp_path / "hf_export"
        torch.manual_seed(1234)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                bos_token_id=2,
                eos_token_id=3,
                pad_token_id=1,
            )
        ).eval()
        model.save_pretrained(hf_source_dir, safe_serialization=True)
        vocab = {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3}
        vocab.update({f"token_{index}": index for index in range(4, 128)})
        tokenizer_model = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_model.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_model,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )
        tokenizer.save_pretrained(hf_source_dir)

        import_result = subprocess.run(
            [
                str(repo_root / "scripts/conversion/convert.sh"),
                "import",
                "--executor",
                "local",
                "--device",
                "cpu",
                "--hf-model",
                str(hf_source_dir),
                "--megatron-path",
                str(megatron_checkpoint_dir),
                "--torch-dtype",
                "float32",
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert import_result.returncode == 0, (
            f"convert.sh import failed with return code {import_result.returncode}\n"
            f"STDOUT:\n{import_result.stdout}\nSTDERR:\n{import_result.stderr}"
        )

        export_result = subprocess.run(
            [
                str(repo_root / "scripts/conversion/convert.sh"),
                "export",
                "--executor",
                "local",
                "--device",
                "cpu",
                "--hf-model",
                str(hf_source_dir),
                "--megatron-path",
                str(megatron_checkpoint_dir),
                "--hf-path",
                str(hf_export_dir),
                "--no-progress",
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert export_result.returncode == 0, (
            f"convert.sh export failed with return code {export_result.returncode}\n"
            f"STDOUT:\n{export_result.stdout}\nSTDERR:\n{export_result.stderr}"
        )

        covered_export_dir = tmp_path / "hf_export_covered"
        covered_export_result = subprocess.run(
            [
                "python",
                "-m",
                "coverage",
                "run",
                "--data-file=/opt/Megatron-Bridge/.coverage",
                "--source=/opt/Megatron-Bridge/",
                "--parallel-mode",
                "scripts/conversion/run_conversion.py",
                "export",
                "--device",
                "cpu",
                "--hf-model",
                str(hf_source_dir),
                "--megatron-path",
                str(megatron_checkpoint_dir),
                "--hf-path",
                str(covered_export_dir),
                "--no-progress",
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert covered_export_result.returncode == 0, (
            f"Covered export failed with return code {covered_export_result.returncode}\n"
            f"STDOUT:\n{covered_export_result.stdout}\nSTDERR:\n{covered_export_result.stderr}"
        )

        original_weights = load_file(hf_source_dir / "model.safetensors")
        exported_weights = load_file(hf_export_dir / "model.safetensors")
        assert original_weights.keys() == exported_weights.keys()
        for name, original_weight in original_weights.items():
            exported_weight = exported_weights[name]
            assert original_weight.dtype == exported_weight.dtype, f"Roundtrip dtype mismatch: {name}"
            assert original_weight.shape == exported_weight.shape, f"Roundtrip shape mismatch: {name}"
            assert torch.equal(original_weight, exported_weight), f"Roundtrip value mismatch: {name}"

        exported_model = LlamaForCausalLM.from_pretrained(hf_export_dir).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            original_logits = model(input_ids).logits
            exported_logits = exported_model(input_ids).logits
        assert torch.equal(original_logits, exported_logits), "Roundtrip forward-pass mismatch"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "torch_dtype,test_name",
        [
            ("float16", "float16_cpu"),
            ("bfloat16", "bfloat16_cpu"),
        ],
    )
    def test_import_with_different_settings(self, tmp_path, torch_dtype, test_name):
        """
        Test importing with different torch_dtype and device_map settings.

        Args:
            tmp_path: Pytest temporary path fixture
            torch_dtype: Model precision to test
            test_name: Name of the test for identification
        """
        # Create temporary output directory
        megatron_output_dir = tmp_path / f"megatron_checkpoint_{test_name}"
        megatron_output_dir.mkdir(exist_ok=True)

        try:
            # Build command with different settings
            cmd = [
                "python",
                "-m",
                "coverage",
                "run",
                "--data-file=/opt/Megatron-Bridge/.coverage",
                "--source=/opt/Megatron-Bridge/",
                "--parallel-mode",
                "scripts/conversion/run_conversion.py",
                "import",
                "--device",
                "cpu",
                "--hf-model",
                "meta-llama/Llama-3.2-1B",
                "--megatron-path",
                str(megatron_output_dir),
                "--torch-dtype",
                torch_dtype,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            # Check that the import completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Import with {test_name} settings failed with return code {result.returncode}"

            # Verify expected settings were used
            # Verify successful completion
            output = result.stdout + result.stderr
            assert "CPU import complete:" in output, (
                f"Import success message not found in {test_name} output. Output: {output}"
            )

            print(f"SUCCESS: {test_name} settings test completed successfully")

        except Exception as e:
            print(f"Error during {test_name} settings test: {e}")
            raise
