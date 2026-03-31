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
Functional tests for Kimi K2.5 VL model conversion (HF ↔ Megatron round-trip).

Requires ``trust_remote_code`` because the Kimi K2.5 model classes are
defined in the HuggingFace repository (not in the standard transformers
library).  The test fixture downloads the config and custom code from
``moonshotai/Kimi-K2.5``, shrinks the model to toy dimensions, and saves
a lightweight checkpoint that the round-trip conversion can process in
seconds.
"""

import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import megatron.bridge.models.conversion.transformers_compat  # noqa: F401 – patch is_torch_fx_available for transformers 5.x
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class TestKimiK25VLConversion:
    """Test Kimi K2.5 VL model conversion with different parallelism configurations."""

    @pytest.fixture(scope="class")
    def kimi_k25_vl_toy_model_path(self, tmp_path_factory):
        """Create a toy Kimi K2.5 VL checkpoint with drastically reduced dimensions.

        Downloads the real config and custom modelling code from HuggingFace,
        then overrides layer counts, expert counts, and vocab size to produce
        a ~300 MB checkpoint instead of the full ~1 TB model.
        """
        temp_dir = tmp_path_factory.mktemp("kimi_k25_vl_toy_model")
        model_dir = temp_dir / "kimi_k25_vl_toy"
        model_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)
        config.torch_dtype = torch.bfloat16

        text = config.text_config
        text.num_hidden_layers = 2
        text.vocab_size = 2048
        text.n_routed_experts = 4
        text.num_experts_per_tok = 2
        # All layers dense to avoid INT4 re-quantization in round-trip export
        text.first_k_dense_replace = 2
        text.n_shared_experts = 1
        text.max_position_embeddings = 4096

        for cfg in (config, text):
            for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
                if getattr(cfg, attr, None) is not None and getattr(cfg, attr) >= 2048:
                    setattr(cfg, attr, 0)

        if hasattr(config, "image_token_id") and config.image_token_id >= 2048:
            config.image_token_id = 3

        config.vision_config.depth = 1

        for cfg in (config, text):
            if hasattr(cfg, "quantization_config"):
                delattr(cfg, "quantization_config")

        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(dtype=torch.bfloat16)

        for m in model.modules():
            if hasattr(m, "_tied_weights_keys"):
                m._tied_weights_keys = []

        model.save_pretrained(model_dir, safe_serialization=True)

        # Copy custom code files so the saved checkpoint can be loaded with
        # trust_remote_code=True from the local path.
        source_file = inspect.getfile(type(model))
        source_dir = Path(source_file).parent
        for py_file in source_dir.glob("*.py"):
            target = model_dir / py_file.name
            if not target.exists():
                shutil.copy2(py_file, target)

        config.save_pretrained(model_dir)

        try:
            tokenizer = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)
            tokenizer.save_pretrained(model_dir)
        except Exception:
            tokenizer_config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "vocab_size": 2048,
            }
            with open(model_dir / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, kimi_k25_vl_toy_model_path):
        """Verify the toy checkpoint was created and contains the expected files."""
        model_path = Path(kimi_k25_vl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "model.safetensors.index.json"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data.get("model_type") == "kimi_k25"
        text_cfg = config_data.get("text_config", config_data)
        assert text_cfg.get("num_hidden_layers") == 2
        assert text_cfg.get("n_routed_experts") == 4
        assert "vision_config" in config_data

        print(f"SUCCESS: Kimi K2.5 VL toy model created at {kimi_k25_vl_toy_model_path}")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
        ],
    )
    def test_kimi_k25_vl_conversion_parallelism(self, kimi_k25_vl_toy_model_path, tmp_path, tp, pp, test_name):
        """Test round-trip conversion (HF → Megatron → HF) with TP=2."""
        test_output_dir = tmp_path / f"kimi_k25_vl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

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
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            kimi_k25_vl_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--trust-remote-code",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent.parent,
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Kimi K2.5 VL {test_name} conversion failed with return code {result.returncode}"

            model_name = Path(kimi_k25_vl_toy_model_path).name
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config.get("model_type") == "kimi_k25"
            text_cfg = saved_config.get("text_config", saved_config)
            assert text_cfg.get("num_hidden_layers") == 2
            assert text_cfg.get("n_routed_experts") == 4
            assert "vision_config" in saved_config

            print(f"SUCCESS: Kimi K2.5 VL {test_name} conversion test completed")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Kimi K2.5 VL {test_name} conversion test: {e}")
            raise
