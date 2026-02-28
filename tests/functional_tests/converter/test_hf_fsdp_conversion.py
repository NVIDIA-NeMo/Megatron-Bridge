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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


DEEPSEEK_V3_OVERRIDES = {
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "kv_lora_rank": 512,
    "max_position_embeddings": 163840,
    "moe_intermediate_size": 768,
    "n_group": 4,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_attention_heads": 32,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 0,
    "q_lora_rank": 512,
    "topk_group": 4,
    "vocab_size": 129280,
}


class TestHFFSDPConversion:
    """
    Test functional conversion between HuggingFace and Megatron-FSDP using hf_fsdp_roundtrip.py.
    """

    @pytest.fixture(scope="class")
    def deepseek_toy_model_path(self, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("deepseek_toy_model")
        model_dir = temp_dir / "deepseek_toy"

        # Create a minimal config and model using Auto classes to avoid direct imports
        config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)

        for key, value in DEEPSEEK_V3_OVERRIDES.items():
            setattr(config, key, value)
        del config.quantization_config

        # Fallback to a generic small model; for conversion flows we only need keys/config
        # Some environments may not have DeepSeek classes; we just ensure a valid HF directory
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path="deepseek-ai/DeepSeek-V3",
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id="deepseek-ai/DeepSeek-V3",
        )
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        for k, v in model.named_parameters():
            if "e_score_correction_bias" in k:
                v.data = v.data.to(torch.float32)

        # Save a tokenizer (use a lightweight compatible tokenizer)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            pass

        # Save model and config
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Ensure config.json exists with expected keys
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,ep,test_name",
        [
            (1, 1, "FSDP_no_parallel"),
            (1, 2, "FSDP_EP"),
        ],
    )
    def test_hf_fsdp_roundtrip(self, deepseek_toy_model_path, tmp_path, tp, ep, test_name):
        """
        Test HF to Megatron-FSDP roundtrip conversion with different parallelism configurations.
        """

        # Create temporary output directory
        test_output_dir = tmp_path / test_name
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "examples/conversion/hf_fsdp_roundtrip.py",
            "--hf-model-id",
            deepseek_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--ep",
            str(ep),
            "--trust-remote-code",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"FSDP Roundtrip failed with return code {result.returncode}"

            # Verify that the converted model was saved
            model_name = Path(deepseek_toy_model_path).name
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in converted model at {converted_model_dir}"
            )

            print(f"SUCCESS: {test_name} FSDP roundtrip test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during {test_name} FSDP roundtrip test: {e}")
            raise
