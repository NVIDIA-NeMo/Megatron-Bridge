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
from transformers import AutoConfig, AutoTokenizer

from megatron.bridge.models.conversion.utils import get_causal_lm_class_via_auto_map


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


class TestDeepSeekConversion:
    """Functional tests for DeepSeek toy conversion paths."""

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
        model_class = get_causal_lm_class_via_auto_map("deepseek-ai/DeepSeek-V3", config)
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # Save a tokenizer (use a lightweight compatible tokenizer)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            pass

        # Save model and config
        model.save_pretrained(model_dir, safe_serialization=True)
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
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_deepseek_conversion_parallelism(self, deepseek_toy_model_path, tmp_path, tp, pp, ep, test_name):
        test_output_dir = tmp_path / f"deepseek_{test_name}"
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
            deepseek_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        assert result.returncode == 0, f"DeepSeek {test_name} conversion failed with {result.returncode}"

        # Verify outputs
        model_name = Path(deepseek_toy_model_path).name
        converted_dir = test_output_dir / model_name
        assert converted_dir.exists()

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        weights_file_safetensors = converted_dir / "model.safetensors"
        weights_file_pytorch = converted_dir / "pytorch_model.bin"
        weights_found = weights_file_safetensors.exists() or weights_file_pytorch.exists()
        if not weights_found:
            shards_st = list(converted_dir.glob("model-*-of-*.safetensors"))
            shards_pt = list(converted_dir.glob("pytorch_model-*-of-*.bin"))
            weights_found = len(shards_st) > 0 or len(shards_pt) > 0
        assert weights_found

        with open(config_file) as f:
            saved = json.load(f)

        assert saved["model_type"] == "deepseek_v3", "Model type should be deepseek_v3"
        assert saved["vocab_size"] == 129280
        assert saved["hidden_size"] == 2048
        assert saved["n_routed_experts"] == 4
        assert saved["num_experts_per_tok"] == 4
        assert saved["num_hidden_layers"] == 2
        assert saved["moe_intermediate_size"] == 768

        print(f"SUCCESS: DeepSeek {test_name} conversion test completed successfully")
        print(f"Converted model saved at: {converted_dir}")
        print(
            f"MoE parameters preserved: {saved['n_routed_experts']} experts, {saved['num_experts_per_tok']} per token"
        )
