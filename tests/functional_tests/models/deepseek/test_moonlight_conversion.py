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
from transformers.dynamic_module_utils import get_class_from_dynamic_module


MOONLIGHT_HF_MODEL_ID_PRIMARY = "moonshotai/Moonlight-16B-A3B-Instruct"
MOONLIGHT_HF_MODEL_ID_FALLBACK = "moonshotai/Moonlight-16B-A3B"

# Keep this toy config intentionally small to make instantiation cheap.
# Moonlight config keys may differ slightly across revisions; we apply overrides
# opportunistically and only assert on keys we set.
MOONLIGHT_OVERRIDES = {
    # Core size reductions
    "num_hidden_layers": 2,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    # MoE-ish knobs (if present)
    "n_group": 2,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 768,
    # Common DeepSeek-like knobs (if present)
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "first_k_dense_replace": 1,
    "topk_group": 2,
}


def _try_load_config():
    """Try Instruct model id first, fallback to base."""
    try:
        return MOONLIGHT_HF_MODEL_ID_PRIMARY, AutoConfig.from_pretrained(
            MOONLIGHT_HF_MODEL_ID_PRIMARY, trust_remote_code=True
        )
    except Exception:
        return MOONLIGHT_HF_MODEL_ID_FALLBACK, AutoConfig.from_pretrained(
            MOONLIGHT_HF_MODEL_ID_FALLBACK, trust_remote_code=True
        )


class TestMoonlightConversion:
    """Functional tests for Moonlight toy conversion paths."""

    @pytest.fixture(scope="class")
    def moonlight_toy_model_path(self, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("moonlight_toy_model")
        model_dir = temp_dir / "moonlight_toy"

        hf_model_id, config = _try_load_config()

        for key, value in MOONLIGHT_OVERRIDES.items():
            setattr(config, key, value)

        # Some configs ship a quantization_config that isn't JSON-serializable.
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        auto_map = getattr(config, "auto_map", None) or {}
        model_class_ref = auto_map.get("AutoModelForCausalLM")
        if model_class_ref is None:
            raise RuntimeError(f"Expected config.auto_map['AutoModelForCausalLM'] for {hf_model_id}, got: {auto_map}")

        model_class = get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path=hf_model_id,
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id=hf_model_id,
        )

        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # Save a tokenizer (lightweight compatible tokenizer is fine for conversion flows).
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
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_moonlight_conversion_parallelism(self, moonlight_toy_model_path, tmp_path, tp, pp, ep, test_name):
        test_output_dir = tmp_path / f"moonlight_{test_name}"
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
            moonlight_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        assert result.returncode == 0, f"Moonlight {test_name} conversion failed with {result.returncode}"

        # Verify outputs
        model_name = Path(moonlight_toy_model_path).name
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

        # Assert the values we explicitly set in the toy config are preserved.
        for k, v in MOONLIGHT_OVERRIDES.items():
            if k in saved:
                assert saved[k] == v, f"Expected {k}={v}, got {saved[k]}"

        assert "model_type" in saved

        print(f"SUCCESS: Moonlight {test_name} conversion test completed successfully")
        print(f"Converted model saved at: {converted_dir}")
