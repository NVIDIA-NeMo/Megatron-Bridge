#!/usr/bin/env python3
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
import subprocess
from pathlib import Path

import pytest


# Minimal GPT-OSS config used for building a tiny local HF directory to test conversion.
GPT_OSS_TOY_OVERRIDES = {
    "architectures": ["GptOssForCausalLM"],
    "hidden_size": 512,
    "intermediate_size": 1536,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "num_local_experts": 8,  # enable MoE to exercise EP handling
    "vocab_size": 32000,
    "torch_dtype": "bfloat16",
}


class TestGptOssConversion:
    """Functional tests for GPT-OSS toy conversion paths."""

    @pytest.fixture(scope="class")
    def gpt_oss_toy_model_path(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("gptoss_toy_model")
        model_dir = tmp_dir / "gpt_oss_toy"

        # Importorskip ensures test is skipped gracefully if transformers lacks GPT-OSS
        transformers = pytest.importorskip("transformers")
        GptOssForCausalLM = getattr(transformers, "GptOssForCausalLM", None)
        GptOssConfig = getattr(transformers, "GptOssConfig", None)
        if GptOssForCausalLM is None or GptOssConfig is None:
            pytest.skip("transformers installation does not include GPT-OSS classes")

        # Build tiny config and model
        config = GptOssConfig(**GPT_OSS_TOY_OVERRIDES)
        model = GptOssForCausalLM(config)
        if hasattr(model, "bfloat16"):
            model = model.bfloat16()

        # Save tokenizer (fallback to gpt2 tokenizer if GPT-OSS doesn't ship one)
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("gpt2")
            tok.save_pretrained(model_dir)
        except Exception:
            pass

        # Save model and an explicit config.json with our overrides
        model.save_pretrained(model_dir, safe_serialization=True)
        with open(model_dir / "config.json", "w") as f:
            json.dump(GPT_OSS_TOY_OVERRIDES, f, indent=2)

        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_gpt_oss_conversion_parallelism(self, gpt_oss_toy_model_path, tmp_path, tp, pp, ep, test_name):
        out_dir = tmp_path / f"gpt_oss_{test_name}"
        out_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/workspace/.coverage",
            "--source=/workspace/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            gpt_oss_toy_model_path,
            "--output-dir",
            str(out_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        assert result.returncode == 0, f"GPT-OSS {test_name} conversion failed with {result.returncode}"

        # Verify output structure
        model_name = Path(gpt_oss_toy_model_path).name
        converted_dir = out_dir / model_name
        assert converted_dir.exists()

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        # weights can be either consolidated or sharded, and in safetensors or bin
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

        # Minimal sanity checks on saved config
        assert saved["num_hidden_layers"] == GPT_OSS_TOY_OVERRIDES["num_hidden_layers"]
        assert saved["num_attention_heads"] == GPT_OSS_TOY_OVERRIDES["num_attention_heads"]
        assert saved.get("num_local_experts", 0) == GPT_OSS_TOY_OVERRIDES["num_local_experts"]
        assert saved["vocab_size"] == GPT_OSS_TOY_OVERRIDES["vocab_size"]

        print(f"SUCCESS: GPT-OSS {test_name} conversion test completed successfully")
        print(f"Converted model saved at: {converted_dir}")
