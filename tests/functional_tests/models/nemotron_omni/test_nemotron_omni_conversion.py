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

"""Functional test: Nemotron Omni roundtrip conversion (HF <-> Megatron).

Uses a toy omni model (small dimensions, random weights) to verify that the
full HF -> Megatron -> HF roundtrip preserves every weight tensor exactly.

The test requires GPU access and uses the sound-aware HF model class from the
real omni checkpoint directory (option (a) from the plan).  For CI, refactor
to bundle the .py model files as test fixtures (option (b)).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


# The local omni checkpoint provides the sound-aware HF model class.
# The VL-only model class on HF Hub does NOT support sound_config.
NEMOTRON_OMNI_HF_ID = (
    "/lustre/fs1/portfolios/coreai/users/aroshanghias/"
    "checkpoints/sft_moe_long_context_nodes64-seq49152-lr1e-5-cp2-0120/megatron_hf"
)

# Toy model overrides: small dimensions for fast testing.
# Explicit values -- NOT SoundConfig() defaults, which differ from checkpoint.
HF_NEMOTRON_OMNI_TOY_MODEL_OVERRIDES = {
    # LLM backbone (Nemotron-H, small)
    "attention_head_dim": 48,
    "chunk_size": 48,
    "expand": 2,
    "hidden_size": 768,
    "hybrid_override_pattern": "M*M-",
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 64,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 24,
    "max_position_embeddings": 8192,
    "n_groups": 8,
    "num_attention_heads": 16,
    "num_hidden_layers": 4,
    "num_key_value_heads": 8,
    "ssm_state_size": 128,
    "vocab_size": 131072,
    # Vision config (RADIO, small)
    "vision_config": {
        "hidden_size": 256,
        "image_size": 384,
        "intermediate_size": 1024,
        "model_type": "radio_vision_model",
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "patch_size": 16,
    },
    # Sound config (Parakeet conformer, small)
    "sound_config": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 128,
        "num_mel_bins": 128,
        "subsampling_factor": 8,
        "conv_kernel_size": 9,
        "projection_hidden_size": 256,
        "projection_bias": False,
        "sampling_rate": 16000,
        "model_type": "parakeet",
    },
    "sound_context_token_id": 29,
}


class TestNemotronOmniConversion:
    """Test Nemotron Omni roundtrip conversion with sound support."""

    @pytest.fixture(scope="class")
    def nemotron_omni_toy_model_path(self, tmp_path_factory):
        """Create a toy omni model from the real checkpoint's model class."""
        temp_dir = tmp_path_factory.mktemp("nemotron_omni_toy_model")
        model_dir = temp_dir / "nemotron_omni_toy"

        config = AutoConfig.from_pretrained(NEMOTRON_OMNI_HF_ID, trust_remote_code=True)

        # Apply LLM and vision overrides
        vision_overrides = None
        sound_overrides = None
        for k, v in HF_NEMOTRON_OMNI_TOY_MODEL_OVERRIDES.items():
            if k == "vision_config":
                vision_overrides = v
            elif k == "sound_config":
                sound_overrides = v
            else:
                setattr(config, k, v)

        if vision_overrides and hasattr(config, "vision_config"):
            for k, v in vision_overrides.items():
                setattr(config.vision_config, k, v)

        if sound_overrides and hasattr(config, "sound_config"):
            for k, v in sound_overrides.items():
                setattr(config.sound_config, k, v)

        # Instantiate model with random weights
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path=NEMOTRON_OMNI_HF_ID,
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=True,
            repo_id=NEMOTRON_OMNI_HF_ID,
        )
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # Save tokenizer (from the same checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(NEMOTRON_OMNI_HF_ID, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        # Save model + config + modeling code
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Also copy audio_model.py and configuration.py if they exist
        checkpoint_dir = Path(NEMOTRON_OMNI_HF_ID)
        for extra_file in ["audio_model.py", "configuration.py", "configuration_nemotron_h.py",
                           "configuration_radio.py"]:
            src = checkpoint_dir / extra_file
            if src.exists():
                shutil.copy(src, model_dir)

        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, nemotron_omni_toy_model_path):
        """Verify the toy omni model directory is valid."""
        model_path = Path(nemotron_omni_toy_model_path)
        assert model_path.exists()

        config_file = model_path / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["hidden_size"] == 768
        assert config_data["num_hidden_layers"] == 4
        assert config_data["num_attention_heads"] == 16
        assert "vision_config" in config_data
        assert "sound_config" in config_data
        assert config_data["sound_config"]["hidden_size"] == 64
        assert config_data["sound_config"]["num_hidden_layers"] == 2
        assert config_data["sound_context_token_id"] == 29

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (1, 1, "TP1"),
            (2, 1, "TP2"),
        ],
    )
    def test_nemotron_omni_conversion_parallelism(
        self, nemotron_omni_toy_model_path, tmp_path, tp, pp, test_name,
    ):
        """Test roundtrip conversion with different parallelism configs."""
        test_output_dir = tmp_path / f"nemotron_omni_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={max(tp * pp, 2)}",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            nemotron_omni_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent.parent.parent,
        )
        print(cmd)

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, (
                f"Nemotron Omni {test_name} conversion failed "
                f"with return code {result.returncode}"
            )

        model_name = Path(nemotron_omni_toy_model_path).name
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), (
            f"Converted model directory not found at {converted_model_dir}"
        )

        config_file = converted_model_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["hidden_size"] == 768
        assert saved_config["num_attention_heads"] == 16
        assert "vision_config" in saved_config
        # Sound config must survive the roundtrip
        assert "sound_config" in saved_config, (
            "sound_config missing from roundtrip config.json"
        )
        assert saved_config["sound_config"]["hidden_size"] == 64
        assert saved_config["sound_config"]["num_hidden_layers"] == 2
        assert saved_config.get("sound_context_token_id") == 29
