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
Tests for ERNIE 4.5 VL MoE model conversion between HuggingFace and Megatron-Core formats.

Usage:
    uv run python -m torch.distributed.run --nproc_per_node=1 -m pytest \
        tests/functional_tests/test_groups/models/ernie_vl/test_ernie45_vl_conversion.py::TestErnie45VLConversion::test_toy_model_creation
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch


try:
    from transformers import AutoTokenizer
    from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeConfig,
    )
    from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeForConditionalGeneration,
    )

    _HAS_ERNIE45_VL = True
except ImportError:
    _HAS_ERNIE45_VL = False


# Tiny model config optimized for fast testing.
# ERNIE 4.5 VL MoE architecture: dual-pool MoE (text_moe + vision_moe) + shared_experts.
# Reduced from 28B to ~10M parameters for fast CI testing.
#
# Key architectural constraints preserved:
# - Layer 0 is dense MLP, layers 1+ are sparse MoE
# - Dual-pool MoE: text_moe (intermediate=1536->64) + vision_moe (intermediate=512->32)
# - Shared experts: moe_num_shared_experts=2
# - GQA: num_attention_heads must be divisible by num_key_value_heads
# - 3D M-RoPE: mrope_section sums to head_dim//2 (here [2, 2, 2] sums to 6 for head_dim=12)
# - num_key_value_heads >= 2 for TP=2 compatibility
HF_ERNIE45_VL_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Ernie4_5_VLMoeForConditionalGeneration"],
    "model_type": "ernie4_5_vl_moe",
    "tie_word_embeddings": True,
    "image_start_token_id": 101304,
    "image_end_token_id": 101305,
    "image_token_id": 100295,
    "video_start_token_id": 101306,
    "video_end_token_id": 101307,
    "video_token_id": 103367,
    "text_config": {
        "model_type": "ernie4_5_vl_moe_text",
        "vocab_size": 2048,
        "hidden_size": 48,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "use_bias": False,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 500000.0,
            "mrope_section": [2, 2, 2],
        },
        # MoE settings: scaled down from production (64 experts -> 4)
        "moe_intermediate_size": [64, 32],
        "moe_k": 2,
        "moe_num_experts": 4,
        "moe_num_shared_experts": 2,
        "moe_norm_min": 1e-12,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        # Layer 0 = dense, layers 1-3 = sparse
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    },
    "vision_config": {
        "model_type": "ernie4_5_vl_moe_vision",
        "depth": 1,
        "hidden_size": 48,
        "hidden_act": "quick_gelu",
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "intermediate_size": 128,
        "temporal_merge_size": 2,
        "rms_norm_eps": 1e-6,
    },
}


@pytest.mark.skipif(not _HAS_ERNIE45_VL, reason="ERNIE 4.5 VL MoE model not available in transformers")
class TestErnie45VLConversion:
    """
    Test ERNIE 4.5 VL MoE model conversion from local HuggingFace model
    with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def ernie45_vl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace ERNIE 4.5 VL MoE toy model from config
        to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures.

        Returns:
            str: Path to the saved HuggingFace model directory.
        """
        temp_dir = tmp_path_factory.mktemp("ernie45_vl_toy_model")
        model_dir = temp_dir / "ernie45_vl_toy"

        # Create ERNIE 4.5 VL MoE config from the toy model config dict
        config = Ernie4_5_VLMoeConfig(**HF_ERNIE45_VL_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Ensure rope_parameters is set on text_config
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 500000.0,
                "mrope_section": [2, 2, 2],
            }

        # Create model with random weights and convert to bfloat16
        model = Ernie4_5_VLMoeForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        # Load tokenizer from local model or a reference model, or create minimal fallback
        _local_tokenizer_path = (
            Path(__file__).parent.parent.parent.parent.parent.parent / "ERNIE-4.5-VL-28B-A3B-Thinking"
        )
        try:
            if _local_tokenizer_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(_local_tokenizer_path))
            else:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            tokenizer.save_pretrained(model_dir)
        except (OSError, ValueError):
            # Create a functional dummy tokenizer from a readily available model
            # so that save_hf_pretrained can re-save it without errors.
            try:
                from tokenizers import Tokenizer as TkTokenizer
                from tokenizers import models as tk_models
                from transformers import PreTrainedTokenizerFast

                # Build a minimal BPE tokenizer with a few tokens
                tk = TkTokenizer(tk_models.BPE())
                tk.add_special_tokens(["<s>", "</s>", "<pad>", "<unk>"])
                # Pad the vocab to the expected size with dummy tokens
                dummy_tokens = [f"<tok_{i}>" for i in range(103420)]
                tk.add_tokens(dummy_tokens)
                fast_tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=tk,
                    bos_token="<s>",
                    eos_token="</s>",
                    pad_token="<pad>",
                    unk_token="<unk>",
                )
                model_dir.mkdir(parents=True, exist_ok=True)
                fast_tokenizer.save_pretrained(model_dir)
            except (OSError, ValueError):
                # Last resort: just create the directory so save_pretrained can write weights
                model_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights (safetensors format)
        model.save_pretrained(model_dir, safe_serialization=True)

        # Overwrite config.json with the toy config to ensure exact key structure
        config_to_save = HF_ERNIE45_VL_MOE_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, ernie45_vl_toy_model_path):
        """
        Test that the ERNIE 4.5 VL MoE toy model is created correctly and can be loaded.

        Args:
            ernie45_vl_toy_model_path: Path to the toy model (from fixture).
        """
        model_path = Path(ernie45_vl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "model.safetensors.index.json"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "ernie4_5_vl_moe"
        assert "text_config" in config_data
        assert "vision_config" in config_data
        assert config_data["text_config"]["hidden_size"] == 48
        assert config_data["text_config"]["num_hidden_layers"] == 4
        assert config_data["text_config"]["num_attention_heads"] == 4
        assert config_data["text_config"]["moe_num_experts"] == 4
        assert config_data["text_config"]["moe_intermediate_size"] == [64, 32]
        assert config_data["text_config"]["mlp_layer_types"] == ["dense", "sparse", "sparse", "sparse"]

        # Verify model can be loaded from pretrained
        _ = Ernie4_5_VLMoeForConditionalGeneration.from_pretrained(
            ernie45_vl_toy_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

        # Try loading the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(ernie45_vl_toy_model_path)
            print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer (OK for conversion testing): {e}")

        print(f"SUCCESS: ERNIE 4.5 VL MoE toy model created and validated at {ernie45_vl_toy_model_path}")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_ernie45_vl_conversion_parallelism(self, ernie45_vl_toy_model_path, tmp_path, tp, pp, ep, test_name):
        """
        Test ERNIE 4.5 VL MoE model conversion with different parallelism configurations.

        Covers:
        - TP (Tensor Parallelism): splits attention heads and MLP across GPUs
        - PP (Pipeline Parallelism): splits transformer layers across GPUs
        - EP (Expert Parallelism): splits MoE experts across GPUs

        The EP test validates that dual-pool MoE (text_moe_layer + vision_moe_layer)
        correctly handles per-pool expert offset when sharding across EP ranks.

        Args:
            ernie45_vl_toy_model_path: Path to the toy model (from fixture).
            tmp_path: Pytest temporary path fixture.
            tp: Tensor parallelism size.
            pp: Pipeline parallelism size.
            ep: Expert parallelism size.
            test_name: Name of the test for identification.
        """
        test_output_dir = tmp_path / f"ernie45_vl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run HF-to-Megatron roundtrip conversion as a subprocess
        # Use sys.executable to ensure the same Python interpreter (venv-aware)
        import sys

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            ernie45_vl_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
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
                pytest.fail(f"ERNIE 4.5 VL {test_name} conversion failed with return code {result.returncode}")

            # Verify converted model directory exists
            model_name = Path(ernie45_vl_toy_model_path).name  # "ernie45_vl_toy"
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Verify the config contains ERNIE 4.5 VL-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "ernie4_5_vl_moe", "Model type should be ernie4_5_vl_moe"
            assert "text_config" in saved_config, "VL model should have text_config"
            assert "vision_config" in saved_config, "VL model should have vision_config"
            assert saved_config["text_config"]["hidden_size"] == 48, "Hidden size should match toy config"
            assert saved_config["text_config"]["num_attention_heads"] == 4, (
                "Number of attention heads should match toy config"
            )

            print(f"SUCCESS: ERNIE 4.5 VL {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during ERNIE 4.5 VL {test_name} conversion test: {e}")
            raise

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,nproc,with_vision,test_name",
        [
            (1, 1, 1, 1, False, "single_gpu"),
            (1, 1, 2, 2, False, "EP2"),
            (1, 1, 1, 1, True, "single_gpu_vision"),
            (1, 1, 2, 2, True, "EP2_vision"),
        ],
    )
    def test_ernie45_vl_forward_backward(
        self, ernie45_vl_toy_model_path, tmp_path, tp, pp, ep, nproc, with_vision, test_name
    ):
        """
        Test ERNIE 4.5 VL MoE model forward and backward pass.

        Builds the Megatron model from the toy HF checkpoint via AutoBridge,
        runs a forward pass and backward pass, and verifies:
        - Forward produces finite output
        - Backward produces gradients on trainable parameters

        When with_vision=True, a dummy image is injected to exercise the full
        vision pipeline: ViT patch embedding -> vision transformer -> resampler
        -> embedding injection -> language model forward. Vision tower and
        resampler gradients are also verified.

        Args:
            ernie45_vl_toy_model_path: Path to the toy model (from fixture).
            tmp_path: Pytest temporary path fixture.
            tp: Tensor parallelism size.
            pp: Pipeline parallelism size.
            ep: Expert parallelism size.
            nproc: Number of processes for torchrun.
            with_vision: Whether to include a dummy image input.
            test_name: Name of the test for identification.
        """
        import sys

        fwd_bwd_script = str(Path(__file__).parent / "ernie45_vl_fwd_bwd.py")

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--nnodes=1",
            fwd_bwd_script,
            "--hf-model-path",
            ernie45_vl_toy_model_path,
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
            "--seq-len",
            "16",
        ]

        if with_vision:
            cmd.append("--with-vision")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent.parent,
                timeout=300,
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout[-3000:]}")
                print(f"STDERR: {result.stderr[-3000:]}")
                pytest.fail(
                    f"ERNIE 4.5 VL {test_name} forward/backward test failed with return code {result.returncode}"
                )

            # Verify the output contains the success marker
            assert "ALL CHECKS PASSED" in result.stdout, (
                f"Forward/backward test did not complete successfully. STDOUT tail: {result.stdout[-1000:]}"
            )

            print(f"SUCCESS: ERNIE 4.5 VL {test_name} forward/backward test passed")

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: ERNIE 4.5 VL {test_name} forward/backward test timed out after 300s")
            raise
        except Exception as e:
            print(f"Error during ERNIE 4.5 VL {test_name} forward/backward test: {e}")
            raise
