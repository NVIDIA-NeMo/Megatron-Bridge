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

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from megatron.core import parallel_state
from transformers import MegatronBertConfig, MegatronBertForMaskedLM

from megatron.bridge import AutoBridge


# Toy model config with minimal layers for testing. Mirrors the shape of the real
# `nvidia/megatron-bert-uncased-345m` checkpoint (Pre-LayerNorm BERT), just much smaller.
HF_BERT_TOY_MODEL_CONFIG = {
    "architectures": ["MegatronBertForMaskedLM"],
    "model_type": "megatron-bert",
    "vocab_size": 128,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "max_position_embeddings": 64,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "initializer_range": 0.02,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
}


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return str(sock.getsockname()[1])


class TestBertConversion:
    """Test BERT (`MegatronBertForMaskedLM`) model conversion from a local toy HuggingFace model."""

    @pytest.fixture(scope="class")
    @classmethod
    def bert_toy_model_path(cls, tmp_path_factory):
        """Create and save a HuggingFace MegatronBertForMaskedLM toy model to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures.

        Returns:
            str: Path to the saved HuggingFace model directory.
        """
        temp_dir = tmp_path_factory.mktemp("bert_toy_model")
        model_dir = temp_dir / "bert_toy"

        config = MegatronBertConfig(**HF_BERT_TOY_MODEL_CONFIG)
        model = MegatronBertForMaskedLM(config).bfloat16()
        model.save_pretrained(model_dir, safe_serialization=True)

        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_BERT_TOY_MODEL_CONFIG, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, bert_toy_model_path):
        """Test that the toy model is created correctly and can be loaded.

        Args:
            bert_toy_model_path: Path to the toy BERT model (from fixture).
        """
        model_path = Path(bert_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        weights_file = model_path / "model.safetensors"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "megatron-bert"
        assert config_data["hidden_size"] == 32
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 4
        assert config_data["vocab_size"] == 128

        model = MegatronBertForMaskedLM.from_pretrained(bert_toy_model_path, torch_dtype=torch.bfloat16)
        assert len(model.bert.encoder.layer) == 2

    @pytest.mark.run_only_on("GPU")
    def test_bert_forward_parity(self, bert_toy_model_path, monkeypatch):
        """Test numerical parity between Hugging Face and Megatron-Core BERT logits."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", _find_free_port())

        hf_model = MegatronBertForMaskedLM.from_pretrained(
            bert_toy_model_path,
            dtype=torch.bfloat16,
        ).cuda()
        hf_model.eval()

        bridge = AutoBridge.from_hf_pretrained(bert_toy_model_path, torch_dtype=torch.bfloat16)
        provider = bridge.to_megatron_provider(load_weights=True)

        try:
            provider.finalize()
            provider.initialize_model_parallel(seed=0)
            megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)[0]
            megatron_model.eval()

            input_ids = torch.tensor([[5, 7, 11, 13, 17, 19]], device="cuda")
            attention_mask = torch.ones_like(input_ids)
            token_type_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], device="cuda")

            with torch.no_grad():
                hf_logits = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).logits
                megatron_logits, _ = megatron_model(input_ids, attention_mask, token_type_ids)

            torch.testing.assert_close(megatron_logits.float(), hf_logits.float(), atol=1e-2, rtol=1e-2)
            cosine_similarity = torch.nn.functional.cosine_similarity(
                megatron_logits.float().flatten(),
                hf_logits.float().flatten(),
                dim=0,
            )
            assert cosine_similarity.item() > 0.9999
        finally:
            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    def test_bert_single_gpu_roundtrip(self, bert_toy_model_path, tmp_path):
        """Test BERT single-GPU roundtrip conversion (HF -> Megatron -> HF).

        This exercises the exact-parity (Level 1) check from the parity-testing skill:
        `hf_megatron_roundtrip.py` fails outright unless every independently serialized
        HF tensor round-trips correctly.

        Args:
            bert_toy_model_path: Path to the toy BERT model (from fixture).
            tmp_path: Pytest temporary path fixture.
        """
        cmd = [
            sys.executable,
            "examples/conversion/hf_megatron_roundtrip.py",
            "--hf-model-id",
            bert_toy_model_path,
            "--output-dir",
            str(tmp_path / "bert_roundtrip"),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )

        assert result.returncode == 0, (
            f"BERT single-GPU roundtrip failed with return code {result.returncode}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_bert_conversion_parallelism(self, bert_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test BERT model conversion with different parallelism configurations.

        Args:
            bert_toy_model_path: Path to the toy BERT model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """
        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"bert_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable,
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
            bert_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent.parent
        )
        assert result.returncode == 0, (
            f"BERT {test_name} conversion failed with return code {result.returncode}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

        # Verify that the converted model was saved
        # The output directory should be named after the last part of the model path
        model_name = Path(bert_toy_model_path).name  # "bert_toy"
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        # Check that essential model files exist
        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"

        # Check for model weights file (could be either safetensors or pytorch_model.bin)
        weights_file_safetensors = converted_model_dir / "model.safetensors"
        weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
        assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
            f"Model weights file not found in converted model at {converted_model_dir}"
        )

        # Verify the config contains BERT-specific parameters
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "megatron-bert", "Model type should be megatron-bert"
        assert saved_config["hidden_size"] == 32, "Hidden size should match toy config"
        assert saved_config["num_attention_heads"] == 4, "Number of attention heads should match toy config"
        assert saved_config["hidden_act"] == "gelu", "Activation should match toy config"
        assert saved_config["hidden_dropout_prob"] == 0.0, "Hidden dropout should match toy config"
        assert saved_config["attention_probs_dropout_prob"] == 0.0, "Attention dropout should match toy config"
