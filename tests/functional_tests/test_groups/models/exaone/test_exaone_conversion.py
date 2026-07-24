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

"""Functional conversion and forward-equivalence tests for EXAONE models."""

import re
import subprocess
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Exaone4_5_ForConditionalGeneration,
    ExaoneMoeConfig,
    ExaoneMoeForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config


pytestmark = pytest.mark.run_only_on("GPU")
_REPO_ROOT = Path(__file__).resolve().parents[5]
_SIMILARITY_RE = re.compile(r"Cosine similarity: ([0-9.]+)")

_EXAONE4_CONFIG = {
    "architectures": ["Exaone4ForCausalLM"],
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 128,
    "sliding_window": 32,
    "sliding_window_pattern": 2,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "tie_word_embeddings": False,
}

_EXAONE45_CONFIG = {
    "architectures": ["Exaone4_5_ForConditionalGeneration"],
    "image_token_id": 67,
    "video_token_id": 68,
    "tie_word_embeddings": False,
    "text_config": {"model_type": "exaone4", **_EXAONE4_CONFIG},
    "vision_config": {
        "depth": 1,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 8,
        "num_key_value_heads": 4,
        "in_channels": 3,
        "patch_size": 4,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 8,
        "out_hidden_size": 64,
        "fullatt_block_indexes": [0],
    },
}

_EXAONE_MOE_CONFIG = {
    "architectures": ["ExaoneMoEForCausalLM"],
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 256,
    "sliding_window": 32,
    "sliding_window_pattern": "LLLG",
    "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
    "tie_word_embeddings": False,
    "is_moe_layer": [False, True, True, True],
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "first_k_dense_replace": 1,
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "num_nextn_predict_layers": 1,
}


def _save_tokenizer(model_dir: Path) -> None:
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "hello": 4,
        "world": 5,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    hf_tokenizer.save_pretrained(model_dir)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, cwd=_REPO_ROOT)
    if result.returncode != 0:
        pytest.fail(
            f"Command failed with return code {result.returncode}: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _assert_roundtrip(model_path: str, output_dir: Path) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            model_path,
            "--output-dir",
            str(output_dir),
            "--tp",
            "2",
            "--pp",
            "1",
        ]
    )

    exported_path = output_dir / Path(model_path).name
    assert (exported_path / "config.json").is_file()
    assert (exported_path / "model.safetensors").is_file()


def _assert_forward_equivalence(model_path: str, model_class: str) -> None:
    result = _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "examples/conversion/compare_hf_and_megatron/compare.py",
            "--hf_model_path",
            model_path,
            "--model_class",
            model_class,
            "--prompt",
            "hello world",
            "--tp",
            "1",
        ]
    )

    match = _SIMILARITY_RE.search(result.stdout)
    assert match is not None, result.stdout
    assert float(match.group(1)) >= 0.98, result.stdout
    assert "Token match: True" in result.stdout


class TestExaone45Conversion:
    @pytest.fixture(scope="class")
    def toy_model_path(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        model_dir = tmp_path_factory.mktemp("exaone45_toy") / "exaone45_toy"
        config = Exaone4_5_Config(**_EXAONE45_CONFIG)
        config.dtype = torch.bfloat16
        config.text_config.dtype = torch.bfloat16
        config.vision_config.dtype = torch.bfloat16
        model = Exaone4_5_ForConditionalGeneration(config).to(dtype=torch.bfloat16)
        model.save_pretrained(model_dir, safe_serialization=True)
        _save_tokenizer(model_dir)
        return str(model_dir)

    def test_hf_megatron_hf_roundtrip(self, toy_model_path: str, tmp_path: Path) -> None:
        _assert_roundtrip(toy_model_path, tmp_path / "exaone45_roundtrip")

    def test_forward_equivalence(self, toy_model_path: str) -> None:
        _assert_forward_equivalence(toy_model_path, "Exaone4_5_ForConditionalGeneration")


class TestExaoneMoeConversion:
    @pytest.fixture(scope="class")
    def toy_model_path(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        model_dir = tmp_path_factory.mktemp("exaone_moe_toy") / "exaone_moe_toy"
        config = ExaoneMoeConfig(**_EXAONE_MOE_CONFIG)
        config.dtype = torch.bfloat16
        model = ExaoneMoeForCausalLM(config).to(dtype=torch.bfloat16)
        model.save_pretrained(model_dir, safe_serialization=True)
        _save_tokenizer(model_dir)
        return str(model_dir)

    def test_hf_megatron_hf_roundtrip(self, toy_model_path: str, tmp_path: Path) -> None:
        _assert_roundtrip(toy_model_path, tmp_path / "exaone_moe_roundtrip")

    def test_forward_equivalence(self, toy_model_path: str) -> None:
        _assert_forward_equivalence(toy_model_path, "ExaoneMoeForCausalLM")
