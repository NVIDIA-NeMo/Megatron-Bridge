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

"""Functional tests for verify_adapter.py with a toy Qwen3 model.

Tests the full adapter verification pipeline:
1. CPU mode — PEFT-only check (no Megatron checkpoint)
2. CPU mode — full verification (PEFT + Megatron checkpoint comparison)
3. GPU mode — TP=2 multi-GPU verification via torchrun
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.model_load_save import temporary_distributed_context


peft = pytest.importorskip("peft", reason="peft library not installed")


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent

HF_QWEN3_TOY_CONFIG = {
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 1536,
    "initializer_range": 0.02,
    "intermediate_size": 8960,
    "max_position_embeddings": 8192,
    "num_attention_heads": 12,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "use_cache": True,
    "vocab_size": 151936,
}

LORA_DIM = 4
LORA_ALPHA = 8
LORA_TARGET_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen3_toy_model_dir(tmp_path_factory):
    """Create a tiny Qwen3 HF model on disk."""
    model_dir = tmp_path_factory.mktemp("qwen3_verify_toy") / "qwen3_toy"
    config = Qwen3Config(**HF_QWEN3_TOY_CONFIG)
    config.torch_dtype = torch.float32

    model = Qwen3ForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.save_pretrained(model_dir)

    return str(model_dir)


@pytest.fixture(scope="module")
def adapter_and_checkpoint(qwen3_toy_model_dir, tmp_path_factory):
    """Build Megatron model with LoRA, save HF adapter and Megatron PEFT checkpoint.

    Returns (adapter_dir, megatron_ckpt_dir) paths.
    """
    from megatron.core import dist_checkpointing

    from megatron.bridge.training.checkpointing import (
        _generate_model_state_dict,
        apply_peft_adapter_filter_to_state_dict,
    )

    adapter_dir = tmp_path_factory.mktemp("qwen3_verify_adapter") / "adapter"
    ckpt_dir = tmp_path_factory.mktemp("qwen3_verify_ckpt") / "iter_0000001"

    bridge = AutoBridge.from_hf_pretrained(qwen3_toy_model_dir)
    lora = LoRA(target_modules=LORA_TARGET_MODULES, dim=LORA_DIM, alpha=LORA_ALPHA, dropout=0.0)

    provider = bridge.to_megatron_provider(load_weights=True)
    provider.pipeline_dtype = torch.float32
    provider.params_dtype = torch.float32
    provider.finalize()
    provider.register_pre_wrap_hook(lambda chunks: lora(chunks, training=False))

    with temporary_distributed_context(backend="gloo"):
        model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
            init_model_with_meta_device=False,
        )

        torch.manual_seed(42)
        for name, param in model[0].named_parameters():
            if "adapter" in name:
                param.data.normal_(0, 0.02)

        bridge.save_hf_adapter(
            model,
            path=adapter_dir,
            peft_config=lora,
            base_model_name_or_path=qwen3_toy_model_dir,
        )

        sharded_sd = _generate_model_state_dict(model, {})
        adapter_sd = apply_peft_adapter_filter_to_state_dict(sharded_sd, lora)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dist_checkpointing.save(adapter_sd, str(ckpt_dir))

    return str(adapter_dir), str(ckpt_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVerifyAdapter:
    """End-to-end tests for examples/conversion/adapter/verify_adapter.py."""

    def test_cpu_peft_only(self, qwen3_toy_model_dir, adapter_and_checkpoint):
        """verify_adapter.py --cpu without --lora-checkpoint (PEFT-only check)."""
        adapter_dir, _ = adapter_and_checkpoint

        cmd = [
            sys.executable,
            "examples/conversion/adapter/verify_adapter.py",
            "--hf-model-path",
            qwen3_toy_model_dir,
            "--hf-adapter-path",
            adapter_dir,
            "--cpu",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
        assert result.returncode == 0, f"verify_adapter.py (CPU, PEFT-only) failed:\n{result.stderr}"
        assert "PASSED" in result.stdout

    def test_cpu_full_verification(self, qwen3_toy_model_dir, adapter_and_checkpoint):
        """verify_adapter.py --cpu with --lora-checkpoint (full Megatron cross-check)."""
        adapter_dir, ckpt_dir = adapter_and_checkpoint

        cmd = [
            sys.executable,
            "examples/conversion/adapter/verify_adapter.py",
            "--hf-model-path",
            qwen3_toy_model_dir,
            "--hf-adapter-path",
            adapter_dir,
            "--lora-checkpoint",
            ckpt_dir,
            "--cpu",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
        assert result.returncode == 0, f"verify_adapter.py (CPU, full) failed:\n{result.stderr}"
        assert "PASSED: adapter export is correct" in result.stdout

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_gpu_parallelism(self, qwen3_toy_model_dir, adapter_and_checkpoint, tp, pp, test_name):
        """verify_adapter.py via torchrun with TP/PP parallelism."""
        adapter_dir, ckpt_dir = adapter_and_checkpoint

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "examples/conversion/adapter/verify_adapter.py",
            "--hf-model-path",
            qwen3_toy_model_dir,
            "--hf-adapter-path",
            adapter_dir,
            "--lora-checkpoint",
            ckpt_dir,
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
        assert result.returncode == 0, (
            f"verify_adapter.py (GPU, {test_name}) failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "PASSED: adapter export is correct" in result.stdout
