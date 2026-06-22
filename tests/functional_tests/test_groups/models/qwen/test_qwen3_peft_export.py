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

"""Functional test for Qwen3 LoRA adapter export to HuggingFace PEFT format.

This test creates a toy Qwen3 model, attaches LoRA adapters, materializes
via AutoBridge, exports to HF PEFT format, and verifies that:

1. The output directory contains ``adapter_config.json`` + ``adapter_model.safetensors``.
2. The config is valid JSON with the expected LoRA hyper-parameters.
3. The safetensors file contains lora_A / lora_B weight pairs for every
   target module.
4. The adapter can be loaded by ``peft.PeftModel.from_pretrained`` (when the
   peft library is available).
5. The PEFT model produces different logits from the base model (adapter has
   effect).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.model_load_save import temporary_distributed_context


HF_QWEN3_TOY_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 1536,
    "initializer_range": 0.02,
    "intermediate_size": 8960,
    "max_position_embeddings": 8192,
    "model_type": "qwen2",
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

EXPECTED_HF_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


@pytest.fixture(scope="module")
def qwen3_toy_model_dir(tmp_path_factory):
    """Create a tiny Qwen3 HF model on disk for conversion tests."""
    model_dir = tmp_path_factory.mktemp("qwen3_peft_toy") / "qwen3_toy"
    config = Qwen3Config(**HF_QWEN3_TOY_CONFIG)
    config.torch_dtype = torch.float32

    model = Qwen3ForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)

    with open(model_dir / "config.json", "w") as f:
        json.dump(HF_QWEN3_TOY_CONFIG, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.save_pretrained(model_dir)

    return str(model_dir)


@pytest.fixture(scope="module")
def exported_adapter_dir(qwen3_toy_model_dir, tmp_path_factory):
    """Export LoRA adapter weights via AutoBridge and return the output path."""
    output_dir = tmp_path_factory.mktemp("qwen3_adapter_out") / "adapter"

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

        # LoRA B is zero-initialized by default, so the adapter would have no
        # effect.  Fill adapter weights with small random values so downstream
        # tests can verify the adapter actually changes model outputs.
        torch.manual_seed(42)
        for name, param in model[0].named_parameters():
            if "adapter" in name:
                param.data.normal_(0, 0.02)

        bridge.save_hf_adapter(
            model,
            path=output_dir,
            peft_config=lora,
            base_model_name_or_path=qwen3_toy_model_dir,
        )

    return str(output_dir)


class TestQwen3PeftExport:
    """Functional tests for Qwen3 LoRA adapter export to HuggingFace PEFT format."""

    def test_output_files_exist(self, exported_adapter_dir):
        """adapter_config.json and adapter_model.safetensors must be present."""
        adapter_dir = Path(exported_adapter_dir)
        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "adapter_model.safetensors").exists()

    def test_adapter_config_contents(self, exported_adapter_dir):
        """Verify adapter_config.json has correct LoRA hyper-parameters."""
        with open(Path(exported_adapter_dir) / "adapter_config.json") as f:
            cfg = json.load(f)

        assert cfg["peft_type"] == "LORA"
        assert cfg["r"] == LORA_DIM
        assert cfg["lora_alpha"] == LORA_ALPHA
        assert cfg["task_type"] == "CAUSAL_LM"
        assert cfg["use_dora"] is False
        assert cfg["inference_mode"] is True
        assert cfg["bias"] == "none"

        actual_modules = set(cfg["target_modules"])
        assert actual_modules == EXPECTED_HF_TARGET_MODULES, (
            f"target_modules mismatch: got {actual_modules}, expected {EXPECTED_HF_TARGET_MODULES}"
        )

    def test_safetensors_weight_pairs(self, exported_adapter_dir):
        """Every target module must have both lora_A and lora_B weights."""
        from safetensors.torch import load_file

        weights = load_file(str(Path(exported_adapter_dir) / "adapter_model.safetensors"))
        assert len(weights) > 0, "No weights found in adapter_model.safetensors"

        lora_a_modules: set[str] = set()
        lora_b_modules: set[str] = set()
        for name in weights:
            if ".lora_A.weight" in name:
                base = name.split(".lora_A.weight")[0]
                lora_a_modules.add(base)
            elif ".lora_B.weight" in name:
                base = name.split(".lora_B.weight")[0]
                lora_b_modules.add(base)

        assert lora_a_modules, "No lora_A weights found"
        assert lora_a_modules == lora_b_modules, (
            f"lora_A and lora_B module sets differ: "
            f"only_A={lora_a_modules - lora_b_modules}, only_B={lora_b_modules - lora_a_modules}"
        )

    def test_weight_shapes(self, exported_adapter_dir):
        """lora_A should have shape (r, in_features), lora_B should have (out_features, r)."""
        from safetensors.torch import load_file

        weights = load_file(str(Path(exported_adapter_dir) / "adapter_model.safetensors"))

        for name, tensor in weights.items():
            if ".lora_A.weight" in name:
                assert tensor.shape[0] == LORA_DIM, f"{name}: expected dim-0 == {LORA_DIM}, got {tensor.shape[0]}"
            elif ".lora_B.weight" in name:
                assert tensor.shape[1] == LORA_DIM, f"{name}: expected dim-1 == {LORA_DIM}, got {tensor.shape[1]}"

    def test_all_layers_have_adapters(self, exported_adapter_dir):
        """Both layers of the toy model should have adapter weights."""
        from safetensors.torch import load_file

        weights = load_file(str(Path(exported_adapter_dir) / "adapter_model.safetensors"))
        layer_ids = set()
        for name in weights:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_ids.add(int(parts[i + 1]))

        assert layer_ids == {0, 1}, f"Expected adapters in layers 0 and 1, found layers {layer_ids}"

    def test_peft_library_loads_adapter(self, qwen3_toy_model_dir, exported_adapter_dir):
        """The exported adapter must be loadable by HuggingFace peft library."""
        peft = pytest.importorskip("peft", reason="peft library not installed")

        base = Qwen3ForCausalLM.from_pretrained(qwen3_toy_model_dir, torch_dtype=torch.float32)
        peft_model = peft.PeftModel.from_pretrained(base, exported_adapter_dir)
        peft_model.eval()

        assert peft_model is not None
        assert hasattr(peft_model, "base_model")

    def test_adapter_changes_logits(self, qwen3_toy_model_dir, exported_adapter_dir):
        """PEFT model logits must differ from the base model (adapter has effect)."""
        peft = pytest.importorskip("peft", reason="peft library not installed")

        tokenizer = AutoTokenizer.from_pretrained(qwen3_toy_model_dir)
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")

        base_model = Qwen3ForCausalLM.from_pretrained(qwen3_toy_model_dir, torch_dtype=torch.float32)
        base_model.eval()
        with torch.no_grad():
            base_logits = base_model(**inputs).logits[0, -1, :]

        peft_base = Qwen3ForCausalLM.from_pretrained(qwen3_toy_model_dir, torch_dtype=torch.float32)
        peft_model = peft.PeftModel.from_pretrained(peft_base, exported_adapter_dir)
        peft_model.eval()
        with torch.no_grad():
            peft_logits = peft_model(**inputs).logits[0, -1, :]

        max_diff = (peft_logits.float() - base_logits.float()).abs().max().item()
        assert max_diff > 1e-6, (
            f"PEFT logits are identical to base model (max diff={max_diff:.2e}). "
            "Adapter weights may not have loaded correctly."
        )
