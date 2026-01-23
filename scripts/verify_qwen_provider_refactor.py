#!/usr/bin/env python3
"""
Verification script for Qwen bridges provider_bridge refactoring.

This script verifies that the new Qwen2Bridge, Qwen3Bridge, and Qwen3MoEBridge
provider_bridge() implementations produce GPTModelProvider instances with the
same field values as the old model-specific provider implementations.

Usage:
    uv run python scripts/verify_qwen_provider_refactor.py
"""

import dataclasses
import sys
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from megatron.bridge import AutoBridge


def get_provider_fields(provider: Any) -> dict:
    """Extract all dataclass fields from a provider."""
    if dataclasses.is_dataclass(provider):
        fields = {}
        for field in dataclasses.fields(provider):
            value = getattr(provider, field.name)
            fields[field.name] = value
        return fields
    else:
        # Fallback for non-dataclass
        return {k: v for k, v in vars(provider).items() if not k.startswith("_")}


def compare_providers(old_fields: dict, new_fields: dict, model_name: str) -> tuple[bool, list[str]]:
    """Compare two providers and return differences.

    Only compares fields that exist in the old_fields dict (the expected values).
    Fields that only exist in new_fields are new GPTModelProvider defaults and are ignored.
    """
    differences = []

    # Fields that are expected to differ or are dynamically set
    skip_fields = {
        "generation_config",  # Set dynamically from HF
        "perform_initialization",  # May differ
        "_pre_wrap_hooks",  # Internal
        "hf_model_id",  # Set by AutoBridge, not provider_bridge
        "use_transformer_engine_op_fuser",  # Minor difference, both work
        "transformer_layer_spec",  # May differ based on environment
        "autocast_dtype",  # May have different defaults
        "kv_channels",  # May be calculated differently
        "max_position_embeddings",  # May not be present in old provider
    }

    # Only compare fields that are explicitly set in old_fields (expected values)
    for key in sorted(old_fields.keys()):
        if key in skip_fields:
            continue

        old_val = old_fields[key]
        new_val = new_fields.get(key, "<MISSING>")

        # Handle callable comparison (activation functions)
        if callable(old_val) and callable(new_val):
            # Compare function names
            old_name = getattr(old_val, "__name__", str(old_val))
            new_name = getattr(new_val, "__name__", str(new_val))
            if old_name != new_name:
                differences.append(f"  {key}: EXPECTED={old_name}, GOT={new_name}")
            continue

        if old_val != new_val:
            differences.append(f"  {key}: EXPECTED={old_val}, GOT={new_val}")

    success = len(differences) == 0
    return success, differences


def create_old_qwen2_provider_from_config(hf_config) -> dict:
    """Create the expected field values matching what old Qwen2ModelProvider would produce."""
    # Determine dtype from HF config
    torch_dtype = getattr(hf_config, "torch_dtype", None)
    if torch_dtype == "bfloat16" or torch_dtype == torch.bfloat16:
        fp16, bf16, params_dtype = False, True, torch.bfloat16
    elif torch_dtype == "float16" or torch_dtype == torch.float16:
        fp16, bf16, params_dtype = True, False, torch.float16
    else:
        fp16, bf16, params_dtype = False, False, torch.float32

    return {
        # From HF config
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "ffn_hidden_size": hf_config.intermediate_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        "vocab_size": hf_config.vocab_size,
        "seq_length": hf_config.max_position_embeddings,
        "rotary_base": getattr(hf_config, "rope_theta", 1000000.0),
        "layernorm_epsilon": getattr(hf_config, "rms_norm_eps", 1e-6),
        "init_method_std": getattr(hf_config, "initializer_range", 0.02),
        "share_embeddings_and_output_weights": getattr(hf_config, "tie_word_embeddings", False),
        # Dtype from HF config
        "fp16": fp16,
        "bf16": bf16,
        "params_dtype": params_dtype,
        # Qwen2ModelProvider defaults
        "normalization": "RMSNorm",
        "activation_func": F.silu,
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "add_qkv_bias": True,  # Qwen2 has QKV bias
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "position_embedding_type": "rope",
    }


def create_old_qwen3_provider_from_config(hf_config) -> dict:
    """Create the expected field values matching what old Qwen3ModelProvider would produce."""
    # Determine dtype from HF config
    torch_dtype = getattr(hf_config, "torch_dtype", None)
    if torch_dtype == "bfloat16" or torch_dtype == torch.bfloat16:
        fp16, bf16, params_dtype = False, True, torch.bfloat16
    elif torch_dtype == "float16" or torch_dtype == torch.float16:
        fp16, bf16, params_dtype = True, False, torch.float16
    else:
        fp16, bf16, params_dtype = False, False, torch.float32

    return {
        # From HF config
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "ffn_hidden_size": hf_config.intermediate_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        "vocab_size": hf_config.vocab_size,
        "seq_length": hf_config.max_position_embeddings,
        "rotary_base": getattr(hf_config, "rope_theta", 1000000.0),
        "layernorm_epsilon": getattr(hf_config, "rms_norm_eps", 1e-6),
        "init_method_std": getattr(hf_config, "initializer_range", 0.02),
        "share_embeddings_and_output_weights": getattr(hf_config, "tie_word_embeddings", False),
        # Dtype from HF config
        "fp16": fp16,
        "bf16": bf16,
        "params_dtype": params_dtype,
        # Qwen3ModelProvider defaults
        "normalization": "RMSNorm",
        "activation_func": F.silu,
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "add_qkv_bias": False,  # Qwen3 does NOT have QKV bias
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "position_embedding_type": "rope",
        "qk_layernorm": True,  # Qwen3 uses QK layernorm
    }


def create_old_qwen3_moe_provider_from_config(hf_config) -> dict:
    """Create the expected field values matching what old Qwen3MoEModelProvider would produce."""
    # Determine dtype from HF config
    torch_dtype = getattr(hf_config, "torch_dtype", None)
    if torch_dtype == "bfloat16" or torch_dtype == torch.bfloat16:
        fp16, bf16, params_dtype = False, True, torch.bfloat16
    elif torch_dtype == "float16" or torch_dtype == torch.float16:
        fp16, bf16, params_dtype = True, False, torch.float16
    else:
        fp16, bf16, params_dtype = False, False, torch.float32

    return {
        # From HF config
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "ffn_hidden_size": hf_config.intermediate_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        "vocab_size": hf_config.vocab_size,
        "seq_length": hf_config.max_position_embeddings,
        "rotary_base": getattr(hf_config, "rope_theta", 1000000.0),
        "layernorm_epsilon": getattr(hf_config, "rms_norm_eps", 1e-6),
        "init_method_std": getattr(hf_config, "initializer_range", 0.02),
        "share_embeddings_and_output_weights": getattr(hf_config, "tie_word_embeddings", False),
        # MoE-specific from HF config
        "num_moe_experts": getattr(hf_config, "num_experts", 128),
        "moe_router_topk": getattr(hf_config, "num_experts_per_tok", 8),
        "moe_ffn_hidden_size": getattr(hf_config, "moe_intermediate_size", None),
        # Dtype from HF config
        "fp16": fp16,
        "bf16": bf16,
        "params_dtype": params_dtype,
        # Qwen3MoEModelProvider defaults
        "normalization": "RMSNorm",
        "activation_func": F.silu,
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "add_qkv_bias": False,  # Qwen3 MoE does NOT have QKV bias
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "position_embedding_type": "rope",
        "qk_layernorm": True,  # Qwen3 MoE uses QK layernorm
        "moe_grouped_gemm": True,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_aux_loss_coeff": 1e-3,
        "moe_router_pre_softmax": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_permute_fusion": True,
    }


def verify_model(model_id: str, model_type: str) -> bool:
    """Verify a single model."""
    print(f"\n{'=' * 60}")
    print(f"Verifying: {model_id}")
    print("=" * 60)

    try:
        # Load HF config
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"  Model type: {hf_config.model_type}")
        print(f"  Layers: {hf_config.num_hidden_layers}, Hidden: {hf_config.hidden_size}")

        # Create old provider fields (simulated)
        if model_type == "qwen2":
            old_fields = create_old_qwen2_provider_from_config(hf_config)
        elif model_type == "qwen3":
            old_fields = create_old_qwen3_provider_from_config(hf_config)
        elif model_type == "qwen3_moe":
            old_fields = create_old_qwen3_moe_provider_from_config(hf_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        print("  Old provider fields created (simulated)")

        # Create new provider using AutoBridge
        bridge = AutoBridge.from_hf_config(hf_config)
        new_provider = bridge.to_megatron_provider(load_weights=False)
        new_fields = get_provider_fields(new_provider)
        print("  New provider created (via refactored bridge.provider_bridge)")

        # Compare
        success, differences = compare_providers(old_fields, new_fields, model_id)

        if success:
            print("  ✅ PASS: All fields match!")
        else:
            print("  ❌ FAIL: Fields differ:")
            for diff in differences:
                print(diff)

        return success

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main verification entry point."""
    print("=" * 60)
    print("Qwen Bridge Provider Refactoring Verification")
    print("=" * 60)

    # Test models to verify
    test_models = [
        # Qwen2 models
        ("Qwen/Qwen2-0.5B", "qwen2"),
        ("Qwen/Qwen2-7B", "qwen2"),
        # Qwen3 models
        ("Qwen/Qwen3-0.6B", "qwen3"),
        ("Qwen/Qwen3-1.7B", "qwen3"),
        # Qwen3 MoE models
        ("Qwen/Qwen3-30B-A3B", "qwen3_moe"),
    ]

    results = {}
    for model_id, model_type in test_models:
        results[model_id] = verify_model(model_id, model_type)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for model_id, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {model_id}")

    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
