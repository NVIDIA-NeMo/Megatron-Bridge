#!/usr/bin/env python3
"""
Verification script for Gemma bridge provider_bridge refactoring.

This script verifies that the Gemma bridges correctly create provider instances
with expected field values via MEGATRON_DEFAULTS and CONFIG_MAPPING.

Usage:
    uv run python scripts/verify_gemma_provider_refactor.py
"""

import dataclasses
import sys
from typing import Any

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
        return {k: v for k, v in vars(provider).items() if not k.startswith("_")}


def verify_model(model_id: str, expected_provider_type: str, expected_defaults: dict) -> bool:
    """Verify a single model."""
    print(f"\n{'=' * 60}")
    print(f"Verifying: {model_id}")
    print("=" * 60)

    try:
        # Load HF config
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"  Model type: {hf_config.model_type}")
        if hasattr(hf_config, "num_hidden_layers"):
            print(f"  Layers: {hf_config.num_hidden_layers}, Hidden: {hf_config.hidden_size}")

        # Create provider using AutoBridge
        bridge = AutoBridge.from_hf_config(hf_config)
        provider = bridge.to_megatron_provider(load_weights=False)
        provider_type = type(provider).__name__
        print(f"  Provider type: {provider_type}")

        # Check provider type
        if provider_type != expected_provider_type:
            print(f"  ❌ FAIL: Expected provider type {expected_provider_type}, got {provider_type}")
            return False

        # Check expected defaults
        fields = get_provider_fields(provider)
        differences = []

        for key, expected_val in expected_defaults.items():
            actual_val = fields.get(key, "<MISSING>")

            # Handle callable comparison
            if callable(expected_val) and callable(actual_val):
                expected_name = getattr(expected_val, "__name__", str(expected_val))
                actual_name = getattr(actual_val, "__name__", str(actual_val))
                if expected_name != actual_name:
                    differences.append(f"  {key}: EXPECTED={expected_name}, ACTUAL={actual_name}")
                continue

            if actual_val != expected_val:
                differences.append(f"  {key}: EXPECTED={expected_val}, ACTUAL={actual_val}")

        if differences:
            print("  ❌ FAIL: Fields differ from expected:")
            for diff in differences:
                print(diff)
            return False

        # Print key config values
        print(f"  normalization: {fields.get('normalization')}")
        print(f"  gated_linear_unit: {fields.get('gated_linear_unit')}")
        print(f"  position_embedding_type: {fields.get('position_embedding_type')}")
        print(f"  layernorm_zero_centered_gamma: {fields.get('layernorm_zero_centered_gamma')}")
        print(f"  share_embeddings_and_output_weights: {fields.get('share_embeddings_and_output_weights')}")

        print("  ✅ PASS: All expected fields match!")
        return True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main verification entry point."""
    print("=" * 60)
    print("Gemma Bridge Provider Refactoring Verification")
    print("=" * 60)

    from megatron.core.activations import fast_gelu
    from megatron.core.transformer.enums import AttnBackend

    # Expected defaults from MEGATRON_DEFAULTS
    gemma_defaults = {
        "normalization": "RMSNorm",
        "activation_func": fast_gelu,
        "gated_linear_unit": True,
        "position_embedding_type": "rope",
        "add_bias_linear": False,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "share_embeddings_and_output_weights": True,
        "layernorm_zero_centered_gamma": True,
        "attention_backend": AttnBackend.flash,
    }

    gemma2_defaults = {
        "normalization": "RMSNorm",
        "activation_func": fast_gelu,
        "gated_linear_unit": True,
        "position_embedding_type": "rope",
        "add_bias_linear": False,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "share_embeddings_and_output_weights": True,
        "layernorm_zero_centered_gamma": True,
        "gradient_accumulation_fusion": False,
    }

    # Gemma3 VL models use Gemma3VLModelProvider
    gemma3_vl_defaults = {
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        "position_embedding_type": "rope",
        "add_bias_linear": False,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "share_embeddings_and_output_weights": True,
        "layernorm_zero_centered_gamma": True,
    }

    # Test models to verify
    test_cases = [
        # Gemma 1
        ("google/gemma-2b", "GemmaModelProvider", gemma_defaults),
        ("google/gemma-7b", "GemmaModelProvider", gemma_defaults),
        # Gemma 2
        ("google/gemma-2-2b", "Gemma2ModelProvider", gemma2_defaults),
        ("google/gemma-2-9b", "Gemma2ModelProvider", gemma2_defaults),
        ("google/gemma-2-27b", "Gemma2ModelProvider", gemma2_defaults),
        # Gemma 3 VL (4b and larger are VL models)
        ("google/gemma-3-4b-it", "Gemma3VLModelProvider", gemma3_vl_defaults),
        ("google/gemma-3-12b-it", "Gemma3VLModelProvider", gemma3_vl_defaults),
        ("google/gemma-3-27b-it", "Gemma3VLModelProvider", gemma3_vl_defaults),
    ]

    results = {}
    for model_id, provider_type, expected_defaults in test_cases:
        results[model_id] = verify_model(model_id, provider_type, expected_defaults)

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
