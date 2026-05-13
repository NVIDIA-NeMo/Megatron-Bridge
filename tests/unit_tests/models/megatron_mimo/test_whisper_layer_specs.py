# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for whisper_layer_specs.py.

Pins the `attn_mask_type == AttnMaskType.no_mask` invariant: Whisper's encoder
is bidirectional, and a future regression that flips this to `causal` would
silently destroy audio quality. Module is loaded directly from disk because
it lives in `examples/`, not the installed package. Skipped automatically
when megatron.core's TE extensions aren't available.
"""

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("megatron.core.extensions.transformer_engine")

from megatron.core.transformer.enums import AttnMaskType


SPECS_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "megatron_mimo"
    / "whisper"
    / "whisper_layer_specs.py"
)


@pytest.fixture(scope="module")
def specs():
    spec = importlib.util.spec_from_file_location("whisper_layer_specs_under_test", SPECS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
class TestWhisperLayerSpecs:
    def test_te_spec_attn_mask_is_no_mask(self, specs):
        spec = specs.get_whisper_layer_with_transformer_engine_spec()
        assert spec.submodules.self_attention.params["attn_mask_type"] == AttnMaskType.no_mask

    def test_local_spec_attn_mask_is_no_mask(self, specs):
        spec = specs.get_whisper_layer_local_spec()
        assert spec.submodules.self_attention.params["attn_mask_type"] == AttnMaskType.no_mask

    def test_te_and_local_specs_use_different_linear_implementations(self, specs):
        """Sanity check that the TE/local toggle actually swaps the linear-layer types."""
        te = specs.get_whisper_layer_with_transformer_engine_spec()
        local = specs.get_whisper_layer_local_spec()
        te_qkv = te.submodules.self_attention.submodules.linear_qkv
        local_qkv = local.submodules.self_attention.submodules.linear_qkv
        assert te_qkv is not local_qkv
