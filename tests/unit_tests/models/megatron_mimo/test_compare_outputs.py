# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for verify_whisper_conversion.py::compare_outputs.

The full verify_whisper_conversion script requires CUDA + NCCL distributed
init, but `compare_outputs` is a pure tensor helper. It is extracted via AST
so the threshold logic can be exercised without importing the rest of the
script.
"""

import ast
from pathlib import Path

import pytest
import torch


VERIFY_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "models"
    / "megatron_mimo"
    / "whisper"
    / "verify_whisper_conversion.py"
)


@pytest.fixture(scope="module")
def compare_outputs():
    tree = ast.parse(VERIFY_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compare_outputs":
            ns: dict = {"torch": torch}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(VERIFY_PATH), "exec"), ns)
            return ns["compare_outputs"]
    raise RuntimeError("compare_outputs not found in verify_whisper_conversion.py")


@pytest.mark.unit
class TestCompareOutputs:
    """Threshold gates: mean_diff < 1e-2, max_diff < 1.0, cosine > 0.9999."""

    def test_identical_tensors_pass(self, compare_outputs):
        a = torch.randn(2, 5, 8)
        assert compare_outputs(a, a, label="identical") is True

    def test_single_spike_fails_max_diff_gate(self, compare_outputs):
        """One large outlier in an otherwise-equal pair: max_diff > 1.0 fails the gate."""
        torch.manual_seed(0)
        a = torch.randn(2, 5, 200)
        b = a.clone()
        b[0, 0, 0] += 5.0  # max_diff = 5; mean_diff ≈ 5/2000 = 2.5e-3 (passes); cos ≈ 1 (passes)
        assert compare_outputs(a, b, label="single-spike") is False

    def test_anti_aligned_tiny_magnitude_fails_cosine_gate(self, compare_outputs):
        """a vs -a with tiny magnitude: mean/max diffs pass but cos = -1 fails."""
        torch.manual_seed(0)
        a = torch.randn(2, 5, 8) * 1e-5
        b = -a
        assert compare_outputs(a, b, label="anti-aligned-tiny") is False

    def test_bulk_offset_fails(self, compare_outputs):
        """Constant offset adds bulk noise — fails mean-diff and/or cosine."""
        a = torch.randn(2, 5, 8)
        b = a + 0.1
        assert compare_outputs(a, b, label="bulk-offset") is False

    def test_small_perturbation_passes(self, compare_outputs):
        a = torch.randn(2, 5, 8) * 10
        b = a + torch.randn_like(a) * 1e-6
        assert compare_outputs(a, b, label="small-perturb") is True

    def test_bf16_inputs_supported(self, compare_outputs):
        a = torch.randn(2, 5, 8).to(torch.bfloat16)
        assert compare_outputs(a, a, label="bf16-identical") is True


# ---------------------------------------------------------------------------
# _make_whisper_config — pure HF-config → Megatron-config translator.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def make_whisper_config():
    """AST-extract `_make_whisper_config` from verify_whisper_conversion.py.

    Skipped when megatron.core's TransformerConfig isn't importable.
    """
    tc_mod = pytest.importorskip("megatron.core.transformer.transformer_config")
    tree = ast.parse(VERIFY_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_make_whisper_config":
            ns: dict = {"torch": torch, "TransformerConfig": tc_mod.TransformerConfig}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(VERIFY_PATH), "exec"), ns)
            return ns["_make_whisper_config"]
    raise RuntimeError("_make_whisper_config not found in verify_whisper_conversion.py")


def _hf_config(*, layers=4, d_model=128, ffn=512, heads=8):
    """Minimal HF-style config namespace with the attrs `_make_whisper_config` reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        encoder_layers=layers,
        d_model=d_model,
        encoder_ffn_dim=ffn,
        encoder_attention_heads=heads,
    )


@pytest.mark.unit
class TestMakeWhisperConfig:
    def test_dimensions_pulled_from_hf_config(self, make_whisper_config):
        cfg = make_whisper_config(_hf_config(layers=6, d_model=512, ffn=2048, heads=8), torch.bfloat16)
        assert cfg.num_layers == 6
        assert cfg.hidden_size == 512
        assert cfg.ffn_hidden_size == 2048
        assert cfg.num_attention_heads == 8

    def test_bf16_dtype_sets_bf16_flag_and_pipeline_dtype(self, make_whisper_config):
        cfg = make_whisper_config(_hf_config(), torch.bfloat16)
        assert cfg.bf16 is True
        assert cfg.pipeline_dtype == torch.bfloat16

    def test_fp32_dtype_disables_bf16(self, make_whisper_config):
        cfg = make_whisper_config(_hf_config(), torch.float32)
        assert cfg.bf16 is False
        assert cfg.pipeline_dtype == torch.float32

    def test_pinned_training_knobs(self, make_whisper_config):
        """Whisper's encoder needs bias on QKV, gelu activation, and zero dropout — pin those."""
        cfg = make_whisper_config(_hf_config(), torch.bfloat16)
        assert cfg.add_bias_linear is True
        assert cfg.add_qkv_bias is True
        assert cfg.hidden_dropout == 0.0
        assert cfg.attention_dropout == 0.0
        assert cfg.gated_linear_unit is False
        assert cfg.normalization == "LayerNorm"
        assert cfg.attention_softmax_in_fp32 is True
        assert cfg.activation_func is torch.nn.functional.gelu

    def test_use_cpu_initialization_enabled(self, make_whisper_config):
        """The verify script runs on CPU init so weight loading is deterministic."""
        cfg = make_whisper_config(_hf_config(), torch.bfloat16)
        assert cfg.use_cpu_initialization is True
        assert cfg.variable_seq_lengths is True
