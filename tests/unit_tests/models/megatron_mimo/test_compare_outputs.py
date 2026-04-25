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
