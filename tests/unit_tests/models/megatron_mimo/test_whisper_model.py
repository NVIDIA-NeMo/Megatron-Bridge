# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for whisper_model.py helpers (CPU-only).

Importing whisper_model.py directly triggers `from megatron.core...` imports
that pull in extensions which may not be available in a CPU-only test env.
The pure helper `_sinusoidal_position_embedding` is extracted via AST so it
can be exercised in isolation.
"""

import ast
import math
from pathlib import Path

import pytest
import torch


WHISPER_MODEL_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "models" / "megatron_mimo" / "whisper" / "whisper_model.py"
)


def _extract_function(source_path: Path, fn_name: str):
    """Compile and return a single top-level function from a Python file.

    Avoids running the module's import side-effects (heavy ML deps).
    """
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            ns: dict = {"math": math, "torch": torch}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, str(source_path), "exec"), ns)
            return ns[fn_name]
    raise RuntimeError(f"{fn_name} not found in {source_path}")


@pytest.fixture(scope="module")
def sinusoidal():
    return _extract_function(WHISPER_MODEL_PATH, "_sinusoidal_position_embedding")


def _hf_reference_sinusoids(max_len: int, d_model: int) -> torch.Tensor:
    """Faithful reproduction of HuggingFace Whisper's `sinusoids()` helper.

    Mirrors transformers/models/whisper/modeling_whisper.py::sinusoids so the
    Megatron-side embedding can be compared element-wise.
    """
    half = d_model // 2
    log_timescale_increment = math.log(10000.0) / (half - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(half).float())
    scaled_time = torch.arange(max_len).float().unsqueeze(1) * inv_timescales.unsqueeze(0)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@pytest.mark.unit
class TestSinusoidalPositionEmbedding:
    def test_matches_hf_reference(self, sinusoidal):
        assert torch.allclose(sinusoidal(8, 4), _hf_reference_sinusoids(8, 4))

    @pytest.mark.parametrize("max_len,d_model", [(1500, 512), (1500, 768), (32, 16)])
    def test_shape(self, sinusoidal, max_len, d_model):
        assert sinusoidal(max_len, d_model).shape == (max_len, d_model)

    def test_layout_is_concatenated_not_interleaved(self, sinusoidal):
        """First half is sines, second half cosines — guards against the [sin, cos, sin, cos] variant."""
        max_len, d_model = 16, 8
        half = d_model // 2
        out = sinusoidal(max_len, d_model)
        # sin(0) == 0 → first row of sin half is all zeros.
        assert torch.allclose(out[0, :half], torch.zeros(half))
        # cos(0) == 1 → first row of cos half is all ones.
        assert torch.allclose(out[0, half:], torch.ones(half))

    def test_dtype_is_default_float(self, sinusoidal):
        assert sinusoidal(8, 4).dtype == torch.get_default_dtype()

    def test_no_grad_required(self, sinusoidal):
        """Result is a leaf tensor with no autograd history (frozen embedding)."""
        assert sinusoidal(8, 4).requires_grad is False

    def test_determinism(self, sinusoidal):
        assert torch.equal(sinusoidal(32, 16), sinusoidal(32, 16))

    def test_finite_values(self, sinusoidal):
        out = sinusoidal(1500, 512)
        assert torch.isfinite(out).all()
        # sin/cos are bounded by ±1.
        assert out.abs().max().item() <= 1.0 + 1e-6
