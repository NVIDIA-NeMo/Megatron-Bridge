# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for OmniModal DP utilities."""

import pytest
import torch
import torch.distributed as dist

from megatron.bridge.data.omni_modal.dp_utils import get_omni_modal_dp_info, slice_batch_for_omni_modal
from megatron.bridge.models.omni_modal.omni_modal_config import ModuleParallelismConfig, OmniModalParallelismConfig


class FakePG:
    """Fake process group for testing."""

    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class FakeGrid:
    """Fake HyperCommGrid for testing."""

    def __init__(self, rank_offset: int, size: int, dp_rank: int, dp_size: int, pp_rank: int, pp_size: int):
        self.rank_offset = rank_offset
        self.size = size
        self._pgs = {
            ("dp",): FakePG(dp_rank, dp_size),
            ("pp",): FakePG(pp_rank, pp_size),
        }

    def get_pg(self, dims):
        return self._pgs[tuple(dims)]


def _make_omni_modal_cfg() -> OmniModalParallelismConfig:
    """Create test OmniModal config for heterogeneous deployment."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=4),
    }
    return OmniModalParallelismConfig(
        module_parallelisms=module_parallelisms,
    )


def test_get_omni_modal_dp_info_encoder_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, first PP stage."""
    omni_modal_cfg = _make_omni_modal_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_omni_modal_dp_info(omni_modal_cfg, grids)

    assert loader_module == "vision"
    assert dp_rank == 0
    assert dp_size == 2
    assert needs_data is True  # First PP stage


def test_get_omni_modal_dp_info_encoder_non_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, not first PP stage."""
    omni_modal_cfg = _make_omni_modal_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=1, pp_size=2),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_omni_modal_dp_info(omni_modal_cfg, grids)

    assert loader_module == "vision"
    assert needs_data is False  # Not first PP stage


def test_get_omni_modal_dp_info_llm_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, first PP stage."""
    omni_modal_cfg = _make_omni_modal_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 4)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_omni_modal_dp_info(omni_modal_cfg, grids)

    assert loader_module == "language"
    assert needs_data is True  # First PP stage


def test_get_omni_modal_dp_info_llm_last_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, last PP stage."""
    omni_modal_cfg = _make_omni_modal_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 5)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=1, dp_size=4, pp_rank=1, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_omni_modal_dp_info(omni_modal_cfg, grids)

    assert loader_module == "language"
    assert needs_data is True  # Last PP stage


def test_get_omni_modal_dp_info_non_participating_rank(monkeypatch):
    """Test heterogeneous mode, rank not in any module."""
    omni_modal_cfg = _make_omni_modal_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 10)  # Outside all grids

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_omni_modal_dp_info(omni_modal_cfg, grids)

    assert needs_data is False
    assert loader_module == "language"  # Default to LLM


# ---------------------------------------------------------------------------
# Tests: slice_batch_for_omni_modal
# ---------------------------------------------------------------------------


class TestSliceBatchForOmniModal:
    """Test per-module DP batch slicing."""

    def test_dp_size_1_returns_original(self):
        batch = {"tokens": torch.randn(4, 2048)}
        result = slice_batch_for_omni_modal(batch, dp_rank=0, dp_size=1)
        assert result is batch  # no copy, same object

    def test_slices_tensors_along_batch_dim(self):
        tokens = torch.arange(12).reshape(4, 3)  # [4, 3]
        batch = {"tokens": tokens}

        s0 = slice_batch_for_omni_modal(batch, dp_rank=0, dp_size=2)
        s1 = slice_batch_for_omni_modal(batch, dp_rank=1, dp_size=2)

        assert s0["tokens"].shape == (2, 3)
        assert s1["tokens"].shape == (2, 3)
        torch.testing.assert_close(s0["tokens"], tokens[0:2])
        torch.testing.assert_close(s1["tokens"], tokens[2:4])

    def test_slices_4_way(self):
        pixels = torch.randn(8, 3, 224, 224)  # 8 images
        batch = {"pixel_values": pixels}

        for rank in range(4):
            sliced = slice_batch_for_omni_modal(batch, dp_rank=rank, dp_size=4)
            assert sliced["pixel_values"].shape == (2, 3, 224, 224)
            torch.testing.assert_close(sliced["pixel_values"], pixels[rank * 2 : rank * 2 + 2])

    def test_recurses_into_nested_dicts(self):
        batch = {
            "tokens": torch.randn(4, 2048),
            "modality_inputs": {
                "vision": {
                    "pixel_values": torch.randn(4, 3, 224, 224),
                }
            },
        }
        sliced = slice_batch_for_omni_modal(batch, dp_rank=1, dp_size=2)

        assert sliced["tokens"].shape[0] == 2
        assert sliced["modality_inputs"]["vision"]["pixel_values"].shape[0] == 2

    def test_preserves_non_tensor_values(self):
        batch = {
            "tokens": torch.randn(4, 10),
            "metadata": "some_string",
            "flags": 42,
        }
        sliced = slice_batch_for_omni_modal(batch, dp_rank=0, dp_size=2)

        assert sliced["metadata"] == "some_string"
        assert sliced["flags"] == 42
        assert sliced["tokens"].shape[0] == 2

    def test_slices_lists(self):
        batch = {
            "tokens": torch.randn(4, 10),
            "filenames": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        }
        sliced = slice_batch_for_omni_modal(batch, dp_rank=1, dp_size=2)

        assert sliced["filenames"] == ["c.jpg", "d.jpg"]

    def test_raises_on_indivisible_batch(self):
        batch = {"tokens": torch.randn(5, 10)}  # 5 not divisible by 2
        with pytest.raises(ValueError, match="not divisible"):
            slice_batch_for_omni_modal(batch, dp_rank=0, dp_size=2)

    def test_none_batch_passthrough(self):
        """None batch should not crash (forward_step passes None for non-data ranks)."""
        # slice_batch_for_omni_modal expects a dict; None is handled by caller.
        # This test documents that dp_size=1 early-return handles the common case.
        result = slice_batch_for_omni_modal({}, dp_rank=0, dp_size=1)
        assert result == {}
