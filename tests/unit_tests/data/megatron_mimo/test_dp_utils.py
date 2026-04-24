# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for MegatronMIMO DP utilities."""

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

from megatron.bridge.data.megatron_mimo.dp_utils import (
    _find_rank_modules,
    get_megatron_mimo_dp_info,
    get_megatron_mimo_sampling_info,
    slice_batch_for_megatron_mimo,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


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


def _make_megatron_mimo_cfg() -> MegatronMIMOParallelismConfig:
    """Create test MegatronMIMO config for heterogeneous deployment."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=4),
    }
    return MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )


def test_get_megatron_mimo_dp_info_encoder_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, first PP stage."""
    megatron_mimo_cfg = _make_megatron_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_megatron_mimo_dp_info(megatron_mimo_cfg, grids)

    assert loader_module == "vision"
    assert dp_rank == 0
    assert dp_size == 2
    assert needs_data is True  # First PP stage


def test_get_megatron_mimo_dp_info_encoder_non_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, not first PP stage."""
    megatron_mimo_cfg = _make_megatron_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=1, pp_size=2),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_megatron_mimo_dp_info(megatron_mimo_cfg, grids)

    assert loader_module == "vision"
    assert needs_data is False  # Not first PP stage


def test_get_megatron_mimo_dp_info_llm_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, first PP stage."""
    megatron_mimo_cfg = _make_megatron_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 4)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_megatron_mimo_dp_info(megatron_mimo_cfg, grids)

    assert loader_module == "language"
    assert needs_data is True  # First PP stage


def test_get_megatron_mimo_dp_info_llm_last_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, last PP stage."""
    megatron_mimo_cfg = _make_megatron_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 5)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=1, dp_size=4, pp_rank=1, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_megatron_mimo_dp_info(megatron_mimo_cfg, grids)

    assert loader_module == "language"
    assert needs_data is True  # Last PP stage


def test_get_megatron_mimo_dp_info_non_participating_rank(monkeypatch):
    """Test heterogeneous mode, rank not in any module."""
    megatron_mimo_cfg = _make_megatron_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 10)  # Outside all grids

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_megatron_mimo_dp_info(megatron_mimo_cfg, grids)

    assert needs_data is False
    assert loader_module == "language"  # Default to LLM


# ---------------------------------------------------------------------------
# Tests: multi-module rank membership (colocated mode)
# ---------------------------------------------------------------------------


def _make_colocated_grids(vision_dp: int = 8, llm_dp: int = 2, llm_tp: int = 4) -> dict:
    """Build fake grids for a colocated 8-GPU setup with a fan-in layout.

    Both grids span ranks [0, 8). Encoder has DP=8/TP=1, LLM has DP=llm_dp/TP=llm_tp.
    Process-group ranks/sizes on the fake grids are set to what rank 0 would see.
    """
    return {
        "vision": FakeGrid(0, 8, dp_rank=0, dp_size=vision_dp, pp_rank=0, pp_size=1),
        "language": FakeGrid(0, 8, dp_rank=0, dp_size=llm_dp, pp_rank=0, pp_size=1),
    }


def _make_colocated_cfg(vision_dp: int = 8, llm_dp: int = 2, llm_tp: int = 4) -> MegatronMIMOParallelismConfig:
    """Build a colocated parallelism config matching _make_colocated_grids."""
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "vision": ModuleParallelismConfig(
                tensor_model_parallel_size=1, data_parallel_size=vision_dp, rank_offset=0
            ),
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=llm_tp, data_parallel_size=llm_dp, rank_offset=0
            ),
        }
    )


def test_find_rank_modules_non_colocated_returns_single(monkeypatch):
    """Non-colocated: a rank belongs to exactly one module grid."""
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2),
        "language": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }
    result = _find_rank_modules(grids)
    assert list(result.keys()) == ["vision"]


def test_find_rank_modules_colocated_returns_all(monkeypatch):
    """Colocated: a rank belongs to every module grid."""
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    grids = _make_colocated_grids()
    result = _find_rank_modules(grids)
    assert set(result.keys()) == {"vision", "language"}


def test_find_rank_modules_non_participating(monkeypatch):
    """A rank outside every grid returns an empty dict."""
    monkeypatch.setattr(dist, "get_rank", lambda: 99)
    grids = _make_colocated_grids()
    assert _find_rank_modules(grids) == {}


def test_get_megatron_mimo_sampling_info_colocated_needs_data_union(monkeypatch):
    """Colocated: needs_data is True if ANY served module needs data on this rank.

    Previously, `_find_rank_module` returned whichever grid appeared first in the
    dict and could report needs_data=False on a rank where the other module
    actually needs data — a silent no-op training bug waiting to happen.
    """
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cfg = _make_colocated_cfg()
    # Encoder at PP=1 stage != 0 (would report False), LLM at PP stage 0 (True).
    # Iteration order puts vision first; the pre-fix code would have returned False.
    grids = {
        "vision": FakeGrid(0, 8, dp_rank=0, dp_size=8, pp_rank=1, pp_size=2),
        "language": FakeGrid(0, 8, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2),
    }
    _, _, needs_data = get_megatron_mimo_sampling_info(cfg, grids)
    assert needs_data is True


def test_get_megatron_mimo_dp_info_colocated_by_module_name(monkeypatch):
    """Colocated: explicit module_name selects that module's DP geometry."""
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cfg = _make_colocated_cfg()
    grids = _make_colocated_grids()

    vision_dp_rank, vision_dp_size, _, vision_name = get_megatron_mimo_dp_info(cfg, grids, module_name="vision")
    assert vision_name == "vision"
    assert vision_dp_size == 8  # encoder is fully DP-parallel

    llm_dp_rank, llm_dp_size, _, llm_name = get_megatron_mimo_dp_info(cfg, grids, module_name="language")
    assert llm_name == "language"
    assert llm_dp_size == 2  # LLM is TP4/DP2


def test_get_megatron_mimo_dp_info_colocated_module_name_not_on_rank(monkeypatch):
    """Explicit module_name for a module this rank doesn't serve returns the
    non-participating sentinel with the requested module name."""
    monkeypatch.setattr(dist, "get_rank", lambda: 99)
    cfg = _make_colocated_cfg()
    grids = _make_colocated_grids()

    dp_rank, dp_size, needs_data, name = get_megatron_mimo_dp_info(cfg, grids, module_name="vision")
    assert (dp_rank, dp_size, needs_data, name) == (0, 1, False, MIMO_LANGUAGE_MODULE_KEY)


def test_get_megatron_mimo_dp_info_colocated_no_module_name_defaults_to_language(monkeypatch, caplog):
    """Colocated: omitting module_name falls back to the language module and logs a warning."""
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cfg = _make_colocated_cfg()
    grids = _make_colocated_grids()

    with caplog.at_level("WARNING", logger="megatron.bridge.data.megatron_mimo.dp_utils"):
        _, dp_size, _, name = get_megatron_mimo_dp_info(cfg, grids)

    assert name == MIMO_LANGUAGE_MODULE_KEY
    assert dp_size == 2  # LLM DP
    assert any("multiple modules" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Tests: slice_batch_for_megatron_mimo
# ---------------------------------------------------------------------------


class TestSliceBatchForMegatronMIMO:
    """Test per-module DP batch slicing."""

    def test_dp_size_1_returns_original(self):
        batch = {"tokens": torch.randn(4, 2048)}
        result = slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=1)
        assert result is batch  # no copy, same object

    def test_slices_tensors_along_batch_dim(self):
        tokens = torch.arange(12).reshape(4, 3)  # [4, 3]
        batch = {"tokens": tokens}

        s0 = slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)
        s1 = slice_batch_for_megatron_mimo(batch, dp_rank=1, dp_size=2)

        assert s0["tokens"].shape == (2, 3)
        assert s1["tokens"].shape == (2, 3)
        torch.testing.assert_close(s0["tokens"], tokens[0:2])
        torch.testing.assert_close(s1["tokens"], tokens[2:4])

    def test_slices_4_way(self):
        pixels = torch.randn(8, 3, 224, 224)  # 8 images
        batch = {"pixel_values": pixels}

        for rank in range(4):
            sliced = slice_batch_for_megatron_mimo(batch, dp_rank=rank, dp_size=4)
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
        sliced = slice_batch_for_megatron_mimo(batch, dp_rank=1, dp_size=2)

        assert sliced["tokens"].shape[0] == 2
        assert sliced["modality_inputs"]["vision"]["pixel_values"].shape[0] == 2

    def test_preserves_non_tensor_values(self):
        batch = {
            "tokens": torch.randn(4, 10),
            "metadata": "some_string",
            "flags": 42,
        }
        sliced = slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)

        assert sliced["metadata"] == "some_string"
        assert sliced["flags"] == 42
        assert sliced["tokens"].shape[0] == 2

    def test_slices_lists(self):
        batch = {
            "tokens": torch.randn(4, 10),
            "filenames": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        }
        sliced = slice_batch_for_megatron_mimo(batch, dp_rank=1, dp_size=2)

        assert sliced["filenames"] == ["c.jpg", "d.jpg"]

    def test_raises_on_indivisible_batch(self):
        batch = {"tokens": torch.randn(5, 10)}  # 5 not divisible by 2
        with pytest.raises(ValueError, match="not divisible"):
            slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)

    def test_none_batch_passthrough(self):
        """None batch should not crash (forward_step passes None for non-data ranks)."""
        # slice_batch_for_megatron_mimo expects a dict; None is handled by caller.
        # This test documents that dp_size=1 early-return handles the common case.
        result = slice_batch_for_megatron_mimo({}, dp_rank=0, dp_size=1)
        assert result == {}
