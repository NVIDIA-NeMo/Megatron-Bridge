# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for MIMO DP utilities."""

import torch.distributed as dist

from megatron.bridge.data.mimo.dp_utils import get_mimo_dp_info
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig


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


def _make_mimo_cfg() -> MimoParallelismConfig:
    """Create test MIMO config for heterogeneous deployment."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=0),
        "llm": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=4),
    }
    return MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )


def test_get_mimo_dp_info_encoder_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, first PP stage."""
    mimo_cfg = _make_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2),
        "llm": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)

    assert loader_module == "vision"
    assert dp_rank == 0
    assert dp_size == 2
    assert needs_data is True  # First PP stage


def test_get_mimo_dp_info_encoder_non_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, not first PP stage."""
    mimo_cfg = _make_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=1, pp_size=2),
        "llm": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)

    assert loader_module == "vision"
    assert needs_data is False  # Not first PP stage


def test_get_mimo_dp_info_llm_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, first PP stage."""
    mimo_cfg = _make_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 4)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "llm": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)

    assert loader_module == "llm"
    assert needs_data is True  # First PP stage


def test_get_mimo_dp_info_llm_last_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, last PP stage."""
    mimo_cfg = _make_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 5)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "llm": FakeGrid(4, 4, dp_rank=1, dp_size=4, pp_rank=1, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)

    assert loader_module == "llm"
    assert needs_data is True  # Last PP stage


def test_get_mimo_dp_info_non_participating_rank(monkeypatch):
    """Test heterogeneous mode, rank not in any module."""
    mimo_cfg = _make_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 10)  # Outside all grids

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "llm": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)

    assert needs_data is False
    assert loader_module == "llm"  # Default to LLM
