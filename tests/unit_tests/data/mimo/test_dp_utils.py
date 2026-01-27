# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
    
    def __init__(self, rank_offset: int, size: int, dp_rank: int, dp_size: int, 
                 pp_rank: int, pp_size: int):
        self.rank_offset = rank_offset
        self.size = size
        self._pgs = {
            ("dp",): FakePG(dp_rank, dp_size),
            ("pp",): FakePG(pp_rank, pp_size),
        }

    def get_pg(self, dims):
        return self._pgs[tuple(dims)]


def _make_mimo_cfg(deployment_mode: str) -> MimoParallelismConfig:
    """Create test MIMO config."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(tensor_parallel=1, data_parallel=2, rank_offset=0),
        "llm": ModuleParallelismConfig(tensor_parallel=1, data_parallel=4, rank_offset=4),
    }
    return MimoParallelismConfig(
        llm_module_name="llm",
        module_parallelisms=module_parallelisms,
        deployment_mode=deployment_mode,
    )


def test_get_mimo_dp_info_colocated(monkeypatch):
    """Test colocated mode selects module with smallest DP size."""
    mimo_cfg = _make_mimo_cfg("colocated")
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    grids = {
        "vision": FakeGrid(0, 8, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "llm": FakeGrid(0, 8, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "vision"  # Smallest DP size
    assert dp_rank == 0
    assert dp_size == 2
    assert needs_data is True  # PP rank 0 needs data


def test_get_mimo_dp_info_colocated_llm_first_pp(monkeypatch):
    """Test colocated mode with LLM as loader, first PP stage."""
    mimo_cfg = _make_mimo_cfg("colocated")
    mimo_cfg.module_parallelisms["vision"].data_parallel = 8  # Make vision larger
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    grids = {
        "vision": FakeGrid(0, 8, dp_rank=0, dp_size=8, pp_rank=0, pp_size=2),
        "llm": FakeGrid(0, 8, dp_rank=0, dp_size=4, pp_rank=0, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "llm"  # Now LLM is smaller
    assert needs_data is True  # First PP stage


def test_get_mimo_dp_info_colocated_llm_last_pp(monkeypatch):
    """Test colocated mode with LLM as loader, last PP stage."""
    mimo_cfg = _make_mimo_cfg("colocated")
    mimo_cfg.module_parallelisms["vision"].data_parallel = 8
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    grids = {
        "vision": FakeGrid(0, 8, dp_rank=0, dp_size=8, pp_rank=1, pp_size=2),
        "llm": FakeGrid(0, 8, dp_rank=0, dp_size=4, pp_rank=1, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "llm"
    assert needs_data is True  # Last PP stage


def test_get_mimo_dp_info_colocated_encoder_middle_pp(monkeypatch):
    """Test colocated mode with encoder as loader, middle PP stage."""
    mimo_cfg = _make_mimo_cfg("colocated")
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    grids = {
        "vision": FakeGrid(0, 8, dp_rank=0, dp_size=2, pp_rank=1, pp_size=3),  # Middle
        "llm": FakeGrid(0, 8, dp_rank=0, dp_size=4, pp_rank=1, pp_size=3),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "vision"
    assert needs_data is False  # Encoder only needs data on first PP


def test_get_mimo_dp_info_homogeneous(monkeypatch):
    """Test homogeneous mode uses LLM settings."""
    mimo_cfg = _make_mimo_cfg("homogeneous")
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    grids = {
        "vision": FakeGrid(0, 8, dp_rank=1, dp_size=4, pp_rank=0, pp_size=1),
        "llm": FakeGrid(0, 8, dp_rank=1, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "llm"
    assert dp_rank == 1
    assert dp_size == 4
    assert needs_data is True


def test_get_mimo_dp_info_heterogeneous_encoder_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, first PP stage."""
    mimo_cfg = _make_mimo_cfg("heterogeneous")
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


def test_get_mimo_dp_info_heterogeneous_encoder_non_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in encoder module, not first PP stage."""
    mimo_cfg = _make_mimo_cfg("heterogeneous")
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=1, pp_size=2),
        "llm": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "vision"
    assert needs_data is False  # Not first PP stage


def test_get_mimo_dp_info_heterogeneous_llm_first_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, first PP stage."""
    mimo_cfg = _make_mimo_cfg("heterogeneous")
    monkeypatch.setattr(dist, "get_rank", lambda: 4)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "llm": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "llm"
    assert needs_data is True  # First PP stage


def test_get_mimo_dp_info_heterogeneous_llm_last_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, last PP stage."""
    mimo_cfg = _make_mimo_cfg("heterogeneous")
    monkeypatch.setattr(dist, "get_rank", lambda: 5)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "llm": FakeGrid(4, 4, dp_rank=1, dp_size=4, pp_rank=1, pp_size=2),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert loader_module == "llm"
    assert needs_data is True  # Last PP stage


def test_get_mimo_dp_info_heterogeneous_non_participating_rank(monkeypatch):
    """Test heterogeneous mode, rank not in any module."""
    mimo_cfg = _make_mimo_cfg("heterogeneous")
    monkeypatch.setattr(dist, "get_rank", lambda: 10)  # Outside all grids

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "llm": FakeGrid(4, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
    }

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(mimo_cfg, grids)
    
    assert needs_data is False
    assert loader_module == "llm"  # Default to LLM
