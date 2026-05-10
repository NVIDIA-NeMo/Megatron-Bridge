# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for MegatronMIMO DP utilities."""

import pytest
import torch
import torch.distributed as dist

from megatron.bridge.data.megatron_mimo.dp_utils import (
    get_megatron_mimo_dp_info,
    get_megatron_mimo_sampling_info,
    slice_batch_for_megatron_mimo,
    slice_batch_for_megatron_mimo_modules,
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

    def __init__(
        self,
        rank_offset: int,
        size: int,
        dp_rank: int,
        dp_size: int,
        pp_rank: int,
        pp_size: int,
        cp_rank: int = 0,
        cp_size: int = 1,
    ):
        self.rank_offset = rank_offset
        self.size = size
        self._pgs = {
            ("dp",): FakePG(dp_rank, dp_size),
            ("pp",): FakePG(pp_rank, pp_size),
            ("cp",): FakePG(cp_rank, cp_size),
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


def test_get_megatron_mimo_sampling_info_colocated_needs_data_union(monkeypatch):
    """Colocated: needs_data is True if ANY served module needs data on this rank.

    A colocated rank may host multiple modules with different pipeline-stage
    roles. The rank needs data if any hosted module needs data.
    """
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    cfg = _make_colocated_cfg()
    # Encoder at PP=1 stage != 0 reports False; LLM at PP stage 0 reports True.
    # Iteration order puts vision first to verify the check considers all modules.
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


# ---------------------------------------------------------------------------
# Tests: slice_batch_for_megatron_mimo_modules
#
# Per-key per-module DP slicing. Colocated ranks serve multiple modules with
# possibly different DP sizes; language keys must be sliced by language DP and
# modality_inputs[<name>] by that encoder's DP. Non-colocated ranks reduce to
# the legacy uniform slicing.
# ---------------------------------------------------------------------------


def _make_batch(global_size: int = 4) -> dict:
    """Construct a representative batch with all the routed key shapes."""
    return {
        "input_ids": torch.arange(global_size * 8).reshape(global_size, 8),
        "labels": torch.arange(global_size * 8, global_size * 16).reshape(global_size, 8),
        "loss_mask": torch.ones(global_size, 8, dtype=torch.float),
        "position_ids": torch.arange(8).unsqueeze(0).repeat(global_size, 1),
        "attention_mask": torch.ones(global_size, 8, dtype=torch.bool),
        "modality_inputs": {
            "vision": {
                "pixel_values": torch.arange(global_size * 3).reshape(global_size, 3),
            },
        },
    }


class TestSliceBatchForMegatronMIMOModules:
    """Per-module per-key DP slicing for colocated heterogeneous TP/DP."""

    @pytest.fixture(autouse=True)
    def _stub_dist_initialized(self, monkeypatch):
        # The helper short-circuits when distributed isn't initialized
        # (defensive for non-distributed tests/legacy callers). These tests
        # exercise the rank-aware path, so stub is_initialized to True and
        # rely on the per-test ``dist.get_rank`` monkeypatch for membership.
        monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def test_routes_language_keys_by_language_dp_colocated(self, monkeypatch):
        """Colocated fan-in (vision DP=4, language DP=2): language keys take
        language DP slice, modality_inputs[vision] takes vision DP slice."""
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        # Rank 1 is in vision_dp_rank=1 (DP=4 → rank 1 sees sample 1) and
        # language_dp_rank=0 (DP=2 → rank 0 sees samples 0..1).
        grids = {
            "vision": FakeGrid(0, 4, dp_rank=1, dp_size=4, pp_rank=0, pp_size=1),
            "language": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        }
        batch = _make_batch(global_size=4)

        sliced = slice_batch_for_megatron_mimo_modules(batch, grids=grids)

        # Language keys: dp_rank=0, dp_size=2 → samples [0:2]
        assert sliced["input_ids"].shape == (2, 8)
        torch.testing.assert_close(sliced["input_ids"], batch["input_ids"][0:2])
        torch.testing.assert_close(sliced["labels"], batch["labels"][0:2])
        torch.testing.assert_close(sliced["loss_mask"], batch["loss_mask"][0:2])
        torch.testing.assert_close(sliced["position_ids"], batch["position_ids"][0:2])
        torch.testing.assert_close(sliced["attention_mask"], batch["attention_mask"][0:2])

        # Modality keys: vision dp_rank=1, dp_size=4 → sample [1:2]
        assert sliced["modality_inputs"]["vision"]["pixel_values"].shape == (1, 3)
        torch.testing.assert_close(
            sliced["modality_inputs"]["vision"]["pixel_values"],
            batch["modality_inputs"]["vision"]["pixel_values"][1:2],
        )

    def test_loss_mask_follows_language_not_encoder(self, monkeypatch):
        """Regression guard for the asymmetric-DP forward-step bug.

        A single uniform DP would collapse every key onto one module's slice,
        so loss_mask could match the wrong module's DP geometry. Lock the
        invariant explicitly: loss_mask routes through language DP.
        """
        monkeypatch.setattr(dist, "get_rank", lambda: 3)
        grids = {
            "vision": FakeGrid(0, 4, dp_rank=3, dp_size=4, pp_rank=0, pp_size=1),
            "language": FakeGrid(0, 4, dp_rank=1, dp_size=2, pp_rank=0, pp_size=1),
        }
        batch = _make_batch(global_size=4)

        sliced = slice_batch_for_megatron_mimo_modules(batch, grids=grids)

        # loss_mask sliced by language DP (rank=1, size=2) → samples [2:4],
        # NOT by vision DP (rank=3, size=4) → would be sample [3:4].
        assert sliced["loss_mask"].shape == (2, 8)
        torch.testing.assert_close(sliced["loss_mask"], batch["loss_mask"][2:4])

    def test_recurses_into_nested_modality_inputs(self, monkeypatch):
        """Encoder-internal nested dicts (e.g. {"clip": {"x": tensor}}) get
        sliced by the encoder's DP, recursively."""
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        grids = {
            "vision": FakeGrid(0, 4, dp_rank=0, dp_size=4, pp_rank=0, pp_size=1),
            "language": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        }
        batch = {
            "input_ids": torch.arange(4 * 8).reshape(4, 8),
            "loss_mask": torch.ones(4, 8),
            "modality_inputs": {
                "vision": {
                    "clip": {
                        "x": torch.arange(4 * 6).reshape(4, 6),
                    },
                },
            },
        }

        sliced = slice_batch_for_megatron_mimo_modules(batch, grids=grids)

        # vision dp_rank=0, dp_size=4 → sample [0:1]
        assert sliced["modality_inputs"]["vision"]["clip"]["x"].shape == (1, 6)
        torch.testing.assert_close(
            sliced["modality_inputs"]["vision"]["clip"]["x"],
            batch["modality_inputs"]["vision"]["clip"]["x"][0:1],
        )

    def test_non_colocated_falls_through_to_uniform_slice(self, monkeypatch):
        """Non-colocated rank (one module on this rank) reduces to legacy
        uniform slicing by that module's DP — preserves existing behavior."""
        monkeypatch.setattr(dist, "get_rank", lambda: 2)
        grids = {
            "vision": FakeGrid(0, 4, dp_rank=2, dp_size=4, pp_rank=0, pp_size=1),
            "language": FakeGrid(4, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        }
        batch = _make_batch(global_size=4)

        sliced = slice_batch_for_megatron_mimo_modules(batch, grids=grids)

        # Rank 2 is in vision only. Vision DP=4, rank=2 → sample [2:3] for everything.
        assert sliced["input_ids"].shape == (1, 8)
        torch.testing.assert_close(sliced["input_ids"], batch["input_ids"][2:3])
        assert sliced["modality_inputs"]["vision"]["pixel_values"].shape == (1, 3)
        torch.testing.assert_close(
            sliced["modality_inputs"]["vision"]["pixel_values"],
            batch["modality_inputs"]["vision"]["pixel_values"][2:3],
        )

    def test_language_keys_with_language_dp_1_passthrough(self, monkeypatch):
        """Fan-in 2-GPU shape: language DP=1 means language keys are not sliced
        (every TP rank sees the full global micro-batch), while modality_inputs
        is still sliced by encoder DP."""
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        # enc(TP=1,DP=2) × llm(TP=2,DP=1), total=2 each.
        grids = {
            "vision": FakeGrid(0, 2, dp_rank=1, dp_size=2, pp_rank=0, pp_size=1),
            "language": FakeGrid(0, 2, dp_rank=0, dp_size=1, pp_rank=0, pp_size=1),
        }
        batch = _make_batch(global_size=2)

        sliced = slice_batch_for_megatron_mimo_modules(batch, grids=grids)

        # language DP=1 → no slice on language keys
        assert sliced["input_ids"].shape == batch["input_ids"].shape
        torch.testing.assert_close(sliced["input_ids"], batch["input_ids"])

        # vision DP=2, rank=1 → sample [1:2]
        assert sliced["modality_inputs"]["vision"]["pixel_values"].shape == (1, 3)
        torch.testing.assert_close(
            sliced["modality_inputs"]["vision"]["pixel_values"],
            batch["modality_inputs"]["vision"]["pixel_values"][1:2],
        )

    def test_language_cp_siblings_keep_identical_language_batch_rows(self, monkeypatch):
        """Language CP does not affect batch slicing.

        enc(TP=1,CP=1,PP=1,DP=2) x lang(TP=1,CP=2,PP=1,DP=1) puts both
        ranks in the same language DP replica but different language CP ranks.
        Language keys stay full-batch on both CP siblings; only the encoder
        modality input follows encoder DP.
        """
        batch = _make_batch(global_size=2)

        def _grids_for_rank(rank: int) -> dict:
            return {
                "vision": FakeGrid(
                    0,
                    2,
                    dp_rank=rank,
                    dp_size=2,
                    pp_rank=0,
                    pp_size=1,
                    cp_rank=0,
                    cp_size=1,
                ),
                "language": FakeGrid(
                    0,
                    2,
                    dp_rank=0,
                    dp_size=1,
                    pp_rank=0,
                    pp_size=1,
                    cp_rank=rank,
                    cp_size=2,
                ),
            }

        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        rank0 = slice_batch_for_megatron_mimo_modules(batch, grids=_grids_for_rank(0))

        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        rank1 = slice_batch_for_megatron_mimo_modules(batch, grids=_grids_for_rank(1))

        for key in ("input_ids", "labels", "loss_mask", "position_ids", "attention_mask"):
            torch.testing.assert_close(rank0[key], batch[key])
            torch.testing.assert_close(rank1[key], batch[key])
            torch.testing.assert_close(rank0[key], rank1[key])

        torch.testing.assert_close(
            rank0["modality_inputs"]["vision"]["pixel_values"],
            batch["modality_inputs"]["vision"]["pixel_values"][0:1],
        )
        torch.testing.assert_close(
            rank1["modality_inputs"]["vision"]["pixel_values"],
            batch["modality_inputs"]["vision"]["pixel_values"][1:2],
        )
