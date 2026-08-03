# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for MegatronMIMO DP utilities."""

import pytest
import torch
import torch.distributed as dist

from megatron.bridge.data.megatron_mimo.dp_utils import (
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


def test_get_megatron_mimo_dp_info_llm_intermediate_pp(monkeypatch):
    """Test heterogeneous mode, rank in LLM module, intermediate PP stage."""
    megatron_mimo_cfg = _make_megatron_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 5)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=1, dp_size=4, pp_rank=1, pp_size=3),
    }

    dp_rank, dp_size, needs_data, loader_module = get_megatron_mimo_dp_info(megatron_mimo_cfg, grids)

    assert loader_module == "language"
    assert dp_rank == 1
    assert dp_size == 4
    assert needs_data is True  # Intermediate LLM PP stages need position_ids.


def test_get_megatron_mimo_sampling_info_llm_intermediate_pp(monkeypatch):
    """Test loader construction remains enabled for intermediate LLM PP stages."""
    megatron_mimo_cfg = _make_megatron_mimo_cfg()
    monkeypatch.setattr(dist, "get_rank", lambda: 5)

    grids = {
        "vision": FakeGrid(0, 4, dp_rank=0, dp_size=2, pp_rank=0, pp_size=1),
        "language": FakeGrid(4, 4, dp_rank=1, dp_size=4, pp_rank=1, pp_size=3),
    }

    sampler_dp_rank, sampler_dp_size, needs_data = get_megatron_mimo_sampling_info(megatron_mimo_cfg, grids)

    assert sampler_dp_rank == 0
    assert sampler_dp_size == 1
    assert needs_data is True


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

    def test_slices_qwen_mrope_position_ids_along_batch_dim(self):
        position_ids = torch.arange(3 * 4 * 5).reshape(3, 4, 5)
        batch = {"position_ids": position_ids}

        sliced = slice_batch_for_megatron_mimo(batch, dp_rank=2, dp_size=4)

        assert sliced["position_ids"].shape == (3, 1, 5)
        torch.testing.assert_close(sliced["position_ids"], position_ids[:, 2:3, :])

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


class TestPatchPackedVisualSlice:
    """Joint-slice for a patch-packed visual encoder dict.

    Some visual adapters produce encoder dicts like:
      - hidden_states: [sum(patches_across_images), feat]
      - grid_thw:      [num_images, 3]

    Naive dim-0 slicing corrupts the pair when per-image patch counts differ.
    These tests verify the joint slicer keeps the pair consistent.
    """

    @staticmethod
    def _make_batch(grids):
        """Build a {modality_inputs: {images: {vision_encoder: {hidden_states, grid_thw}}}} batch.

        ``grids`` is a list of (t, h, w) triples. Each image's hidden_states
        rows are filled with its image index, so we can verify which image each
        row came from after slicing.
        """
        grid_thw = torch.tensor(grids, dtype=torch.long)
        patches_per_image = grid_thw.prod(dim=-1)
        total_patches = int(patches_per_image.sum().item())
        feat_dim = 4
        hidden_states = torch.zeros(total_patches, feat_dim, dtype=torch.float32)
        offset = 0
        for img_idx, ppi in enumerate(patches_per_image.tolist()):
            hidden_states[offset : offset + ppi].fill_(float(img_idx))
            offset += ppi
        return {
            "modality_inputs": {
                "images": {
                    "vision_encoder": {
                        "hidden_states": hidden_states,
                        "grid_thw": grid_thw,
                    }
                }
            }
        }

    def test_uniform_grid_dp2(self):
        """Uniform grid: joint slice matches naive even split (regression-safe)."""
        # 4 images, all (1, 4, 4): 16 patches each, 64 total.
        batch = self._make_batch([(1, 4, 4)] * 4)
        s0 = slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)
        s1 = slice_batch_for_megatron_mimo(batch, dp_rank=1, dp_size=2)

        e0 = s0["modality_inputs"]["images"]["vision_encoder"]
        e1 = s1["modality_inputs"]["images"]["vision_encoder"]

        # grid_thw splits by image-count
        assert e0["grid_thw"].shape == (2, 3)
        assert e1["grid_thw"].shape == (2, 3)
        # hidden_states splits by patch-count (= image-count * patches/image)
        assert e0["hidden_states"].shape == (32, 4)
        assert e1["hidden_states"].shape == (32, 4)
        # Each rank's rows carry its image indices: rank 0 gets {0, 1}; rank 1 gets {2, 3}
        assert set(e0["hidden_states"].unique().tolist()) == {0.0, 1.0}
        assert set(e1["hidden_states"].unique().tolist()) == {2.0, 3.0}

    def test_variable_grid_dp2(self):
        """Variable grid: joint slice keeps hidden_states aligned with grid_thw."""
        # Per-image patch counts: 4, 16, 9, 25; total 54 patches.
        # Naive dim-0 split (54/2 = 27) would bisect mid-image. Joint slicer
        # must split at image boundary: rank 0 gets images [0, 1], rank 1 gets [2, 3].
        batch = self._make_batch([(1, 2, 2), (1, 4, 4), (1, 3, 3), (1, 5, 5)])
        s0 = slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)
        s1 = slice_batch_for_megatron_mimo(batch, dp_rank=1, dp_size=2)

        e0 = s0["modality_inputs"]["images"]["vision_encoder"]
        e1 = s1["modality_inputs"]["images"]["vision_encoder"]

        # 2 images per shard
        assert e0["grid_thw"].shape == (2, 3)
        assert e1["grid_thw"].shape == (2, 3)
        # Rank 0: images 0+1 -> 4 + 16 = 20 patches
        assert e0["hidden_states"].shape == (20, 4)
        # Rank 1: images 2+3 -> 9 + 25 = 34 patches
        assert e1["hidden_states"].shape == (34, 4)
        # And each rank's patch rows correspond to its assigned images.
        assert set(e0["hidden_states"].unique().tolist()) == {0.0, 1.0}
        assert set(e1["hidden_states"].unique().tolist()) == {2.0, 3.0}

    def test_dp4_with_variable_grids(self):
        """4-way joint slice across 4 variable-size images."""
        # 4 images, 4 different sizes: 1 image per shard at DP=4.
        batch = self._make_batch([(1, 2, 2), (1, 4, 4), (1, 3, 3), (1, 5, 5)])
        for rank in range(4):
            sliced = slice_batch_for_megatron_mimo(batch, dp_rank=rank, dp_size=4)
            enc = sliced["modality_inputs"]["images"]["vision_encoder"]
            assert enc["grid_thw"].shape == (1, 3)
            # Each rank's hidden_states is exactly that image's patches,
            # all tagged with its image index.
            expected_patches = int(enc["grid_thw"].prod().item())
            assert enc["hidden_states"].shape == (expected_patches, 4)
            assert enc["hidden_states"].unique().tolist() == [float(rank)]

    def test_raises_when_num_images_not_divisible(self):
        """3 images / DP=2 should raise with a clear error."""
        batch = self._make_batch([(1, 2, 2), (1, 3, 3), (1, 4, 4)])
        with pytest.raises(ValueError, match="not divisible by encoder DP"):
            slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)

    def test_raises_when_patch_count_mismatch(self):
        """hidden_states dim 0 must match the patch count implied by grid_thw."""
        batch = self._make_batch([(1, 2, 2), (1, 3, 3)])
        batch["modality_inputs"]["images"]["vision_encoder"]["hidden_states"] = torch.zeros(12, 4)
        with pytest.raises(ValueError, match="sum\\(grid_thw products\\)"):
            slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)

    def test_extra_keys_passed_through(self):
        """Encoder dicts may carry other entries (kwargs, masks), and they pass through unchanged."""
        batch = self._make_batch([(1, 2, 2), (1, 3, 3)])
        # Inject a non-(hidden_states/grid_thw) entry. It should not be sliced.
        batch["modality_inputs"]["images"]["vision_encoder"]["encoder_meta"] = "fixed-string"
        sliced = slice_batch_for_megatron_mimo(batch, dp_rank=0, dp_size=2)
        enc = sliced["modality_inputs"]["images"]["vision_encoder"]
        assert enc["encoder_meta"] == "fixed-string"
