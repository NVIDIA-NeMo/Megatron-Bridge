# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO DDP wrapping utilities."""

from unittest.mock import MagicMock, patch

from megatron.bridge.models.mimo.mimo_builder import is_current_rank_in_grid
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig
from megatron.bridge.models.mimo.mimo_ddp import wrap_mimo_model_distributed


class TestIsCurrentRankInGrid:
    """Test cases for is_current_rank_in_grid helper."""

    @patch("torch.distributed.get_rank")
    def test_rank_in_grid(self, mock_get_rank):
        """Rank within grid range should return True."""
        mock_get_rank.return_value = 2

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is True

    @patch("torch.distributed.get_rank")
    def test_rank_at_grid_start(self, mock_get_rank):
        """Rank at grid start should return True."""
        mock_get_rank.return_value = 4

        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is True

    @patch("torch.distributed.get_rank")
    def test_rank_at_grid_end_exclusive(self, mock_get_rank):
        """Rank at grid end (exclusive) should return False."""
        mock_get_rank.return_value = 8

        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is False

    @patch("torch.distributed.get_rank")
    def test_rank_before_grid(self, mock_get_rank):
        """Rank before grid range should return False."""
        mock_get_rank.return_value = 2

        mock_grid = MagicMock()
        mock_grid.rank_offset = 4
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is False

    @patch("torch.distributed.get_rank")
    def test_rank_after_grid(self, mock_get_rank):
        """Rank after grid range should return False."""
        mock_get_rank.return_value = 10

        mock_grid = MagicMock()
        mock_grid.rank_offset = 0
        mock_grid.size = 4

        assert is_current_rank_in_grid(mock_grid) is False


class TestWrapMimoModelDistributed:
    """Test cases for wrap_mimo_model_distributed."""

    def _create_mock_mimo_model(self, has_language_model=True, modality_names=None):
        """Create a mock MimoModel for testing."""
        mock_model = MagicMock()

        if has_language_model:
            mock_model.language_model = MagicMock()
            mock_model.language_model.config = MagicMock()
        else:
            mock_model.language_model = None

        if modality_names:
            mock_model.modality_submodules = {}
            for name in modality_names:
                submodule = MagicMock()
                submodule.encoders = {"encoder": MagicMock()}
                submodule.encoders["encoder"].config = MagicMock()
                mock_model.modality_submodules[name] = submodule
        else:
            mock_model.modality_submodules = {}

        return mock_model

    def _create_mock_grid(self, rank_offset=0, size=4):
        """Create a mock HyperCommGrid."""
        mock_grid = MagicMock()
        mock_grid.rank_offset = rank_offset
        mock_grid.size = size
        return mock_grid

    def _create_mimo_parallelism_config(self, modules):
        """Create a MimoParallelismConfig."""
        module_parallelisms = {
            name: ModuleParallelismConfig(
                tensor_model_parallel_size=config.get("tp", 1),
                data_parallel_size=config.get("dp", 1),
                rank_offset=config.get("rank_offset", 0),
            )
            for name, config in modules.items()
        }
        return MimoParallelismConfig(
            module_parallelisms=module_parallelisms,
        )

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_wrap_language_model(self, mock_get_rank, mock_ddp):
        """Test that language model is wrapped with DDP when rank participates."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        mimo_model = self._create_mock_mimo_model(has_language_model=True)
        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2},
            }
        )

        grids = {"llm": self._create_mock_grid(rank_offset=0, size=4)}
        pg_collections = {"llm": MagicMock()}

        result = wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Should wrap language model
        mock_ddp.assert_called_once()
        assert result.language_model == mock_ddp.return_value

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_skip_language_model_non_participating_rank(self, mock_get_rank, mock_ddp):
        """Test that language model is NOT wrapped when rank doesn't participate."""
        mock_get_rank.return_value = 10  # Outside grid range

        mimo_model = self._create_mock_mimo_model(has_language_model=True)
        original_lm = mimo_model.language_model

        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2},
            }
        )

        grids = {"llm": self._create_mock_grid(rank_offset=0, size=4)}
        pg_collections = {"llm": MagicMock()}

        result = wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Should NOT wrap language model
        mock_ddp.assert_not_called()
        assert result.language_model == original_lm

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_wrap_modality_submodules(self, mock_get_rank, mock_ddp):
        """Test that modality submodules are wrapped with DDP."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        mimo_model = self._create_mock_mimo_model(has_language_model=True, modality_names=["images"])
        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2},
                "images": {"tp": 1, "dp": 4},
            }
        )

        grids = {
            "llm": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=0, size=4),
        }
        pg_collections = {
            "llm": MagicMock(),
            "images": MagicMock(),
        }

        wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Should wrap both language model and images submodule
        assert mock_ddp.call_count == 2

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_skip_modality_submodule_no_grid(self, mock_get_rank, mock_ddp):
        """Test that modality submodules without grids are skipped."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        mimo_model = self._create_mock_mimo_model(has_language_model=True, modality_names=["images", "audio"])
        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2},
                "images": {"tp": 1, "dp": 4},
                # Note: no "audio" in parallelism config
            }
        )

        # Only llm and images have grids
        grids = {
            "llm": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=0, size=4),
        }
        pg_collections = {
            "llm": MagicMock(),
            "images": MagicMock(),
        }

        wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Should wrap llm and images, but not audio (no grid)
        assert mock_ddp.call_count == 2

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_heterogeneous_different_rank_ranges(self, mock_get_rank, mock_ddp):
        """Test heterogeneous deployment with different rank ranges per module."""
        mock_get_rank.return_value = 4  # In images grid but not llm grid
        mock_ddp.return_value = MagicMock()

        mimo_model = self._create_mock_mimo_model(has_language_model=True, modality_names=["images"])
        original_lm = mimo_model.language_model

        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2, "rank_offset": 0},
                "images": {"tp": 2, "dp": 2, "rank_offset": 4},
            }
        )

        grids = {
            "llm": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=4, size=4),
        }
        pg_collections = {
            "llm": None,  # Rank 4 doesn't participate in LLM
            "images": MagicMock(),
        }

        result = wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Should wrap only images (rank 4 is in images grid, not llm grid)
        assert mock_ddp.call_count == 1
        # Language model should be unchanged
        assert result.language_model == original_lm

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_no_language_model(self, mock_get_rank, mock_ddp):
        """Test model without language model."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        mimo_model = self._create_mock_mimo_model(has_language_model=False, modality_names=["images"])
        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2},
                "images": {"tp": 1, "dp": 4},
            }
        )

        grids = {
            "llm": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=0, size=4),
        }
        pg_collections = {
            "llm": MagicMock(),
            "images": MagicMock(),
        }

        result = wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Should wrap only images (no language model)
        assert mock_ddp.call_count == 1
        assert result.language_model is None

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_returns_same_model_instance(self, mock_get_rank, mock_ddp):
        """Test that wrap_mimo_model_distributed returns the same model instance."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        mimo_model = self._create_mock_mimo_model(has_language_model=True)
        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2},
            }
        )

        grids = {"llm": self._create_mock_grid(rank_offset=0, size=4)}
        pg_collections = {"llm": MagicMock()}

        result = wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Should return the same model instance (modified in-place)
        assert result is mimo_model

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_ddp_called_with_correct_args(self, mock_get_rank, mock_ddp):
        """Test that DDP is called with correct arguments."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        mimo_model = self._create_mock_mimo_model(has_language_model=True)
        # Capture original config before wrapping (wrapping replaces language_model)
        original_lm_config = mimo_model.language_model.config
        original_lm = mimo_model.language_model

        ddp_config = MagicMock()
        mimo_parallelism_config = self._create_mimo_parallelism_config(
            {
                "llm": {"tp": 2, "dp": 2},
            }
        )

        grids = {"llm": self._create_mock_grid(rank_offset=0, size=4)}
        llm_pg_collection = MagicMock()
        pg_collections = {"llm": llm_pg_collection}

        wrap_mimo_model_distributed(mimo_model, ddp_config, mimo_parallelism_config, grids, pg_collections)

        # Verify DDP call arguments
        mock_ddp.assert_called_once()
        call_kwargs = mock_ddp.call_args.kwargs
        assert call_kwargs["ddp_config"] == ddp_config
        assert call_kwargs["pg_collection"] == llm_pg_collection
        assert call_kwargs["config"] == original_lm_config
        assert call_kwargs["module"] == original_lm
