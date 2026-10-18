# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO DDP wrapping utilities."""

from unittest.mock import MagicMock, patch

import pytest
from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_ddp import wrap_megatron_mimo_model_distributed


class TestWrapMegatronMIMOModelDistributed:
    """Test cases for wrap_megatron_mimo_model_distributed."""

    def _create_mock_megatron_mimo_model(self, has_language_model=True, modality_names=None):
        """Create a mock MimoModel (MCore) for testing."""
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

    def _create_megatron_mimo_parallelism_config(self, modules):
        """Create an MegatronMIMOParallelismConfig."""
        module_parallelisms = {
            name: ModuleParallelismConfig(
                tensor_model_parallel_size=config.get("tp", 1),
                data_parallel_size=config.get("dp", 1),
                rank_offset=config.get("rank_offset", 0),
            )
            for name, config in modules.items()
        }
        return MegatronMIMOParallelismConfig(
            module_parallelisms=module_parallelisms,
        )

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_wrap_language_model(self, mock_get_rank, mock_ddp):
        """Test that language model is wrapped with DDP when rank participates."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        megatron_mimo_model = self._create_mock_megatron_mimo_model(has_language_model=True)
        ddp_config = MagicMock()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
            }
        )

        grids = {"language": self._create_mock_grid(rank_offset=0, size=4)}
        pg_collections = {"language": MagicMock()}

        result = wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

        # Should wrap language model
        mock_ddp.assert_called_once()
        assert result.language_model == mock_ddp.return_value

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_skip_language_model_non_participating_rank(self, mock_get_rank, mock_ddp):
        """Test that language model is NOT wrapped when rank doesn't participate."""
        mock_get_rank.return_value = 10  # Outside grid range

        megatron_mimo_model = self._create_mock_megatron_mimo_model(has_language_model=True)
        original_lm = megatron_mimo_model.language_model

        ddp_config = MagicMock()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
            }
        )

        grids = {"language": self._create_mock_grid(rank_offset=0, size=4)}
        pg_collections = {"language": MagicMock()}

        result = wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

        # Should NOT wrap language model
        mock_ddp.assert_not_called()
        assert result.language_model == original_lm

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_wrap_modality_submodules(self, mock_get_rank, mock_ddp):
        """Test that modality submodules are wrapped with DDP."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        megatron_mimo_model = self._create_mock_megatron_mimo_model(has_language_model=True, modality_names=["images"])
        ddp_config = DistributedDataParallelConfig()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
                "images": {"tp": 1, "dp": 4},
            }
        )

        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=0, size=4),
        }
        pg_collections = {
            "language": MagicMock(),
            "images": MagicMock(),
        }

        wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

        wrapped_modality = megatron_mimo_model.modality_submodules["images"]
        assert wrapped_modality.module.pg_collection is pg_collections["images"]

        # Should wrap both language model and images submodule
        assert mock_ddp.call_count == 2

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_encoder_ddp_disables_overlap(self, mock_get_rank, mock_ddp):
        """Encoder DDP never overlaps communication; the language DDP keeps the configured overlap."""
        mock_get_rank.return_value = 0

        megatron_mimo_model = self._create_mock_megatron_mimo_model(has_language_model=True, modality_names=["images"])
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=True, overlap_param_gather=True
        )
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
                "images": {"tp": 1, "dp": 4},
            }
        )
        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=0, size=4),
        }
        pg_collections = {"language": MagicMock(), "images": MagicMock()}

        with pytest.warns(UserWarning, match="Disabling gradient-reduce and parameter-gather overlap"):
            wrap_megatron_mimo_model_distributed(
                megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
            )

        language_ddp_config, encoder_ddp_config = (call.kwargs["ddp_config"] for call in mock_ddp.call_args_list)
        assert language_ddp_config is ddp_config
        assert not encoder_ddp_config.overlap_grad_reduce
        assert not encoder_ddp_config.overlap_param_gather
        assert encoder_ddp_config.use_distributed_optimizer
        assert encoder_ddp_config is not ddp_config
        assert ddp_config.overlap_grad_reduce and ddp_config.overlap_param_gather

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_encoder_ddp_reuses_config_when_overlap_is_disabled(self, mock_get_rank, mock_ddp):
        """Already-synchronous encoder DDP does not clone or warn."""
        mock_get_rank.return_value = 0
        model = self._create_mock_megatron_mimo_model(has_language_model=True, modality_names=["images"])
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=False, overlap_param_gather=False)
        parallelism = self._create_megatron_mimo_parallelism_config({"language": {"dp": 2}, "images": {"dp": 2}})
        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=2),
            "images": self._create_mock_grid(rank_offset=0, size=2),
        }

        with patch("megatron.bridge.models.megatron_mimo.megatron_mimo_ddp.warn_rank_0") as warn:
            wrap_megatron_mimo_model_distributed(
                model,
                ddp_config,
                parallelism,
                grids,
                {"language": MagicMock(), "images": MagicMock()},
            )

        warn.assert_not_called()
        assert [call.kwargs["ddp_config"] for call in mock_ddp.call_args_list] == [ddp_config, ddp_config]

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_encoder_overlap_warning_emitted_when_rank_zero_is_outside_encoder_grid(self, mock_get_rank, mock_ddp):
        """Global rank zero warns even when the encoder starts at a nonzero rank."""
        mock_get_rank.return_value = 0
        model = self._create_mock_megatron_mimo_model(has_language_model=True, modality_names=["images"])
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, overlap_param_gather=True)
        parallelism = self._create_megatron_mimo_parallelism_config(
            {"language": {"dp": 2}, "images": {"dp": 2, "rank_offset": 2}}
        )
        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=2),
            "images": self._create_mock_grid(rank_offset=2, size=2),
        }

        with pytest.warns(UserWarning, match="Disabling gradient-reduce and parameter-gather overlap"):
            wrap_megatron_mimo_model_distributed(
                model,
                ddp_config,
                parallelism,
                grids,
                {"language": MagicMock(), "images": None},
            )

        mock_ddp.assert_called_once()

    @pytest.mark.parametrize("module_name,rank", [("language", 0), ("images", 2)])
    @patch("megatron.core.optimizer.layer_wise_optimizer.tag_params_for_buffer_routing")
    @patch("megatron.core.optimizer.layer_wise_optimizer.LayerWiseDistributedOptimizer.compute_full_param_layout")
    @patch("megatron.core.distributed.DistributedDataParallel")
    def test_layer_wise_layout_uses_component_overlap_config(self, mock_ddp, mock_layout, mock_tag, module_name, rank):
        """Layout and DDP must agree on encoder overlap before constructing buckets."""
        model = self._create_mock_megatron_mimo_model(modality_names=["images"])
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=128,
            grad_reduce_in_fp32=True,
        )
        parallelism = self._create_megatron_mimo_parallelism_config(
            {"language": {"dp": 2}, "images": {"dp": 2, "rank_offset": 2}}
        )
        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=2),
            "images": self._create_mock_grid(rank_offset=2, size=2),
        }
        pg = MagicMock()
        groups = {name: MagicMock() for name in ("dp_group", "pp_group", "intra_dp_cp_group", "intra_expt_dp_group")}
        groups["pp_group"].rank.return_value = 0
        groups["intra_dp_cp_group"].size.return_value = 2
        groups["intra_expt_dp_group"].size.return_value = 2
        pg.setup_process_groups_for_ddp.return_value = groups

        with patch("torch.distributed.get_rank", return_value=rank):
            wrap_megatron_mimo_model_distributed(
                model,
                ddp_config,
                parallelism,
                grids,
                {
                    "language": pg if module_name == "language" else None,
                    "images": pg if module_name == "images" else None,
                },
                use_layer_wise_distributed_optimizer=True,
                use_layer_wise_param_layout=True,
            )

        mock_ddp.assert_called_once()
        mock_layout.assert_called_once()
        component_config = mock_ddp.call_args.kwargs["ddp_config"]
        assert mock_layout.call_args.args[3] is component_config
        assert pg.setup_process_groups_for_ddp.call_args.args[2] is component_config
        assert mock_ddp.call_args.kwargs["full_param_layout"] is mock_layout.return_value
        assert mock_layout.call_args.args[1] == (128 if module_name == "language" else None)
        assert component_config.overlap_grad_reduce == (module_name == "language")
        assert component_config.overlap_param_gather == (module_name == "language")
        assert component_config.grad_reduce_in_fp32
        assert ddp_config.overlap_grad_reduce and ddp_config.overlap_param_gather

    @patch("megatron.core.optimizer.layer_wise_optimizer.tag_params_for_buffer_routing")
    @patch("megatron.core.optimizer.layer_wise_optimizer.LayerWiseDistributedOptimizer.compute_full_param_layout")
    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_compact_layer_wise_layout_keeps_default_ddp_buffers(self, mock_get_rank, mock_ddp, mock_layout, mock_tag):
        """Compact LayerWise mode supplies no layout and leaves the DDP mode disabled."""
        mock_get_rank.return_value = 0
        model = self._create_mock_megatron_mimo_model(has_language_model=True, modality_names=["images"])
        language_model = model.language_model
        encoder = model.modality_submodules["images"]
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=False, overlap_grad_reduce=False, overlap_param_gather=False
        )
        parallelism = self._create_megatron_mimo_parallelism_config({"language": {"dp": 2}, "images": {"dp": 2}})
        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=2),
            "images": self._create_mock_grid(rank_offset=0, size=2),
        }
        pg_collections = {"language": MagicMock(), "images": MagicMock()}

        wrap_megatron_mimo_model_distributed(
            model,
            ddp_config,
            parallelism,
            grids,
            pg_collections,
            use_layer_wise_distributed_optimizer=True,
        )

        mock_tag.assert_not_called()
        mock_layout.assert_not_called()
        for pg in pg_collections.values():
            pg.setup_process_groups_for_ddp.assert_not_called()
        assert [call.kwargs["module"] for call in mock_ddp.call_args_list] == [language_model, encoder]
        assert [call.kwargs["full_param_layout"] for call in mock_ddp.call_args_list] == [None, None]
        assert [call.kwargs["ddp_config"] for call in mock_ddp.call_args_list] == [ddp_config, ddp_config]
        assert not ddp_config.use_distributed_optimizer

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_skip_modality_submodule_no_grid(self, mock_get_rank, mock_ddp):
        """Test that modality submodules without grids are skipped."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        megatron_mimo_model = self._create_mock_megatron_mimo_model(
            has_language_model=True, modality_names=["images", "audio"]
        )
        ddp_config = DistributedDataParallelConfig()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
                "images": {"tp": 1, "dp": 4},
                # Note: no "audio" in parallelism config
            }
        )

        # Only llm and images have grids
        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=0, size=4),
        }
        pg_collections = {
            "language": MagicMock(),
            "images": MagicMock(),
        }

        wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

        # Should wrap llm and images, but not audio (no grid)
        assert mock_ddp.call_count == 2

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_heterogeneous_different_rank_ranges(self, mock_get_rank, mock_ddp):
        """Test heterogeneous deployment with different rank ranges per module."""
        mock_get_rank.return_value = 4  # In images grid but not llm grid
        mock_ddp.return_value = MagicMock()

        megatron_mimo_model = self._create_mock_megatron_mimo_model(has_language_model=True, modality_names=["images"])
        original_lm = megatron_mimo_model.language_model

        ddp_config = DistributedDataParallelConfig()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2, "rank_offset": 0},
                "images": {"tp": 2, "dp": 2, "rank_offset": 4},
            }
        )

        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=4, size=4),
        }
        pg_collections = {
            "language": None,  # Rank 4 doesn't participate in LLM
            "images": MagicMock(),
        }

        result = wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

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

        megatron_mimo_model = self._create_mock_megatron_mimo_model(
            has_language_model=False, modality_names=["images"]
        )
        ddp_config = DistributedDataParallelConfig()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
                "images": {"tp": 1, "dp": 4},
            }
        )

        grids = {
            "language": self._create_mock_grid(rank_offset=0, size=4),
            "images": self._create_mock_grid(rank_offset=0, size=4),
        }
        pg_collections = {
            "language": MagicMock(),
            "images": MagicMock(),
        }

        result = wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

        # Should wrap only images (no language model)
        assert mock_ddp.call_count == 1
        assert result.language_model is None

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_returns_same_model_instance(self, mock_get_rank, mock_ddp):
        """Test that wrap_megatron_mimo_model_distributed returns the same model instance."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        megatron_mimo_model = self._create_mock_megatron_mimo_model(has_language_model=True)
        ddp_config = MagicMock()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
            }
        )

        grids = {"language": self._create_mock_grid(rank_offset=0, size=4)}
        pg_collections = {"language": MagicMock()}

        result = wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

        # Should return the same model instance (modified in-place)
        assert result is megatron_mimo_model

    @patch("megatron.core.distributed.DistributedDataParallel")
    @patch("torch.distributed.get_rank")
    def test_ddp_called_with_correct_args(self, mock_get_rank, mock_ddp):
        """Test that DDP is called with correct arguments."""
        mock_get_rank.return_value = 0
        mock_ddp.return_value = MagicMock()

        megatron_mimo_model = self._create_mock_megatron_mimo_model(has_language_model=True)
        # Capture original config before wrapping (wrapping replaces language_model)
        original_lm_config = megatron_mimo_model.language_model.config
        original_lm = megatron_mimo_model.language_model

        ddp_config = MagicMock()
        megatron_mimo_parallelism_config = self._create_megatron_mimo_parallelism_config(
            {
                "language": {"tp": 2, "dp": 2},
            }
        )

        grids = {"language": self._create_mock_grid(rank_offset=0, size=4)}
        llm_pg_collection = MagicMock()
        pg_collections = {"language": llm_pg_collection}

        wrap_megatron_mimo_model_distributed(
            megatron_mimo_model, ddp_config, megatron_mimo_parallelism_config, grids, pg_collections
        )

        # Verify DDP call arguments
        mock_ddp.assert_called_once()
        call_kwargs = mock_ddp.call_args.kwargs
        assert call_kwargs["ddp_config"] == ddp_config
        assert call_kwargs["pg_collection"] == llm_pg_collection
        assert call_kwargs["config"] == original_lm_config
        assert call_kwargs["module"] == original_lm
