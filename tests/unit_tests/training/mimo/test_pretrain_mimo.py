# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO pretrain entrypoint wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cfg():
    cfg = MagicMock()
    cfg.train = SimpleNamespace(
        rampup_batch_size=None,
        global_batch_size=1,
        micro_batch_size=1,
        decrease_batch_size_if_needed=False,
    )
    cfg.data_parallel_size = 1
    return cfg


def _make_setup_output(module_to_grid_map):
    return SimpleNamespace(
        model=MagicMock(),
        mimo_infra=SimpleNamespace(module_to_grid_map=module_to_grid_map),
        multimodule_communicator=MagicMock(),
        multimodule_pg_collection=MagicMock(),
        module_to_grid_tuple=[(MagicMock(), MagicMock())],
        optimizer=MagicMock(),
        schedulers={},
        train_data_iterator=iter([]),
        valid_data_iterator=None,
        global_state=MagicMock(),
    )


@patch(
    "megatron.bridge.training.pretrain_mimo.is_current_rank_in_grid",
    side_effect=lambda grid: grid.rank_offset <= 4 < (grid.rank_offset + grid.size),
)
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_set_mimo_random_seeds_calls_model_parallel_cuda_manual_seed(mock_dist, _mock_in_grid):
    """_set_mimo_random_seeds should derive TP/PP ranks from grids and call model_parallel_cuda_manual_seed."""
    from megatron.bridge.training.pretrain_mimo import _set_mimo_random_seeds

    mock_dist.get_rank.return_value = 4  # e.g. first rank of vision encoder

    # Build a mock grid: vision ranks [4,8), TP=2, PP=1
    tp_pg = MagicMock()
    pp_pg = MagicMock()
    mock_dist.get_group_rank.side_effect = lambda pg, rank: {tp_pg: 0, pp_pg: 0}[pg]

    grid = MagicMock()
    grid.rank_offset = 4
    grid.size = 4
    grid.get_pg.side_effect = lambda dims: {"tp": tp_pg, "pp": pp_pg}[dims[0]]

    mimo_infra = SimpleNamespace(module_to_grid_map={"vision": grid})
    cfg = SimpleNamespace(rng=SimpleNamespace(seed=42))

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        import torch

        with patch.object(torch.cuda, "device_count", return_value=1):
            _set_mimo_random_seeds(cfg, mimo_infra)

        # pp_rank=0, so seed stays 42. tp_rank=0 passed explicitly.
        mock_seed.assert_called_once_with(42, tp_rank=0, ep_rank=0, etp_rank=0)


@patch(
    "megatron.bridge.training.pretrain_mimo.is_current_rank_in_grid",
    side_effect=lambda grid: grid.rank_offset <= 2 < (grid.rank_offset + grid.size),
)
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_set_mimo_random_seeds_offsets_by_pp_rank(mock_dist, _mock_in_grid):
    """PP rank > 0 should offset the seed by 100 * pp_rank."""
    from megatron.bridge.training.pretrain_mimo import _set_mimo_random_seeds

    mock_dist.get_rank.return_value = 2

    tp_pg = MagicMock()
    pp_pg = MagicMock()
    # tp_rank=1, pp_rank=1
    mock_dist.get_group_rank.side_effect = lambda pg, rank: {tp_pg: 1, pp_pg: 1}[pg]

    grid = MagicMock()
    grid.rank_offset = 0
    grid.size = 4
    grid.get_pg.side_effect = lambda dims: {"tp": tp_pg, "pp": pp_pg}[dims[0]]

    mimo_infra = SimpleNamespace(module_to_grid_map={"llm": grid})
    cfg = SimpleNamespace(rng=SimpleNamespace(seed=42))

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        import torch

        with patch.object(torch.cuda, "device_count", return_value=1):
            _set_mimo_random_seeds(cfg, mimo_infra)

        # seed = 42 + 100 * 1 = 142, tp_rank=1
        mock_seed.assert_called_once_with(142, tp_rank=1, ep_rank=0, etp_rank=0)


@patch("megatron.bridge.training.pretrain_mimo.train_mimo")
@patch("megatron.bridge.training.pretrain_mimo.setup_mimo")
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_pretrain_mimo_calls_setup_and_train(mock_dist, mock_setup_mimo, mock_train_mimo):
    """pretrain_mimo should call setup_mimo then train_mimo."""
    from megatron.bridge.training.pretrain_mimo import pretrain_mimo

    cfg = _make_cfg()

    mock_dist.get_rank.return_value = 0
    setup_output = _make_setup_output(module_to_grid_map={"language": MagicMock()})
    mock_setup_mimo.return_value = setup_output

    with (
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
    ):
        pretrain_mimo(
            cfg=cfg,
            mimo_provider=MagicMock(),
            forward_step_func=MagicMock(),
            build_data_iterators_fn=MagicMock(),
            global_state=MagicMock(),
        )

    mock_setup_mimo.assert_called_once()
    mock_train_mimo.assert_called_once()


@patch("megatron.bridge.training.pretrain_mimo.unwrap_mimo_model")
@patch("megatron.bridge.training.pretrain_mimo.get_model_config")
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_setup_mimo_asserts_when_constructor_fields_missing(mock_dist, mock_get_model_config, mock_unwrap_mimo_model):
    """setup_mimo guardrail should fail when module_to_grid_map is missing at construction."""
    from megatron.bridge.training.pretrain_mimo import setup_mimo

    cfg = _make_cfg()
    mock_dist.get_rank.return_value = 0
    mock_dist.get_world_size.return_value = 8

    # Model with missing module_to_grid_map
    unwrapped_model = MagicMock()
    unwrapped_model.mimo_config = SimpleNamespace(module_to_grid_map=None)
    mock_unwrap_mimo_model.return_value = unwrapped_model

    mock_model_config = MagicMock()
    mock_model_config.pipeline_dtype = None
    mock_model_config.bf16 = True
    mock_get_model_config.return_value = mock_model_config

    # Provider that returns infra with an active grid map
    mock_provider = MagicMock()
    mock_infra = MagicMock()
    mock_infra.module_to_grid_map = {"language": MagicMock()}
    mock_infra.topology = {"language": []}
    mock_infra.module_output_ndim = {"language": 3}
    mock_provider.build_infra.return_value = mock_infra
    mock_provider.provide_distributed_model.return_value = [MagicMock()]

    with (
        patch("megatron.bridge.training.pretrain_mimo.validate_no_stub_ranks"),
        patch("megatron.bridge.training.pretrain_mimo._set_mimo_random_seeds"),
        patch("megatron.bridge.training.pretrain_mimo.build_pg_collection_for_schedule"),
        patch("megatron.bridge.training.pretrain_mimo.get_module_to_grid_tuple"),
        patch("megatron.bridge.training.pretrain_mimo.MultiModulePipelineCommunicator"),
    ):
        with pytest.raises(AssertionError, match="module_to_grid_map must be set"):
            setup_mimo(
                cfg=cfg,
                mimo_provider=mock_provider,
                build_optimizer=True,
                global_state=MagicMock(),
            )
