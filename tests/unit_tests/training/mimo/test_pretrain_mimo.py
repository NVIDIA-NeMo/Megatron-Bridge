# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO pretrain entrypoint wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cfg():
    """
    Create and return a test mock configuration with sensible default training fields.
    
    The returned object is a MagicMock with a `train` SimpleNamespace containing:
    - `rampup_batch_size = None`
    - `global_batch_size = 1`
    - `micro_batch_size = 1`
    - `decrease_batch_size_if_needed = False`
    
    Also sets `data_parallel_size = 1` on the mock.
    
    Returns:
        MagicMock: A mock configuration object with the `train` namespace and `data_parallel_size` set.
    """
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
    """
    Create a SimpleNamespace that mimics the structure returned by setup_mimo for unit tests.
    
    Parameters:
        module_to_grid_map (dict): Mapping from module name to grid object used by tests (e.g., {"vision": grid}).
    
    Returns:
        SimpleNamespace: Namespace with the following test-oriented attributes:
            - model: a MagicMock representing the model.
            - mimo_infra: SimpleNamespace with `module_to_grid_map` set to the provided mapping.
            - multimodule_communicator: a MagicMock representing inter-module communication.
            - train_data_iterator: an empty iterator for training data.
            - valid_data_iterator: None (no validation iterator).
            - global_state: a MagicMock representing global training state.
    """
    return SimpleNamespace(
        model=MagicMock(),
        mimo_infra=SimpleNamespace(module_to_grid_map=module_to_grid_map),
        multimodule_communicator=MagicMock(),
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
    cfg = SimpleNamespace(seed=42, rng=None)

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
    cfg = SimpleNamespace(seed=42, rng=None)

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        import torch

        with patch.object(torch.cuda, "device_count", return_value=1):
            _set_mimo_random_seeds(cfg, mimo_infra)

        # seed = 42 + 100 * 1 = 142, tp_rank=1
        mock_seed.assert_called_once_with(142, tp_rank=1, ep_rank=0, etp_rank=0)


@patch("megatron.bridge.training.pretrain_mimo.train_mimo")
@patch("megatron.bridge.training.pretrain_mimo.setup_mimo")
@patch("megatron.bridge.training.pretrain_mimo.unwrap_mimo_model")
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_pretrain_mimo_uses_constructor_wired_config(
    mock_dist, mock_unwrap_mimo_model, mock_setup_mimo, mock_train_mimo
):
    """Pretrain path should use constructor-wired config, not mutate it."""
    from megatron.bridge.training.pretrain_mimo import pretrain_mimo

    cfg = _make_cfg()
    opt_config = MagicMock()
    opt_config.finalize = MagicMock()
    schedulers = {}
    forward_step_func = MagicMock()

    mock_dist.get_rank.return_value = 0

    sentinel_grid_map = {"language": MagicMock()}
    setup_output = _make_setup_output(module_to_grid_map=sentinel_grid_map)
    mock_setup_mimo.return_value = setup_output

    original_grid_map = {"language": MagicMock()}
    unwrapped_model = MagicMock()
    unwrapped_model.mimo_config = SimpleNamespace(
        module_to_grid_map=original_grid_map,
    )
    mock_unwrap_mimo_model.return_value = unwrapped_model

    with (
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
        patch("megatron.core.models.mimo.optimizer.get_mimo_optimizer") as mock_get_optimizer,
    ):
        mock_get_optimizer.return_value = MagicMock()

        pretrain_mimo(
            cfg=cfg,
            mimo_provider=MagicMock(),
            forward_step_func=forward_step_func,
            build_data_iterators_fn=MagicMock(),
            opt_config=opt_config,
            schedulers=schedulers,
            global_state=MagicMock(),
        )

    # No post-construction mutation: keep original references/values.
    assert unwrapped_model.mimo_config.module_to_grid_map is original_grid_map
    mock_train_mimo.assert_called_once()


@patch("megatron.bridge.training.pretrain_mimo.setup_mimo")
@patch("megatron.bridge.training.pretrain_mimo.unwrap_mimo_model")
@patch("megatron.bridge.training.pretrain_mimo.dist")
def test_pretrain_mimo_asserts_when_constructor_fields_missing(mock_dist, mock_unwrap_mimo_model, mock_setup_mimo):
    """Guardrail should fail when constructor-time config wiring is missing."""
    from megatron.bridge.training.pretrain_mimo import pretrain_mimo

    cfg = _make_cfg()
    opt_config = MagicMock()
    opt_config.finalize = MagicMock()

    mock_dist.get_rank.return_value = 0

    # Infra indicates MIMO-parallel path is active.
    mock_setup_mimo.return_value = _make_setup_output(module_to_grid_map={"language": MagicMock()})

    # Missing constructor-wired fields should trigger assertion.
    unwrapped_model = MagicMock()
    unwrapped_model.mimo_config = SimpleNamespace(
        module_to_grid_map=None,
    )
    mock_unwrap_mimo_model.return_value = unwrapped_model

    with (
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
    ):
        with pytest.raises(AssertionError, match="module_to_grid_map must be set"):
            pretrain_mimo(
                cfg=cfg,
                mimo_provider=MagicMock(),
                forward_step_func=MagicMock(),
                build_data_iterators_fn=MagicMock(),
                opt_config=opt_config,
                schedulers={},
                global_state=MagicMock(),
            )
