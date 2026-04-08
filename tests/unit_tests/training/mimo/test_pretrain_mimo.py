# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO pretrain and setup wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cfg():
    cfg = MagicMock()
    cfg.train.rampup_batch_size = None
    cfg.train.global_batch_size = 1
    cfg.train.micro_batch_size = 1
    cfg.train.decrease_batch_size_if_needed = False
    cfg.data_parallel_size = 1
    cfg.checkpoint.load = None
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.non_persistent_ckpt_type = None
    cfg.checkpoint.save_rng = False
    return cfg


def _make_setup_output(module_to_grid_map):
    global_state = MagicMock()
    global_state.train_state.step = 0
    mock_checkpoint_manager = MagicMock()
    mock_checkpoint_manager.checkpointing_context = None
    return SimpleNamespace(
        model=MagicMock(),
        mimo_infra=SimpleNamespace(
            module_to_grid_map=module_to_grid_map,
            pg_collections={"language": MagicMock()},
        ),
        multimodule_communicator=MagicMock(),
        multimodule_pg_collection=MagicMock(),
        module_to_grid_tuple=[(MagicMock(), MagicMock())],
        optimizer=MagicMock(),
        schedulers={},
        train_data_iterator=iter([]),
        valid_data_iterator=None,
        global_state=global_state,
        checkpoint_manager=mock_checkpoint_manager,
    )


@patch(
    "megatron.bridge.training.setup_mimo.is_current_rank_in_grid",
    side_effect=lambda grid: grid.rank_offset <= 4 < (grid.rank_offset + grid.size),
)
@patch("megatron.bridge.training.setup_mimo.dist")
def test_set_mimo_random_seeds_calls_model_parallel_cuda_manual_seed(mock_dist, _mock_in_grid):
    """_set_mimo_random_seeds should derive TP/PP ranks from grids and call model_parallel_cuda_manual_seed."""
    from megatron.bridge.training.setup_mimo import _set_mimo_random_seeds

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
    "megatron.bridge.training.setup_mimo.is_current_rank_in_grid",
    side_effect=lambda grid: grid.rank_offset <= 2 < (grid.rank_offset + grid.size),
)
@patch("megatron.bridge.training.setup_mimo.dist")
def test_set_mimo_random_seeds_offsets_by_pp_rank(mock_dist, _mock_in_grid):
    """PP rank > 0 should offset the seed by 100 * pp_rank."""
    from megatron.bridge.training.setup_mimo import _set_mimo_random_seeds

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


def test_get_rng_state_namespaces_key_with_module_name():
    """get_rng_state should namespace ShardedObject key when module_name is set."""
    from megatron.bridge.training.checkpointing import get_rng_state

    pg = MagicMock()
    pg.pp.rank.return_value = 0
    pg.pp.size.return_value = 1
    pg.tp.rank.return_value = 0
    pg.tp.size.return_value = 2
    pg.dp_cp.rank.return_value = 0
    pg.dp_cp.size.return_value = 1
    pg.ep = None  # no EP

    # Without module_name: key is "rng_state"
    result = get_rng_state(False, "torch_dist", pg_collection=pg)
    assert result.key == "rng_state"

    # With module_name: key is namespaced
    result = get_rng_state(False, "torch_dist", pg_collection=pg, module_name="language")
    assert result.key == "rng_state.language"

    result = get_rng_state(False, "torch_dist", pg_collection=pg, module_name="vision")
    assert result.key == "rng_state.vision"


@patch("megatron.bridge.training.pretrain_mimo._finish_train")
@patch("megatron.bridge.training.pretrain_mimo.train_mimo")
@patch("megatron.bridge.training.pretrain_mimo.setup_mimo")
@patch("megatron.bridge.training.pretrain_mimo.dist")
@patch("megatron.bridge.training.pretrain_mimo.mimo_runtime_config_update")
@patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None)
def test_pretrain_mimo_calls_setup_and_train(
    mock_runtime_update, mock_dist, mock_setup_mimo, mock_train_mimo, mock_finish
):
    """pretrain_mimo should call setup_mimo then train_mimo."""
    from megatron.bridge.training.pretrain_mimo import pretrain_mimo

    cfg = _make_cfg()

    mock_dist.get_rank.return_value = 0
    mock_dist.is_initialized.return_value = True
    setup_output = _make_setup_output(module_to_grid_map={"language": MagicMock()})
    mock_setup_mimo.return_value = setup_output

    pretrain_mimo(
        cfg=cfg,
        forward_step_func=MagicMock(),
        build_data_iterators_fn=MagicMock(return_value=(iter([]), None)),
        global_state=MagicMock(),
    )

    mock_setup_mimo.assert_called_once()
    mock_train_mimo.assert_called_once()
    mock_finish.assert_called_once()


def test_finish_train_calls_cleanup():
    """_finish_train should finalize async saves, shut down NVRx/FT, and flush loggers."""
    from megatron.bridge.training.train import _finish_train

    global_state = MagicMock()
    checkpoint_manager = MagicMock()

    with (
        patch("megatron.bridge.training.train.safe_shutdown_nvrx_straggler_manager") as m_nvrx,
        patch("megatron.bridge.training.train.fault_tolerance") as m_ft,
        patch("megatron.bridge.training.train.destroy_global_state") as m_destroy,
    ):
        _finish_train(global_state, checkpoint_manager)

    # Async saves finalized
    checkpoint_manager.finalize_async_saves.assert_called_once_with(
        state=global_state,
        blocking=True,
        terminate=True,
    )

    # NVRx shutdown
    m_nvrx.assert_called_once_with(global_state.nvrx_straggler_manager)

    # Fault tolerance lifecycle
    m_ft.on_checkpointing_start.assert_called_once_with(global_state)
    m_ft.on_checkpointing_end.assert_called_once()
    m_ft.shutdown.assert_called_once_with(global_state)

    # Logger flush (MagicMock is truthy)
    global_state.wandb_logger.finish.assert_called_once()
    global_state._comet_logger.end.assert_called_once()

    # GlobalState destroyed
    m_destroy.assert_called_once()


@patch("megatron.bridge.training.setup_mimo.unwrap_mimo_model")
@patch("megatron.bridge.training.setup_mimo.get_model_config")
@patch("megatron.bridge.training.setup_mimo.dist")
def test_setup_mimo_asserts_when_constructor_fields_missing(mock_dist, mock_get_model_config, mock_unwrap_mimo_model):
    """setup_mimo guardrail should fail when module_to_grid_map is missing at construction."""
    from megatron.bridge.training.setup_mimo import setup_mimo

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

    # Set cfg.model to a provider that returns infra with an active grid map
    mock_infra = MagicMock()
    mock_infra.module_to_grid_map = {"language": MagicMock()}
    mock_infra.topology = {"language": []}
    mock_infra.module_output_ndim = {"language": 3}
    cfg.model.build_infra.return_value = mock_infra
    cfg.model.provide_distributed_model.return_value = [MagicMock()]

    with (
        patch("megatron.bridge.training.setup_mimo.validate_no_stub_ranks"),
        patch("megatron.bridge.training.setup_mimo._set_mimo_random_seeds"),
        patch("megatron.bridge.training.setup_mimo.build_pg_collection_for_schedule"),
        patch("megatron.bridge.training.setup_mimo.get_module_to_grid_tuple"),
        patch("megatron.bridge.training.setup_mimo.MultiModulePipelineCommunicator"),
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
    ):
        with pytest.raises(AssertionError, match="module_to_grid_map must be set"):
            setup_mimo(
                cfg=cfg,
                build_optimizer=True,
                global_state=MagicMock(),
            )
