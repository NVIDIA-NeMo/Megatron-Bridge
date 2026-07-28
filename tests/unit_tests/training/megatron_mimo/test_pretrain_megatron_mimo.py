# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MegatronMIMO pretrain and setup wiring."""

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
    cfg.rng.seed = 1234
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
        megatron_mimo_infra=SimpleNamespace(
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


@patch("megatron.bridge.models.megatron_mimo.build_model.dist")
def test_set_megatron_mimo_random_seeds_calls_model_parallel_cuda_manual_seed(mock_dist):
    """_set_per_module_random_seeds should derive TP/PP/EP/ETP ranks from the active module PGC."""
    from megatron.bridge.models.megatron_mimo.build_model import _set_per_module_random_seeds

    mock_dist.get_rank.return_value = 4  # e.g. first rank of vision encoder

    tp_pg = MagicMock()
    pp_pg = MagicMock()
    ep_pg = MagicMock()
    expt_tp_pg = MagicMock()
    mock_dist.get_group_rank.side_effect = lambda pg, rank: {
        tp_pg: 0,
        pp_pg: 0,
        ep_pg: 0,
        expt_tp_pg: 0,
    }[pg]

    grid = MagicMock()
    grid.rank_offset = 4
    grid.size = 4
    grid.is_current_rank_in_grid.return_value = True
    pg_collection = SimpleNamespace(tp=tp_pg, pp=pp_pg, ep=ep_pg, expt_tp=expt_tp_pg)

    megatron_mimo_infra = SimpleNamespace(
        module_to_grid_map={"vision": grid},
        pg_collections={"vision": pg_collection},
    )

    with patch(
        "megatron.bridge.models.megatron_mimo.build_model.tensor_parallel.model_parallel_cuda_manual_seed"
    ) as mock_seed:
        import torch

        with patch.object(torch.cuda, "device_count", return_value=1):
            _set_per_module_random_seeds(megatron_mimo_infra, seed=42)

        # EP=ETP=1 resolves to rank 0/0, preserving dense behavior.
        mock_seed.assert_called_once_with(42, tp_rank=0, ep_rank=0, etp_rank=0)


@patch("megatron.bridge.models.megatron_mimo.build_model.dist")
def test_set_megatron_mimo_random_seeds_offsets_by_pp_rank(mock_dist):
    """PP rank offsets the seed while EP/ETP ranks come from the active module PGC."""
    from megatron.bridge.models.megatron_mimo.build_model import _set_per_module_random_seeds

    mock_dist.get_rank.return_value = 2

    tp_pg = MagicMock()
    pp_pg = MagicMock()
    ep_pg = MagicMock()
    expt_tp_pg = MagicMock()
    mock_dist.get_group_rank.side_effect = lambda pg, rank: {
        tp_pg: 1,
        pp_pg: 1,
        ep_pg: 2,
        expt_tp_pg: 3,
    }[pg]

    grid = MagicMock()
    grid.rank_offset = 0
    grid.size = 4
    grid.is_current_rank_in_grid.return_value = True
    pg_collection = SimpleNamespace(tp=tp_pg, pp=pp_pg, ep=ep_pg, expt_tp=expt_tp_pg)

    megatron_mimo_infra = SimpleNamespace(
        module_to_grid_map={"llm": grid},
        pg_collections={"llm": pg_collection},
    )

    with patch(
        "megatron.bridge.models.megatron_mimo.build_model.tensor_parallel.model_parallel_cuda_manual_seed"
    ) as mock_seed:
        import torch

        with patch.object(torch.cuda, "device_count", return_value=1):
            _set_per_module_random_seeds(megatron_mimo_infra, seed=42)

        # seed = 42 + 100 * 1 = 142
        mock_seed.assert_called_once_with(142, tp_rank=1, ep_rank=2, etp_rank=3)


def test_bridge_parallel_state_globals_sets_expert_groups_from_pg_collection():
    """Compatibility globals should mirror the expert groups on the active module PGC."""
    from megatron.core import parallel_state as mpu

    from megatron.bridge.models.megatron_mimo.build_model import _bridge_parallel_state_globals

    global_names = (
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP_WITH_CP",
        "_PIPELINE_MODEL_PARALLEL_GROUP",
        "_EXPERT_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP",
        "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        "_CONTEXT_PARALLEL_GROUP",
        "_MODEL_PARALLEL_GROUP",
    )
    saved_globals = {name: getattr(mpu, name) for name in global_names}

    groups = {
        name: object()
        for name in (
            "tp",
            "dp",
            "dp_cp",
            "pp",
            "ep",
            "expt_tp",
            "expt_dp",
            "intra_expt_dp",
            "tp_ep",
            "tp_ep_pp",
            "cp",
            "mp",
        )
    }
    pg_collection = SimpleNamespace(**groups)

    try:
        _bridge_parallel_state_globals(pg_collection)

        assert mpu._TENSOR_MODEL_PARALLEL_GROUP is groups["tp"]
        assert mpu._DATA_PARALLEL_GROUP is groups["dp"]
        assert mpu._DATA_PARALLEL_GROUP_WITH_CP is groups["dp_cp"]
        assert mpu._PIPELINE_MODEL_PARALLEL_GROUP is groups["pp"]
        assert mpu._EXPERT_MODEL_PARALLEL_GROUP is groups["ep"]
        assert mpu._EXPERT_TENSOR_PARALLEL_GROUP is groups["expt_tp"]
        assert mpu._EXPERT_TENSOR_PARALLEL_GROUP is not groups["tp"]
        assert mpu._EXPERT_DATA_PARALLEL_GROUP is groups["expt_dp"]
        assert mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP is groups["intra_expt_dp"]
        assert mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is groups["tp_ep"]
        assert mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is groups["tp_ep_pp"]
        assert mpu._CONTEXT_PARALLEL_GROUP is groups["cp"]
        assert mpu._MODEL_PARALLEL_GROUP is groups["mp"]
    finally:
        for name, value in saved_globals.items():
            setattr(mpu, name, value)


def test_get_rng_state_namespaces_key_with_module_name():
    """get_rng_state should namespace ShardedObject key when module_name is set.

    Unit test: mocks ``torch.cuda.get_rng_state`` and the Megatron CUDA RNG
    tracker so the test runs without a GPU (``get_rng_state`` otherwise calls
    these unconditionally at ``checkpointing.py:425-426``).
    """
    from megatron.bridge.training.checkpointing import get_rng_state

    pg = MagicMock()
    pg.pp.rank.return_value = 0
    pg.pp.size.return_value = 1
    pg.tp.rank.return_value = 0
    pg.tp.size.return_value = 2
    pg.dp_cp.rank.return_value = 0
    pg.dp_cp.size.return_value = 1
    pg.ep = None  # no EP

    with (
        patch("torch.cuda.get_rng_state", return_value=b"dummy_cuda_rng_state"),
        patch("megatron.bridge.training.checkpointing.tensor_parallel.get_cuda_rng_tracker") as mock_tracker,
    ):
        mock_tracker.return_value.get_states.return_value = {}

        # Without module_name: key is "rng_state"
        result = get_rng_state(False, "torch_dist", pg_collection=pg)
        assert result.key == "rng_state"

        # With module_name: key is namespaced
        result = get_rng_state(False, "torch_dist", pg_collection=pg, module_name="language")
        assert result.key == "rng_state.language"

        result = get_rng_state(False, "torch_dist", pg_collection=pg, module_name="vision")
        assert result.key == "rng_state.vision"


@patch("megatron.bridge.training.pretrain_megatron_mimo._finish_train")
@patch("megatron.bridge.training.pretrain_megatron_mimo.train_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.setup_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.dist")
@patch("megatron.bridge.training.pretrain_megatron_mimo.megatron_mimo_runtime_config_update")
@patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None)
def test_pretrain_megatron_mimo_calls_setup_and_train(
    mock_runtime_update, mock_dist, mock_setup_megatron_mimo, mock_train_megatron_mimo, mock_finish
):
    """pretrain_megatron_mimo should call setup_megatron_mimo then train_megatron_mimo."""
    from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo

    cfg = _make_cfg()

    mock_dist.get_rank.return_value = 0
    mock_dist.is_initialized.return_value = True
    setup_output = _make_setup_output(module_to_grid_map={"language": MagicMock()})
    mock_setup_megatron_mimo.return_value = setup_output

    pretrain_megatron_mimo(
        cfg=cfg,
        forward_step_func=MagicMock(),
        build_data_iterators_fn=MagicMock(return_value=(iter([]), None)),
        global_state=MagicMock(),
    )

    mock_setup_megatron_mimo.assert_called_once()
    mock_train_megatron_mimo.assert_called_once()
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

    # Fault tolerance lifecycle — mirror the exact contract at train.py:1445-1448.
    m_ft.on_checkpointing_start.assert_called_once_with(global_state)
    m_ft.on_checkpointing_end.assert_called_once_with(global_state=global_state, is_async_finalization=True)
    m_ft.shutdown.assert_called_once_with(global_state)

    # Logger flush (MagicMock is truthy)
    global_state.wandb_logger.finish.assert_called_once()
    global_state._comet_logger.end.assert_called_once()

    # GlobalState destroyed
    m_destroy.assert_called_once()


@patch("megatron.bridge.training.setup_megatron_mimo.unwrap_megatron_mimo_model")
@patch("megatron.bridge.training.setup_megatron_mimo.get_model_config")
@patch("megatron.bridge.training.setup_megatron_mimo.dist")
def test_setup_megatron_mimo_asserts_when_constructor_fields_missing(
    mock_dist, mock_get_model_config, mock_unwrap_megatron_mimo_model
):
    """setup_megatron_mimo guardrail should fail when module_to_grid_map is missing at construction."""
    from megatron.bridge.training.setup_megatron_mimo import setup_megatron_mimo

    cfg = _make_cfg()
    mock_dist.get_rank.return_value = 0
    mock_dist.get_world_size.return_value = 8

    # Model with missing module_to_grid_map
    unwrapped_model = MagicMock()
    unwrapped_model.mimo_config = SimpleNamespace(module_to_grid_map=None)
    mock_unwrap_megatron_mimo_model.return_value = unwrapped_model

    mock_model_config = MagicMock()
    mock_model_config.pipeline_dtype = None
    mock_model_config.bf16 = True
    mock_get_model_config.return_value = mock_model_config

    # Set cfg.model to a provider that returns infra with an active grid map
    mock_infra = MagicMock()
    mock_infra.module_to_grid_map = {"language": MagicMock()}
    mock_infra.topology = {"language": []}
    mock_infra.module_output_ndim = {"language": 3}
    model = MagicMock()

    with (
        patch(
            "megatron.bridge.models.megatron_mimo.build_megatron_mimo_model", return_value=(model, mock_infra)
        ) as mock_build_model,
        patch("megatron.bridge.training.setup_megatron_mimo.build_pg_collection_for_schedule"),
        patch("megatron.bridge.training.setup_megatron_mimo.get_module_to_grid_tuple"),
        patch("megatron.bridge.training.setup_megatron_mimo.MultiModulePipelineCommunicator"),
        patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
        patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
    ):
        mock_state = MagicMock()
        mock_state.cfg = cfg
        with pytest.raises(AssertionError, match="module_to_grid_map must be set"):
            setup_megatron_mimo(state=mock_state)

    mock_build_model.assert_called_once()
    assert mock_build_model.call_args.kwargs["seed"] == cfg.rng.seed
