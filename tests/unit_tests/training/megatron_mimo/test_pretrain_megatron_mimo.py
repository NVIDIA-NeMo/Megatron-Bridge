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
    language_pg = MagicMock()
    return SimpleNamespace(
        model=MagicMock(),
        megatron_mimo_infra=SimpleNamespace(
            module_to_grid_map=module_to_grid_map,
            pg_collections={"language": language_pg},
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
        active_module_name="language",
        local_pg_collection=language_pg,
    )


@patch(
    "megatron.bridge.training.setup_megatron_mimo.is_current_rank_in_grid",
    side_effect=lambda grid: grid.rank_offset <= 4 < (grid.rank_offset + grid.size),
)
@patch("megatron.bridge.training.setup_megatron_mimo.dist")
def test_set_megatron_mimo_random_seeds_calls_model_parallel_cuda_manual_seed(mock_dist, _mock_in_grid):
    """_set_megatron_mimo_random_seeds should derive TP/PP ranks from grids and call model_parallel_cuda_manual_seed."""
    from megatron.bridge.training.setup_megatron_mimo import _set_megatron_mimo_random_seeds

    mock_dist.get_rank.return_value = 4  # e.g. first rank of vision encoder

    # Build a mock grid: vision ranks [4,8), TP=2, PP=1
    tp_pg = MagicMock()
    pp_pg = MagicMock()
    mock_dist.get_group_rank.side_effect = lambda pg, rank: {tp_pg: 0, pp_pg: 0}[pg]

    grid = MagicMock()
    grid.rank_offset = 4
    grid.size = 4
    grid.get_pg.side_effect = lambda dims: {"tp": tp_pg, "pp": pp_pg}[dims[0]]

    megatron_mimo_infra = SimpleNamespace(module_to_grid_map={"vision": grid})
    cfg = SimpleNamespace(rng=SimpleNamespace(seed=42))

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        import torch

        with patch.object(torch.cuda, "device_count", return_value=1):
            _set_megatron_mimo_random_seeds(cfg, megatron_mimo_infra)

        # pp_rank=0, so seed stays 42. tp_rank=0 passed explicitly.
        mock_seed.assert_called_once_with(42, tp_rank=0, ep_rank=0, etp_rank=0)


@patch(
    "megatron.bridge.training.setup_megatron_mimo.is_current_rank_in_grid",
    side_effect=lambda grid: grid.rank_offset <= 2 < (grid.rank_offset + grid.size),
)
@patch("megatron.bridge.training.setup_megatron_mimo.dist")
def test_set_megatron_mimo_random_seeds_offsets_by_pp_rank(mock_dist, _mock_in_grid):
    """PP rank > 0 should offset the seed by 100 * pp_rank."""
    from megatron.bridge.training.setup_megatron_mimo import _set_megatron_mimo_random_seeds

    mock_dist.get_rank.return_value = 2

    tp_pg = MagicMock()
    pp_pg = MagicMock()
    # tp_rank=1, pp_rank=1
    mock_dist.get_group_rank.side_effect = lambda pg, rank: {tp_pg: 1, pp_pg: 1}[pg]

    grid = MagicMock()
    grid.rank_offset = 0
    grid.size = 4
    grid.get_pg.side_effect = lambda dims: {"tp": tp_pg, "pp": pp_pg}[dims[0]]

    megatron_mimo_infra = SimpleNamespace(module_to_grid_map={"llm": grid})
    cfg = SimpleNamespace(rng=SimpleNamespace(seed=42))

    with patch("megatron.core.tensor_parallel.model_parallel_cuda_manual_seed") as mock_seed:
        import torch

        with patch.object(torch.cuda, "device_count", return_value=1):
            _set_megatron_mimo_random_seeds(cfg, megatron_mimo_infra)

        # seed = 42 + 100 * 1 = 142, tp_rank=1
        mock_seed.assert_called_once_with(142, tp_rank=1, ep_rank=0, etp_rank=0)


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


@patch("megatron.bridge.training.pretrain_megatron_mimo._finish_train")
@patch("megatron.bridge.training.pretrain_megatron_mimo.train_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.setup_megatron_mimo")
@patch("megatron.bridge.training.pretrain_megatron_mimo.dist")
@patch("megatron.bridge.training.pretrain_megatron_mimo.megatron_mimo_runtime_config_update")
@patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None)
@patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None)
def test_pretrain_megatron_mimo_threads_active_module_from_setup(
    mock_runtime_update, mock_dist, mock_setup_megatron_mimo, mock_train_megatron_mimo, mock_finish
):
    """pretrain_megatron_mimo must pass active_module_name and local_pg_collection
    from setup_output into train_megatron_mimo. Prevents reintroduction of the
    rank-dependent inline recompute that fails in colocated mode."""
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

    _, kwargs = mock_train_megatron_mimo.call_args
    assert kwargs["active_module_name"] == setup_output.active_module_name
    assert kwargs["local_pg_collection"] is setup_output.local_pg_collection


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
    cfg.model.build_infra.return_value = mock_infra
    cfg.model.provide_distributed_model.return_value = [MagicMock()]

    with (
        patch("megatron.bridge.training.setup_megatron_mimo.validate_no_stub_ranks"),
        patch("megatron.bridge.training.setup_megatron_mimo._set_megatron_mimo_random_seeds"),
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


# ---------------------------------------------------------------------------
# Tests: train_step_megatron_mimo schedule dispatch
# ---------------------------------------------------------------------------


def _make_mimo_model_stub(mode, lm_has_pp: bool):
    """Build a minimal MimoModel-like stub for dispatch tests."""
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    stub = MagicMock()
    stub.role.mode = mode
    stub.lm_has_pp = lm_has_pp
    stub.role.has_language_module = True
    stub.role.is_last_stage.return_value = False
    # Make MIMO_LANGUAGE_MODULE_KEY resolvable for post-schedule loss code paths
    _ = MIMO_LANGUAGE_MODULE_KEY  # suppress unused warning
    return stub


def _make_minimal_train_step_kwargs():
    """Minimal kwargs for train_step_megatron_mimo; values only need to reach the dispatch."""
    return dict(
        forward_step_func=MagicMock(),
        data_iterator=iter([]),
        model=MagicMock(),
        optimizer=MagicMock(),
        schedulers={},
        global_state=MagicMock(),
        multimodule_communicator=MagicMock(),
        multimodule_pg_collection=MagicMock(),
        infra=SimpleNamespace(pg_collections={"language": MagicMock()}),
        module_to_grid_tuple=[],
        num_microbatches=1,
        seq_length=4,
        micro_batch_size=1,
    )


def test_train_step_raises_for_colocated_with_llm_pp_gt_one():
    """Colocated + LLM PP>1 must raise NotImplementedError until the three-phase schedule is wired."""
    from megatron.core.models.mimo.config.role import ModuleLayout

    from megatron.bridge.training.train_megatron_mimo import train_step_megatron_mimo

    stub = _make_mimo_model_stub(mode=ModuleLayout.COLOCATED, lm_has_pp=True)
    with patch("megatron.bridge.training.train_megatron_mimo.unwrap_megatron_mimo_model", return_value=stub):
        with pytest.raises(NotImplementedError, match="three-phase schedule"):
            train_step_megatron_mimo(**_make_minimal_train_step_kwargs())


@pytest.mark.parametrize(
    "mode_name, lm_has_pp",
    [
        ("COLOCATED", False),  # colocated + LLM PP=1: standard schedule
        ("NON_COLOCATED", False),  # non-colocated + LLM PP=1
        (
            "NON_COLOCATED",
            True,
        ),  # non-colocated + LLM PP>1: standard schedule (MultiModulePipelineCommunicator handles it)
    ],
    ids=["colocated_pp1", "non_colocated_pp1", "non_colocated_pp_gt1"],
)
def test_train_step_non_three_phase_cases_dispatch_to_standard_schedule(mode_name, lm_has_pp):
    """Every mode that isn't colocated-PP>1 should reach forward_backward_pipelining_without_interleaving."""
    from megatron.core.models.mimo.config.role import ModuleLayout

    from megatron.bridge.training.train_megatron_mimo import train_step_megatron_mimo

    stub = _make_mimo_model_stub(mode=getattr(ModuleLayout, mode_name), lm_has_pp=lm_has_pp)
    # Use a sentinel exception thrown from the mocked schedule as the "got past the
    # dispatch" marker. This avoids exercising downstream code paths (loss
    # aggregation, torch.distributed calls) that aren't the subject under test.
    sentinel = RuntimeError("DISPATCH_REACHED_STANDARD_SCHEDULE")
    with (
        patch("megatron.bridge.training.train_megatron_mimo.unwrap_megatron_mimo_model", return_value=stub),
        patch(
            "megatron.bridge.training.train_megatron_mimo.forward_backward_pipelining_without_interleaving",
            side_effect=sentinel,
        ) as mock_fb,
        patch("megatron.bridge.training.train_megatron_mimo.zero_grad_buffer_for_multimodule"),
    ):
        with pytest.raises(RuntimeError, match="DISPATCH_REACHED_STANDARD_SCHEDULE"):
            train_step_megatron_mimo(**_make_minimal_train_step_kwargs())
        assert mock_fb.called, f"standard schedule not called for mode={mode_name}, lm_has_pp={lm_has_pp}"
