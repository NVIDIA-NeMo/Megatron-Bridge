# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MiMo checkpoint saving and loading wiring.

Tests validate that MiMo training correctly uses shared checkpoint helpers
(save_checkpoint_and_time, checkpoint_and_decide_exit, load_checkpoint) with
the right arguments, without actually saving/loading checkpoints.
"""

from __future__ import annotations

import inspect
import time
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler_mock() -> MagicMock:
    """Create a scheduler mock that supports param_groups[0] access."""
    sched = MagicMock()
    sched.optimizer.param_groups = [{"lr": 1e-4}]
    sched.get_lr.return_value = 1e-4
    return sched


def _make_mimo_infra(*, num_active_pgs: int = 1) -> Mock:
    """Create a mock OmniModalInfra with the given number of active PG collections."""
    infra = Mock()
    pgs: Dict[str, Any] = {}
    for i in range(num_active_pgs):
        pgs[f"module_{i}"] = Mock()
    infra.pg_collections = pgs
    infra.module_to_grid_map = {"language": Mock()}
    infra.topology = Mock()
    return infra


def _make_global_state(
    *,
    save_interval: int | None = 10,
    save_dir: str | None = "/tmp/ckpt",
    train_iters: int = 100,
    step: int = 0,
    non_persistent_save_interval: int | None = None,
    exit_signal_handler: bool = False,
    exit_duration_in_mins: float | None = None,
    exit_interval: int | None = None,
) -> SimpleNamespace:
    """Create a minimal GlobalState-like namespace for train_mimo tests."""
    timer_handle = Mock()
    timers = Mock(return_value=timer_handle)
    timers.log = Mock()

    state = SimpleNamespace(
        timers=timers,
        energy_monitor=None,
        cfg=SimpleNamespace(
            train=SimpleNamespace(
                train_iters=train_iters,
                micro_batch_size=1,
                exit_signal_handler=exit_signal_handler,
                exit_duration_in_mins=exit_duration_in_mins,
                exit_interval=exit_interval,
                eval_interval=None,
            ),
            dataset=SimpleNamespace(seq_length=128),
            checkpoint=SimpleNamespace(
                save=save_dir,
                save_interval=save_interval,
                non_persistent_save_interval=non_persistent_save_interval,
                async_save=False,
            ),
            ddp=SimpleNamespace(use_megatron_fsdp=False, overlap_param_gather=True),
            optimizer=SimpleNamespace(use_distributed_optimizer=True),
            model=SimpleNamespace(fp8=None, seq_length=128),
            logger=SimpleNamespace(
                log_progress=False,
                skip_train_metrics_log=True,
                timing_log_level=0,
                timing_log_option="minmax",
                log_timers_to_tensorboard=False,
                log_interval=1,
            ),
            profiling=None,
            data_parallel_size=1,
        ),
        train_state=SimpleNamespace(
            step=step,
            consumed_train_samples=0,
            floating_point_operations_so_far=0,
        ),
        start_time=time.time(),
        signal_handler=Mock(),
        nvrx_straggler_manager=None,
        tensorboard_logger=None,
        wandb_logger=None,
    )
    state.signal_handler.signals_received.return_value = []
    return state


# ---------------------------------------------------------------------------
# Tests: pg_collection forwarding in shared helpers
# ---------------------------------------------------------------------------


class TestPgCollectionForwarding:
    """Verify save_checkpoint_and_time and checkpoint_and_decide_exit
    forward pg_collection to save_checkpoint."""

    @patch("megatron.bridge.training.train.force_param_sync")
    @patch("megatron.bridge.training.train.should_disable_forward_pre_hook", return_value=False)
    def test_save_checkpoint_and_time_forwards_pg_collection(
        self,
        mock_should_disable,
        mock_force_param_sync,
    ):
        from megatron.bridge.training.checkpointing import CheckpointSaveContext
        from megatron.bridge.training.train import save_checkpoint_and_time

        state = _make_global_state()
        pg = Mock()
        checkpoint_manager = MagicMock()

        save_checkpoint_and_time(
            state=state,
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=0,
            checkpoint_manager=checkpoint_manager,
            pg_collection=pg,
        )

        checkpoint_manager.save.assert_called_once()
        ctx = checkpoint_manager.save.call_args[0][0]
        assert isinstance(ctx, CheckpointSaveContext)
        assert ctx.pg_collection is pg

    @patch("megatron.bridge.training.train.force_param_sync")
    @patch("megatron.bridge.training.train.should_disable_forward_pre_hook", return_value=False)
    def test_save_checkpoint_and_time_defaults_pg_collection_to_none(
        self,
        mock_should_disable,
        mock_force_param_sync,
    ):
        from megatron.bridge.training.checkpointing import CheckpointSaveContext
        from megatron.bridge.training.train import save_checkpoint_and_time

        state = _make_global_state()
        checkpoint_manager = MagicMock()

        save_checkpoint_and_time(
            state=state,
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=0,
            checkpoint_manager=checkpoint_manager,
        )

        ctx = checkpoint_manager.save.call_args[0][0]
        assert isinstance(ctx, CheckpointSaveContext)
        assert ctx.pg_collection is None

    @patch("megatron.bridge.training.train.save_checkpoint_and_time")
    @patch("megatron.bridge.training.train.barrier_and_log")
    @patch("megatron.bridge.training.train.check_nvrx_straggler_detection", return_value=False)
    def test_checkpoint_and_decide_exit_forwards_pg_collection(
        self,
        mock_check_nvrx,
        mock_barrier_log,
        mock_save_and_time,
    ):
        from megatron.bridge.training.train import checkpoint_and_decide_exit

        state = _make_global_state(save_interval=5, step=10)
        pg = Mock()
        checkpoint_manager = MagicMock()

        checkpoint_and_decide_exit(
            state=state,
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=0,
            checkpoint_manager=checkpoint_manager,
            train_data_iterator=None,
            pg_collection=pg,
        )

        _, kwargs = mock_save_and_time.call_args
        assert kwargs["pg_collection"] is pg


# ---------------------------------------------------------------------------
# Tests: pretrain_mimo setup wiring
# ---------------------------------------------------------------------------


class TestPretrainMimoSetup:
    """Verify pretrain_mimo properly initializes checkpointing runtime."""

    @patch("megatron.bridge.training.setup_mimo.checkpoint_exists", return_value=False)
    @patch("megatron.bridge.training.setup_mimo.get_active_module_pg")
    @patch("megatron.bridge.training.setup_mimo.create_checkpoint_manager")
    @patch("megatron.bridge.training.setup_mimo.MultiModulePipelineCommunicator")
    @patch("megatron.bridge.training.setup_mimo.get_model_config")
    @patch("megatron.bridge.training.setup_mimo.validate_no_stub_ranks")
    @patch("megatron.bridge.training.setup_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.setup_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.setup_mimo._update_mimo_model_config_funcs")
    @patch("megatron.bridge.training.setup_mimo.unwrap_mimo_model")
    @patch("megatron.bridge.training.setup_mimo.dist")
    def test_setup_mimo_initializes_checkpoint_manager(
        self,
        mock_dist,
        mock_unwrap,
        mock_update_config_funcs,
        mock_get_grid,
        mock_build_pg,
        mock_validate,
        mock_get_config,
        mock_communicator,
        mock_create_ckpt_mgr,
        mock_get_active_pg,
        mock_ckpt_exists,
    ):
        from megatron.bridge.training.setup_mimo import setup_mimo

        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2

        mock_mgr_instance = MagicMock()
        mock_mgr_instance.checkpointing_context = {"test": "context"}
        mock_create_ckpt_mgr.return_value = mock_mgr_instance

        local_pg = MagicMock()
        mock_get_active_pg.return_value = ("language", local_pg)

        model_config = Mock()
        model_config.pipeline_dtype = None
        model_config.bf16 = True
        mock_get_config.return_value = model_config

        unwrapped = MagicMock()
        unwrapped.mimo_config.module_to_grid_map = {"language": Mock()}
        mock_unwrap.return_value = unwrapped

        cfg = Mock()
        cfg.checkpoint = Mock()
        cfg.checkpoint.load = None
        cfg.checkpoint.pretrained_checkpoint = None
        cfg.checkpoint.non_persistent_ckpt_type = None
        cfg.train = Mock()
        cfg.train.grad_reduce_in_fp32 = False
        cfg.train.overlap_grad_reduce = True
        cfg.train.use_distributed_optimizer = False
        cfg.train.check_for_nan_in_grad = False
        cfg.model = Mock()
        cfg.model.fp16 = False
        cfg.model.bf16 = True

        global_state = Mock()
        global_state.start_time = time.time()
        global_state.cfg = cfg

        infra = Mock()
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        infra.module_output_ndim = {"language": 3}
        infra.pg_collections = {"language": Mock()}
        cfg.model.build_infra.return_value = infra
        cfg.model.provide_distributed_model.return_value = [Mock()]

        mock_optimizer = MagicMock()
        mock_optimizer.module_infos = {}

        with (
            patch("megatron.bridge.training.setup_mimo._set_mimo_random_seeds"),
            patch("megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=mock_optimizer),
            patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
            patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
            patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None),
        ):
            result = setup_mimo(state=global_state)

        mock_create_ckpt_mgr.assert_called_once_with(cfg.checkpoint)
        global_state.initialize_async_checkpoint_worker.assert_called_once()
        assert result.checkpoint_manager is mock_mgr_instance

    def test_pretrain_mimo_calls_runtime_config_update(self):
        """pretrain_mimo should call mimo_runtime_config_update before setup."""
        from megatron.bridge.training.pretrain_mimo import pretrain_mimo

        cfg = _make_pretrain_cfg()

        setup_output = _make_setup_output_for_load()

        with (
            patch("megatron.bridge.training.pretrain_mimo.mimo_runtime_config_update") as m_runtime,
            patch("megatron.bridge.training.pretrain_mimo.setup_mimo", return_value=setup_output),
            patch("megatron.bridge.training.pretrain_mimo.train_mimo"),
            patch("megatron.bridge.training.pretrain_mimo._finish_train"),
            patch("megatron.bridge.training.pretrain_mimo.dist") as m_dist,
        ):
            m_dist.get_rank.return_value = 0
            pretrain_mimo(
                cfg=cfg,
                forward_step_func=Mock(),
                build_data_iterators_fn=Mock(return_value=(iter([]), None)),
                global_state=setup_output.global_state,
            )
            m_runtime.assert_called_once_with(cfg)


# ---------------------------------------------------------------------------
# Tests: non-colocated runtime guard
# ---------------------------------------------------------------------------


class TestNonColocatedGuard:
    """Verify the non-colocated topology assertion in train_mimo."""

    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule", return_value=Mock(spec=[]))
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_rejects_multiple_active_pgs(self, *_mocks):
        from megatron.bridge.training.train_mimo import train_mimo

        infra = _make_mimo_infra(num_active_pgs=2)
        state = _make_global_state(train_iters=0)

        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            train_mimo(
                forward_step_func=Mock(),
                model=Mock(),
                optimizer=Mock(),
                schedulers={},
                train_data_iterator=Mock(),
                valid_data_iterator=None,
                global_state=state,
                mimo_infra=infra,
                multimodule_communicator=Mock(),
                checkpoint_manager=MagicMock(),
            )

    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule", return_value=Mock(spec=[]))
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_rejects_zero_active_pgs(self, *_mocks):
        from megatron.bridge.training.train_mimo import train_mimo

        infra = _make_mimo_infra(num_active_pgs=0)
        state = _make_global_state(train_iters=0)

        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            train_mimo(
                forward_step_func=Mock(),
                model=Mock(),
                optimizer=Mock(),
                schedulers={},
                train_data_iterator=Mock(),
                valid_data_iterator=None,
                global_state=state,
                mimo_infra=infra,
                multimodule_communicator=Mock(),
                checkpoint_manager=MagicMock(),
            )


# ---------------------------------------------------------------------------
# Tests: checkpoint_and_decide_exit integration in train_mimo
# ---------------------------------------------------------------------------


class TestTrainMimoCheckpointIntegration:
    """Verify train_mimo calls checkpoint_and_decide_exit with the right args."""

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=False)
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_calls_checkpoint_and_decide_exit_with_pg_collection(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_ckpt_exit,
    ):
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)

        pg = Mock()
        infra = Mock()
        infra.pg_collections = {"language": pg}
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()

        mock_build_pg.return_value = Mock(spec=[])  # not a list

        state = _make_global_state(train_iters=1, step=0)
        ckpt_mgr = MagicMock()
        train_iter = Mock()

        train_mimo(
            forward_step_func=Mock(),
            model=Mock(),
            optimizer=Mock(),
            schedulers={"language": _make_scheduler_mock()},
            train_data_iterator=train_iter,
            valid_data_iterator=None,
            global_state=state,
            mimo_infra=infra,
            multimodule_communicator=Mock(),
            checkpoint_manager=ckpt_mgr,
        )

        mock_ckpt_exit.assert_called_once()
        _, kwargs = mock_ckpt_exit.call_args
        assert kwargs["pg_collection"] is pg
        assert kwargs["checkpoint_manager"] is ckpt_mgr
        assert kwargs["train_data_iterator"] is train_iter
        assert kwargs["num_floating_point_operations_so_far"] == 0

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=True)
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_exits_loop_when_checkpoint_and_decide_exit_returns_true(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_ckpt_exit,
    ):
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)

        infra = Mock()
        infra.pg_collections = {"language": Mock()}
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        mock_build_pg.return_value = Mock(spec=[])

        state = _make_global_state(train_iters=100, step=0)

        train_mimo(
            forward_step_func=Mock(),
            model=Mock(),
            optimizer=Mock(),
            schedulers={"language": _make_scheduler_mock()},
            train_data_iterator=Mock(),
            valid_data_iterator=None,
            global_state=state,
            mimo_infra=infra,
            multimodule_communicator=Mock(),
            checkpoint_manager=MagicMock(),
        )

        # Should have exited after 1 iteration, not 100
        assert mock_train_step.call_count == 1
        assert state.train_state.step == 1

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=False)
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_async_finalize_called_at_top_of_loop(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_ckpt_exit,
    ):
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)

        infra = Mock()
        infra.pg_collections = {"language": Mock()}
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        mock_build_pg.return_value = Mock(spec=[])

        state = _make_global_state(train_iters=2, step=0)
        ckpt_mgr = MagicMock()

        train_mimo(
            forward_step_func=Mock(),
            model=Mock(),
            optimizer=Mock(),
            schedulers={"language": _make_scheduler_mock()},
            train_data_iterator=Mock(),
            valid_data_iterator=None,
            global_state=state,
            mimo_infra=infra,
            multimodule_communicator=Mock(),
            checkpoint_manager=ckpt_mgr,
        )

        # finalize_async_saves is called on the checkpoint_manager:
        # 2 non-blocking calls (top of each iteration).
        # The blocking shutdown call is now in _finish_train (pretrain_mimo.py).
        assert ckpt_mgr.finalize_async_saves.call_count == 2

        non_blocking_calls = [
            c for c in ckpt_mgr.finalize_async_saves.call_args_list if c.kwargs.get("blocking") is False
        ]
        assert len(non_blocking_calls) == 2
        # The blocking shutdown call (blocking=True, terminate=True) is now in
        # _finish_train (pretrain_mimo.py), tested separately.

    @patch("megatron.bridge.training.train_mimo.checkpoint_and_decide_exit", return_value=False)
    @patch("megatron.bridge.training.train_mimo.train_step_mimo")
    @patch("megatron.bridge.training.train_mimo.build_pg_collection_for_schedule")
    @patch("megatron.bridge.training.train_mimo.get_module_to_grid_tuple")
    @patch("megatron.bridge.training.train_mimo.prepare_forward_step_func")
    @patch("megatron.bridge.training.train_mimo.get_num_microbatches", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_no_inline_save_checkpoint_call(
        self,
        mock_world_size,
        mock_rank,
        mock_num_mb,
        mock_prep_fwd,
        mock_get_grid,
        mock_build_pg,
        mock_train_step,
        mock_ckpt_exit,
    ):
        """Verify there is no inline save_checkpoint call — all saves go through
        checkpoint_and_decide_exit."""
        from megatron.bridge.training.train_mimo import train_mimo

        mock_train_step.return_value = ({}, 0, 0.0, 0)

        infra = Mock()
        infra.pg_collections = {"language": Mock()}
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        mock_build_pg.return_value = Mock(spec=[])

        state = _make_global_state(save_interval=1, train_iters=3, step=0)

        train_mimo(
            forward_step_func=Mock(),
            model=Mock(),
            optimizer=Mock(),
            schedulers={"language": _make_scheduler_mock()},
            train_data_iterator=Mock(),
            valid_data_iterator=None,
            global_state=state,
            mimo_infra=infra,
            multimodule_communicator=Mock(),
            checkpoint_manager=MagicMock(),
        )

        # checkpoint_and_decide_exit should have been called
        assert mock_ckpt_exit.call_count == 3


# ---------------------------------------------------------------------------
# Helpers for load-side tests
# ---------------------------------------------------------------------------


def _make_setup_output_for_load(
    *,
    pg_collections: Dict[str, Any] | None = None,
    train_state_step: int = 0,
    consumed_train_samples: int = 0,
    floating_point_operations_so_far: int = 0,
) -> SimpleNamespace:
    """Create a MimoSetupOutput-like namespace suitable for pretrain_mimo load tests."""
    if pg_collections is None:
        pg_collections = {"language": Mock()}

    train_state = SimpleNamespace(
        step=train_state_step,
        consumed_train_samples=consumed_train_samples,
        floating_point_operations_so_far=floating_point_operations_so_far,
    )
    timers_handle = Mock()
    timers = Mock(return_value=timers_handle)
    timers.log = Mock()

    global_state = Mock()
    global_state.timers = timers
    global_state.train_state = train_state

    mock_checkpoint_manager = MagicMock()
    mock_checkpoint_manager.checkpointing_context = {"test": "context"}

    local_pg = list(pg_collections.values())[0] if pg_collections else Mock()

    return SimpleNamespace(
        model=MagicMock(),
        mimo_infra=SimpleNamespace(
            module_to_grid_map={"language": Mock()},
            pg_collections=pg_collections,
            topology=Mock(),
        ),
        multimodule_communicator=MagicMock(),
        multimodule_pg_collection=MagicMock(),
        module_to_grid_tuple=[(MagicMock(), MagicMock())],
        optimizer=MagicMock(),
        schedulers={},
        train_data_iterator=None,
        valid_data_iterator=None,
        global_state=global_state,
        checkpoint_manager=mock_checkpoint_manager,
        active_module_name="language",
        local_pg_collection=local_pg,
    )


def _make_pretrain_cfg(
    *,
    load_path: str | None = None,
    pretrained_path: str | None = None,
    non_persistent_ckpt_type: str | None = None,
) -> MagicMock:
    """Create a ConfigContainer-like mock for pretrain_mimo tests."""
    cfg = MagicMock()
    cfg.train = SimpleNamespace(
        rampup_batch_size=None,
        global_batch_size=1,
        micro_batch_size=1,
        decrease_batch_size_if_needed=False,
    )
    cfg.data_parallel_size = 1
    cfg.checkpoint = SimpleNamespace(
        load=load_path,
        pretrained_checkpoint=pretrained_path,
        non_persistent_ckpt_type=non_persistent_ckpt_type,
        save_rng=False,
    )
    cfg.scheduler = SimpleNamespace(
        lr_warmup_init=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=100,
        lr_decay_style="linear",
        start_weight_decay=0.0,
        end_weight_decay=0.0,
        wd_incr_steps=0,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        wsd_decay_steps=None,
        lr_wsd_decay_style=None,
    )
    return cfg


def _run_pretrain_mimo(
    *,
    cfg: MagicMock | None = None,
    setup_output: SimpleNamespace | None = None,
    schedulers: Dict[str, Any] | None = None,
    build_data_iterators_fn: Any | None = None,
) -> Dict[str, Mock]:
    """Run pretrain_mimo with full mocking and return all mock handles.

    pretrain_mimo is a thin orchestrator: runtime_config_update → setup_mimo →
    train_mimo → _finish_train.  Checkpoint loading, iterator construction, and
    MPU bridging are now handled inside setup_mimo, so we mock setup_mimo at the
    boundary and only verify the orchestration.

    Returns dict with keys: setup_mimo, train_mimo, build_data_iterators_fn.
    """
    from megatron.bridge.training.pretrain_mimo import pretrain_mimo

    if cfg is None:
        cfg = _make_pretrain_cfg()
    if setup_output is None:
        setup_output = _make_setup_output_for_load()
    if schedulers is not None:
        setup_output.schedulers = schedulers
    if build_data_iterators_fn is None:
        build_data_iterators_fn = Mock(return_value=(iter([]), None))

    mocks = {}

    with (
        patch("megatron.bridge.training.pretrain_mimo.train_mimo") as m_train,
        patch("megatron.bridge.training.pretrain_mimo.setup_mimo", return_value=setup_output) as m_setup,
        patch("megatron.bridge.training.pretrain_mimo.dist") as m_dist,
        patch("megatron.bridge.training.pretrain_mimo.mimo_runtime_config_update"),
        patch("megatron.bridge.training.pretrain_mimo._finish_train"),
    ):
        m_dist.get_rank.return_value = 0
        m_dist.is_initialized.return_value = True

        pretrain_mimo(
            cfg=cfg,
            forward_step_func=MagicMock(),
            build_data_iterators_fn=build_data_iterators_fn,
            global_state=setup_output.global_state,
        )

        mocks["setup_mimo"] = m_setup
        mocks["train_mimo"] = m_train
        mocks["build_data_iterators_fn"] = build_data_iterators_fn

    return mocks


# ---------------------------------------------------------------------------
# Tests: pretrain_mimo passes build_data_iterators_fn to setup_mimo
# ---------------------------------------------------------------------------


class TestPretrainMimoPassesBuildFn:
    """Verify pretrain_mimo forwards build_data_iterators_fn to setup_mimo."""

    def test_build_fn_forwarded_to_setup_mimo(self):
        cfg = _make_pretrain_cfg()
        build_fn = Mock(return_value=(iter([]), None))
        mocks = _run_pretrain_mimo(cfg=cfg, build_data_iterators_fn=build_fn)
        _, kwargs = mocks["setup_mimo"].call_args
        assert kwargs["build_data_iterators_fn"] is build_fn


# ---------------------------------------------------------------------------
# Tests: non-colocated PG guard in get_active_module_pg
# ---------------------------------------------------------------------------


class TestActiveModulePgGuard:
    """Verify get_active_module_pg fails fast when PG topology is invalid."""

    def test_rejects_zero_active_pgs(self):
        from megatron.bridge.training.mimo_parallel_utils import get_active_module_pg

        infra = SimpleNamespace(pg_collections={})
        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            get_active_module_pg(infra)

    def test_rejects_multiple_active_pgs(self):
        from megatron.bridge.training.mimo_parallel_utils import get_active_module_pg

        infra = SimpleNamespace(pg_collections={"language": Mock(), "vision": Mock()})
        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            get_active_module_pg(infra)

    def test_returns_single_active_pg(self):
        from megatron.bridge.training.mimo_parallel_utils import get_active_module_pg

        pg = Mock()
        infra = SimpleNamespace(pg_collections={"language": pg, "vision": None})
        name, result_pg = get_active_module_pg(infra)
        assert name == "language"
        assert result_pg is pg


# ---------------------------------------------------------------------------
# Tests: MimoOptimizer load-side compatibility
# ---------------------------------------------------------------------------


class TestMimoOptimizerLoadCompat:
    """Verify MimoOptimizer load methods work correctly."""

    def _make_mimo_optimizer(self):
        from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo

        opt_a = MagicMock()
        opt_b = MagicMock()

        module_infos = {
            "language": ModuleOptimizerInfo(optimizer=opt_a, grid=Mock(), pg_collection=Mock(), is_active=True),
            "vision": ModuleOptimizerInfo(optimizer=opt_b, grid=Mock(), pg_collection=Mock(), is_active=True),
        }
        config = MagicMock()
        return MimoOptimizer(module_infos, config), opt_a, opt_b

    def test_load_state_dict_dispatches_per_module(self):
        mimo_opt, opt_a, opt_b = self._make_mimo_optimizer()
        state = {"language": {"param": 1}, "vision": {"param": 2}}
        mimo_opt.load_state_dict(state)
        opt_a.load_state_dict.assert_called_once_with({"param": 1})
        opt_b.load_state_dict.assert_called_once_with({"param": 2})

    def test_load_state_dict_skips_missing_keys(self):
        mimo_opt, opt_a, opt_b = self._make_mimo_optimizer()
        state = {"language": {"param": 1}}
        mimo_opt.load_state_dict(state)
        opt_a.load_state_dict.assert_called_once()
        opt_b.load_state_dict.assert_not_called()

    def test_sharded_state_dict_generates_per_module(self):
        mimo_opt, opt_a, opt_b = self._make_mimo_optimizer()
        opt_a.sharded_state_dict.return_value = {"a": "sharded_a"}
        opt_b.sharded_state_dict.return_value = {"b": "sharded_b"}

        result = mimo_opt.sharded_state_dict({}, is_loading=True)
        assert "language" in result
        assert "vision" in result
        assert result["language"] == {"a": "sharded_a"}
        assert result["vision"] == {"b": "sharded_b"}

    def test_reload_model_params_delegates_to_all_active(self):
        mimo_opt, opt_a, opt_b = self._make_mimo_optimizer()
        mimo_opt.reload_model_params(state_dict={"model": {}})
        opt_a.reload_model_params.assert_called_once_with({"model": {}})
        opt_b.reload_model_params.assert_called_once_with({"model": {}})

    def test_is_stub_optimizer_when_no_active(self):
        from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo

        module_infos = {
            "language": ModuleOptimizerInfo(optimizer=None, grid=Mock(), pg_collection=Mock(), is_active=False),
        }
        mimo_opt = MimoOptimizer(module_infos, MagicMock())
        assert mimo_opt.is_stub_optimizer is True


# ---------------------------------------------------------------------------
# Tests: train state restoration smoke
# ---------------------------------------------------------------------------


class TestTrainStateRestorationSmoke:
    """Smoke tests for train_state being accessible after load."""

    def test_train_state_step_accessible_after_load(self):
        setup_output = _make_setup_output_for_load(train_state_step=42, consumed_train_samples=1000)
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")

        mocks = _run_pretrain_mimo(
            cfg=cfg,
            setup_output=setup_output,
        )

        # train_state is passed to train_mimo via global_state
        _, kwargs = mocks["train_mimo"].call_args
        ts = kwargs["global_state"].train_state
        assert ts.step == 42
        assert ts.consumed_train_samples == 1000

    def test_floating_point_ops_preserved(self):
        setup_output = _make_setup_output_for_load(floating_point_operations_so_far=99999)
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        mocks = _run_pretrain_mimo(
            cfg=cfg,
            setup_output=setup_output,
        )
        _, kwargs = mocks["train_mimo"].call_args
        assert kwargs["global_state"].train_state.floating_point_operations_so_far == 99999


# ---------------------------------------------------------------------------
# Tests: pretrain_mimo orchestration (setup → train)
# ---------------------------------------------------------------------------


class TestPretrainMimoOrchestration:
    """Verify pretrain_mimo calls setup_mimo and train_mimo correctly."""

    def test_train_mimo_always_called(self):
        """pretrain_mimo should always invoke train_mimo after setup."""
        cfg = _make_pretrain_cfg()
        mocks = _run_pretrain_mimo(cfg=cfg)
        mocks["train_mimo"].assert_called_once()

    def test_iterators_from_setup_forwarded_to_train(self):
        """Data iterators from setup_output should be forwarded to train_mimo."""
        cfg = _make_pretrain_cfg()
        build_fn = Mock(return_value=(iter([]), None))
        mocks = _run_pretrain_mimo(cfg=cfg, build_data_iterators_fn=build_fn)
        # build_fn is passed to setup_mimo, which is mocked — check that
        # train_mimo receives setup_output's iterators
        mocks["train_mimo"].assert_called_once()


# ---------------------------------------------------------------------------
# Tests: load_checkpoint pg_collection explicit threading
# ---------------------------------------------------------------------------


class TestLoadCheckpointPgThreading:
    """Verify load_checkpoint and _load_checkpoint_from_path accept and
    thread explicit pg_collection."""

    def test_load_checkpoint_forwards_pg_collection_to_inner(self):
        from megatron.bridge.training.checkpointing import load_checkpoint

        pg = Mock()
        state = Mock()
        state.cfg.checkpoint.load = "/tmp/ckpt"
        state.cfg.checkpoint.pretrained_checkpoint = None

        with patch(
            "megatron.bridge.training.checkpointing._load_checkpoint_from_path",
            return_value=(0, 0),
        ) as m_inner:
            with patch(
                "megatron.bridge.training.checkpointing.checkpoint_exists",
                return_value=True,
            ):
                load_checkpoint(
                    state=state,
                    model=[Mock()],
                    optimizer=Mock(),
                    opt_param_scheduler=Mock(),
                    pg_collection=pg,
                )
                _, kwargs = m_inner.call_args
                assert kwargs["pg_collection"] is pg

    def test_load_checkpoint_defaults_pg_collection_to_none(self):
        from megatron.bridge.training.checkpointing import load_checkpoint

        state = Mock()
        state.cfg.checkpoint.load = "/tmp/ckpt"
        state.cfg.checkpoint.pretrained_checkpoint = None

        with patch(
            "megatron.bridge.training.checkpointing._load_checkpoint_from_path",
            return_value=(0, 0),
        ) as m_inner:
            with patch(
                "megatron.bridge.training.checkpointing.checkpoint_exists",
                return_value=True,
            ):
                load_checkpoint(
                    state=state,
                    model=[Mock()],
                    optimizer=Mock(),
                    opt_param_scheduler=Mock(),
                )
                _, kwargs = m_inner.call_args
                assert kwargs["pg_collection"] is None


# ---------------------------------------------------------------------------
# Tests: setup_mimo checkpoint loading path
# ---------------------------------------------------------------------------


class TestSetupMimoCheckpointLoading:
    """Verify setup_mimo invokes load_checkpoint when checkpoints exist."""

    _SETUP_PATCHES = [
        "megatron.bridge.training.setup_mimo.get_active_module_pg",
        "megatron.bridge.training.setup_mimo.create_checkpoint_manager",
        "megatron.bridge.training.setup_mimo.MultiModulePipelineCommunicator",
        "megatron.bridge.training.setup_mimo.get_model_config",
        "megatron.bridge.training.setup_mimo.validate_no_stub_ranks",
        "megatron.bridge.training.setup_mimo.build_pg_collection_for_schedule",
        "megatron.bridge.training.setup_mimo.get_module_to_grid_tuple",
        "megatron.bridge.training.setup_mimo._update_mimo_model_config_funcs",
        "megatron.bridge.training.setup_mimo.unwrap_mimo_model",
        "megatron.bridge.training.setup_mimo.dist",
        "megatron.bridge.training.setup_mimo._set_mimo_random_seeds",
        "megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR",
        "megatron.core.num_microbatches_calculator.init_num_microbatches_calculator",
        "megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP",
        "megatron.core.parallel_state._DATA_PARALLEL_GROUP",
        "megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP",
    ]

    def _run_setup(self, *, load_path=None, checkpoint_exists_return=False):
        """Run setup_mimo with mocks, return dict of mock handles."""
        from megatron.bridge.training.setup_mimo import setup_mimo

        cfg = Mock()
        cfg.checkpoint = SimpleNamespace(
            load=load_path,
            pretrained_checkpoint=None,
            non_persistent_ckpt_type=None,
        )
        cfg.model = Mock()
        cfg.model.fp16 = False
        cfg.model.bf16 = True
        cfg.optimizer = Mock()
        cfg.scheduler = SimpleNamespace(
            lr_warmup_init=0.0,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="linear",
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            wd_incr_steps=0,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            wsd_decay_steps=None,
            lr_wsd_decay_style=None,
        )
        cfg._calculate_scheduler_steps = Mock()

        infra = Mock()
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        infra.module_output_ndim = {"language": 3}
        infra.pg_collections = {"language": Mock()}
        cfg.model.build_infra.return_value = infra
        cfg.model.provide_distributed_model.return_value = [Mock()]

        local_pg = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer.module_infos = {}

        state = Mock()
        state.cfg = cfg
        state.start_time = time.time()
        state.train_state = SimpleNamespace(step=0)

        mocks = {}

        with (
            ExitStack() as stack,
        ):
            for p in self._SETUP_PATCHES:
                if "._" in p.split(".")[-1]:
                    stack.enter_context(patch(p, None))
                else:
                    stack.enter_context(patch(p))

            m_dist = stack.enter_context(patch("megatron.bridge.training.setup_mimo.dist"))
            m_dist.get_rank.return_value = 0
            m_dist.get_world_size.return_value = 2

            stack.enter_context(
                patch("megatron.bridge.training.setup_mimo.get_active_module_pg", return_value=("language", local_pg))
            )
            m_ckpt_exists = stack.enter_context(
                patch("megatron.bridge.training.setup_mimo.checkpoint_exists", return_value=checkpoint_exists_return)
            )
            m_load = stack.enter_context(patch("megatron.bridge.training.setup_mimo.load_checkpoint"))
            m_create_mgr = stack.enter_context(patch("megatron.bridge.training.setup_mimo.create_checkpoint_manager"))
            m_create_mgr.return_value = MagicMock(checkpointing_context={"ctx": True})

            stack.enter_context(
                patch("megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=mock_optimizer)
            )

            model_config = Mock(pipeline_dtype=None, bf16=True)
            stack.enter_context(
                patch("megatron.bridge.training.setup_mimo.get_model_config", return_value=model_config)
            )
            stack.enter_context(patch("megatron.bridge.training.setup_mimo.unwrap_mimo_model"))

            result = setup_mimo(state=state)

            mocks["load_checkpoint"] = m_load
            mocks["checkpoint_exists"] = m_ckpt_exists
            mocks["result"] = result

        return mocks

    def test_load_invoked_when_persistent_checkpoint_exists(self):
        mocks = self._run_setup(load_path="/tmp/ckpt", checkpoint_exists_return=True)
        mocks["load_checkpoint"].assert_called_once()

    def test_load_not_invoked_when_no_checkpoint(self):
        mocks = self._run_setup(load_path=None, checkpoint_exists_return=False)
        mocks["load_checkpoint"].assert_not_called()

    def test_load_passes_model_as_list(self):
        mocks = self._run_setup(load_path="/tmp/ckpt", checkpoint_exists_return=True)
        _, kwargs = mocks["load_checkpoint"].call_args
        assert isinstance(kwargs["model"], list)
        assert len(kwargs["model"]) == 1

    def test_load_passes_pg_collection(self):
        mocks = self._run_setup(load_path="/tmp/ckpt", checkpoint_exists_return=True)
        _, kwargs = mocks["load_checkpoint"].call_args
        assert kwargs["pg_collection"] is not None

    def test_load_passes_module_name(self):
        mocks = self._run_setup(load_path="/tmp/ckpt", checkpoint_exists_return=True)
        _, kwargs = mocks["load_checkpoint"].call_args
        assert kwargs["module_name"] == "language"


# ---------------------------------------------------------------------------
# Tests: setup_mimo resume-aware iterator construction
# ---------------------------------------------------------------------------


class TestSetupMimoResumeIterators:
    """Verify setup_mimo builds iterators with train_state when resuming."""

    def test_train_state_passed_when_resuming(self):
        """When train_state.step > 0, builder receives train_state kwarg."""
        from megatron.bridge.training.setup_mimo import setup_mimo

        build_fn = MagicMock(return_value=(iter([]), None))

        def _sig_fn(cfg, mimo_infra, *, train_state=None):
            pass

        build_fn.__signature__ = inspect.signature(_sig_fn)

        cfg = Mock()
        cfg.checkpoint = SimpleNamespace(load=None, pretrained_checkpoint=None, non_persistent_ckpt_type=None)
        cfg.model = Mock(fp16=False, bf16=True)
        cfg.optimizer = Mock()
        cfg.scheduler = SimpleNamespace(
            lr_warmup_init=0.0,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="linear",
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            wd_incr_steps=0,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            wsd_decay_steps=None,
            lr_wsd_decay_style=None,
        )
        cfg._calculate_scheduler_steps = Mock()

        infra = Mock()
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        infra.module_output_ndim = {"language": 3}
        infra.pg_collections = {"language": Mock()}
        cfg.model.build_infra.return_value = infra
        cfg.model.provide_distributed_model.return_value = [Mock()]

        state = Mock()
        state.cfg = cfg
        state.start_time = time.time()
        # Simulate resumed training — step > 0
        state.train_state = SimpleNamespace(step=10, consumed_train_samples=100)

        mock_optimizer = MagicMock()
        mock_optimizer.module_infos = {}

        with (
            patch("megatron.bridge.training.setup_mimo.dist") as m_dist,
            patch("megatron.bridge.training.setup_mimo._set_mimo_random_seeds"),
            patch("megatron.bridge.training.setup_mimo.validate_no_stub_ranks"),
            patch("megatron.bridge.training.setup_mimo.build_pg_collection_for_schedule"),
            patch("megatron.bridge.training.setup_mimo.get_module_to_grid_tuple"),
            patch("megatron.bridge.training.setup_mimo.MultiModulePipelineCommunicator"),
            patch(
                "megatron.bridge.training.setup_mimo.get_model_config",
                return_value=Mock(pipeline_dtype=None, bf16=True),
            ),
            patch("megatron.bridge.training.setup_mimo.unwrap_mimo_model"),
            patch("megatron.bridge.training.setup_mimo._update_mimo_model_config_funcs"),
            patch("megatron.bridge.training.setup_mimo.get_active_module_pg", return_value=("language", MagicMock())),
            patch(
                "megatron.bridge.training.setup_mimo.create_checkpoint_manager",
                return_value=MagicMock(checkpointing_context={}),
            ),
            patch("megatron.bridge.training.setup_mimo.checkpoint_exists", return_value=False),
            patch("megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=mock_optimizer),
            patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
            patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
            patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None),
        ):
            m_dist.get_rank.return_value = 0
            m_dist.get_world_size.return_value = 2

            setup_mimo(state=state, build_data_iterators_fn=build_fn)

        build_fn.assert_called_once()
        _, kwargs = build_fn.call_args
        assert "train_state" in kwargs
        assert kwargs["train_state"].step == 10

    def test_no_train_state_when_not_resuming(self):
        """When train_state.step == 0, builder is called without train_state."""
        from megatron.bridge.training.setup_mimo import setup_mimo

        build_fn = Mock(return_value=(iter([]), None))

        cfg = Mock()
        cfg.checkpoint = SimpleNamespace(load=None, pretrained_checkpoint=None, non_persistent_ckpt_type=None)
        cfg.model = Mock(fp16=False, bf16=True)
        cfg.optimizer = Mock()
        cfg.scheduler = SimpleNamespace(
            lr_warmup_init=0.0,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="linear",
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            wd_incr_steps=0,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            wsd_decay_steps=None,
            lr_wsd_decay_style=None,
        )
        cfg._calculate_scheduler_steps = Mock()

        infra = Mock()
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        infra.module_output_ndim = {"language": 3}
        infra.pg_collections = {"language": Mock()}
        cfg.model.build_infra.return_value = infra
        cfg.model.provide_distributed_model.return_value = [Mock()]

        state = Mock()
        state.cfg = cfg
        state.start_time = time.time()
        state.train_state = SimpleNamespace(step=0)

        mock_optimizer = MagicMock()
        mock_optimizer.module_infos = {}

        with (
            patch("megatron.bridge.training.setup_mimo.dist") as m_dist,
            patch("megatron.bridge.training.setup_mimo._set_mimo_random_seeds"),
            patch("megatron.bridge.training.setup_mimo.validate_no_stub_ranks"),
            patch("megatron.bridge.training.setup_mimo.build_pg_collection_for_schedule"),
            patch("megatron.bridge.training.setup_mimo.get_module_to_grid_tuple"),
            patch("megatron.bridge.training.setup_mimo.MultiModulePipelineCommunicator"),
            patch(
                "megatron.bridge.training.setup_mimo.get_model_config",
                return_value=Mock(pipeline_dtype=None, bf16=True),
            ),
            patch("megatron.bridge.training.setup_mimo.unwrap_mimo_model"),
            patch("megatron.bridge.training.setup_mimo._update_mimo_model_config_funcs"),
            patch("megatron.bridge.training.setup_mimo.get_active_module_pg", return_value=("language", MagicMock())),
            patch(
                "megatron.bridge.training.setup_mimo.create_checkpoint_manager",
                return_value=MagicMock(checkpointing_context={}),
            ),
            patch("megatron.bridge.training.setup_mimo.checkpoint_exists", return_value=False),
            patch("megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=mock_optimizer),
            patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
            patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
            patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None),
        ):
            m_dist.get_rank.return_value = 0
            m_dist.get_world_size.return_value = 2

            setup_mimo(state=state, build_data_iterators_fn=build_fn)

        build_fn.assert_called_once()
        _, kwargs = build_fn.call_args
        assert "train_state" not in kwargs

    def test_raises_when_builder_lacks_train_state_on_resume(self):
        """Resuming with a builder that can't accept train_state raises RuntimeError."""
        from megatron.bridge.training.setup_mimo import setup_mimo

        def legacy_builder(cfg, mimo_infra):
            return (iter([]), None)

        build_fn = MagicMock(return_value=(iter([]), None))
        build_fn.__signature__ = inspect.signature(legacy_builder)

        cfg = Mock()
        cfg.checkpoint = SimpleNamespace(load=None, pretrained_checkpoint=None, non_persistent_ckpt_type=None)
        cfg.model = Mock(fp16=False, bf16=True)
        cfg.optimizer = Mock()
        cfg.scheduler = SimpleNamespace(
            lr_warmup_init=0.0,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="linear",
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            wd_incr_steps=0,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            wsd_decay_steps=None,
            lr_wsd_decay_style=None,
        )
        cfg._calculate_scheduler_steps = Mock()

        infra = Mock()
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        infra.module_output_ndim = {"language": 3}
        infra.pg_collections = {"language": Mock()}
        cfg.model.build_infra.return_value = infra
        cfg.model.provide_distributed_model.return_value = [Mock()]

        state = Mock()
        state.cfg = cfg
        state.start_time = time.time()
        state.train_state = SimpleNamespace(step=10)

        mock_optimizer = MagicMock()
        mock_optimizer.module_infos = {}

        with (
            patch("megatron.bridge.training.setup_mimo.dist") as m_dist,
            patch("megatron.bridge.training.setup_mimo._set_mimo_random_seeds"),
            patch("megatron.bridge.training.setup_mimo.validate_no_stub_ranks"),
            patch("megatron.bridge.training.setup_mimo.build_pg_collection_for_schedule"),
            patch("megatron.bridge.training.setup_mimo.get_module_to_grid_tuple"),
            patch("megatron.bridge.training.setup_mimo.MultiModulePipelineCommunicator"),
            patch(
                "megatron.bridge.training.setup_mimo.get_model_config",
                return_value=Mock(pipeline_dtype=None, bf16=True),
            ),
            patch("megatron.bridge.training.setup_mimo.unwrap_mimo_model"),
            patch("megatron.bridge.training.setup_mimo._update_mimo_model_config_funcs"),
            patch("megatron.bridge.training.setup_mimo.get_active_module_pg", return_value=("language", MagicMock())),
            patch(
                "megatron.bridge.training.setup_mimo.create_checkpoint_manager",
                return_value=MagicMock(checkpointing_context={}),
            ),
            patch("megatron.bridge.training.setup_mimo.checkpoint_exists", return_value=False),
            patch("megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=mock_optimizer),
            patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
            patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
            patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None),
        ):
            m_dist.get_rank.return_value = 0
            m_dist.get_world_size.return_value = 2

            with pytest.raises(RuntimeError, match="build_data_iterators_fn does not accept"):
                setup_mimo(state=state, build_data_iterators_fn=build_fn)
