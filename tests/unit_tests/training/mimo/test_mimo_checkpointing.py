# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MiMo checkpoint saving and loading wiring.

Tests validate that MiMo training correctly uses shared checkpoint helpers
(save_checkpoint_and_time, checkpoint_and_decide_exit, load_checkpoint) with
the right arguments, without actually saving/loading checkpoints.
"""

from __future__ import annotations

import inspect
import time
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
    """Create a mock MimoModelInfra with the given number of active PG collections."""
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

    @patch("megatron.bridge.training.setup_mimo.DefaultCheckpointManager")
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
        mock_default_ckpt_mgr,
    ):
        from megatron.bridge.training.setup_mimo import setup_mimo

        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2

        mock_mgr_instance = MagicMock()
        mock_mgr_instance.checkpointing_context = {"test": "context"}
        mock_default_ckpt_mgr.return_value = mock_mgr_instance

        model_config = Mock()
        model_config.pipeline_dtype = None
        model_config.bf16 = True
        mock_get_config.return_value = model_config

        unwrapped = MagicMock()
        unwrapped.mimo_config.module_to_grid_map = {"language": Mock()}
        mock_unwrap.return_value = unwrapped

        global_state = Mock()
        global_state.start_time = time.time()
        global_state.cfg = None

        cfg = Mock()
        cfg.checkpoint = Mock()
        cfg.train = Mock()
        cfg.train.grad_reduce_in_fp32 = False
        cfg.train.overlap_grad_reduce = True
        cfg.train.use_distributed_optimizer = False
        cfg.train.check_for_nan_in_grad = False
        cfg.model = Mock()
        cfg.model.fp16 = False
        cfg.model.bf16 = True

        infra = Mock()
        infra.module_to_grid_map = {"language": Mock()}
        infra.topology = Mock()
        infra.module_output_ndim = {"language": 3}
        infra.pg_collections = {"language": Mock()}
        cfg.model.build_infra.return_value = infra
        cfg.model.provide_distributed_model.return_value = [Mock()]

        with (
            patch("megatron.bridge.training.setup_mimo._set_mimo_random_seeds"),
            patch("megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR", None),
            patch("megatron.core.num_microbatches_calculator.init_num_microbatches_calculator"),
        ):
            result = setup_mimo(cfg=cfg, build_optimizer=False, global_state=global_state)

        mock_default_ckpt_mgr.assert_called_once_with(cfg.checkpoint)
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
            patch("megatron.bridge.training.pretrain_mimo.checkpoint_exists", return_value=False),
            patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None),
            patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None),
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
    checkpoint_exists_return: bool = False,
    build_data_iterators_fn: Any | None = None,
) -> Dict[str, Mock]:
    """Run pretrain_mimo with full mocking and return all mock handles.

    Returns dict with keys: setup_mimo, load_checkpoint, checkpoint_exists,
    train_mimo, build_data_iterators_fn.
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
        patch("megatron.bridge.training.pretrain_mimo.load_checkpoint") as m_load,
        patch(
            "megatron.bridge.training.pretrain_mimo.checkpoint_exists",
            return_value=checkpoint_exists_return,
        ) as m_ckpt_exists,
        patch("megatron.bridge.training.pretrain_mimo.dist") as m_dist,
        patch("megatron.bridge.training.pretrain_mimo.mimo_runtime_config_update"),
        patch("megatron.bridge.training.pretrain_mimo._finish_train"),
        patch("megatron.core.parallel_state._TENSOR_MODEL_PARALLEL_GROUP", None),
        patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP", None),
        patch("megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP", None),
    ):
        m_dist.get_rank.return_value = 0
        m_dist.get_world_size.return_value = 2

        pretrain_mimo(
            cfg=cfg,
            forward_step_func=MagicMock(),
            build_data_iterators_fn=build_data_iterators_fn,
            global_state=setup_output.global_state,
        )

        mocks["setup_mimo"] = m_setup
        mocks["load_checkpoint"] = m_load
        mocks["checkpoint_exists"] = m_ckpt_exists
        mocks["train_mimo"] = m_train
        mocks["build_data_iterators_fn"] = build_data_iterators_fn

    return mocks


# ---------------------------------------------------------------------------
# Tests: load_checkpoint invocation from pretrain_mimo
# ---------------------------------------------------------------------------


class TestPretrainMimoLoadCheckpoint:
    """Verify pretrain_mimo invokes load_checkpoint with correct arguments."""

    def test_load_invoked_when_persistent_checkpoint_exists(self):
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        mocks = _run_pretrain_mimo(cfg=cfg, checkpoint_exists_return=True)
        mocks["load_checkpoint"].assert_called_once()

    def test_load_invoked_when_pretrained_checkpoint_exists(self):
        cfg = _make_pretrain_cfg(pretrained_path="/tmp/pretrained")
        mocks = _run_pretrain_mimo(cfg=cfg, checkpoint_exists_return=True)
        mocks["load_checkpoint"].assert_called_once()

    def test_load_invoked_for_non_persistent_intent_without_persistent_path(self):
        """Non-persistent resume intent should trigger load even without cfg.checkpoint.load."""
        cfg = _make_pretrain_cfg(non_persistent_ckpt_type="local")
        mocks = _run_pretrain_mimo(cfg=cfg, checkpoint_exists_return=False)
        mocks["load_checkpoint"].assert_called_once()

    def test_load_not_invoked_when_no_checkpoint_intent(self):
        cfg = _make_pretrain_cfg()
        mocks = _run_pretrain_mimo(cfg=cfg, checkpoint_exists_return=False)
        mocks["load_checkpoint"].assert_not_called()

    def test_load_forwards_list_wrapped_model(self):
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        setup_output = _make_setup_output_for_load()
        mocks = _run_pretrain_mimo(
            cfg=cfg,
            setup_output=setup_output,
            checkpoint_exists_return=True,
        )
        _, kwargs = mocks["load_checkpoint"].call_args
        assert isinstance(kwargs["model"], list)
        assert len(kwargs["model"]) == 1
        assert kwargs["model"][0] is setup_output.model

    def test_load_forwards_explicit_pg_collection(self):
        pg = Mock()
        setup_output = _make_setup_output_for_load(pg_collections={"language": pg})
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        mocks = _run_pretrain_mimo(
            cfg=cfg,
            setup_output=setup_output,
            checkpoint_exists_return=True,
        )
        _, kwargs = mocks["load_checkpoint"].call_args
        assert kwargs["pg_collection"] is pg

    def test_load_forwards_checkpointing_context(self):
        setup_output = _make_setup_output_for_load()
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        mocks = _run_pretrain_mimo(
            cfg=cfg,
            setup_output=setup_output,
            checkpoint_exists_return=True,
        )
        _, kwargs = mocks["load_checkpoint"].call_args
        assert kwargs["checkpointing_context"] is setup_output.checkpoint_manager.checkpointing_context

    def test_load_forwards_first_scheduler(self):
        sched_a = _make_scheduler_mock()
        sched_b = _make_scheduler_mock()
        schedulers = {"language": sched_a, "vision": sched_b}
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        mocks = _run_pretrain_mimo(
            cfg=cfg,
            schedulers=schedulers,
            checkpoint_exists_return=True,
        )
        _, kwargs = mocks["load_checkpoint"].call_args
        assert kwargs["opt_param_scheduler"] is sched_a


# ---------------------------------------------------------------------------
# Tests: non-colocated PG guard in pretrain_mimo load path
# ---------------------------------------------------------------------------


class TestPretrainMimoLoadPgGuard:
    """Verify pretrain_mimo fails fast when PG topology is invalid."""

    def test_rejects_zero_active_pgs_in_pretrain(self):
        setup_output = _make_setup_output_for_load(pg_collections={})
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            _run_pretrain_mimo(
                cfg=cfg,
                setup_output=setup_output,
                checkpoint_exists_return=True,
            )

    def test_rejects_multiple_active_pgs_in_pretrain(self):
        setup_output = _make_setup_output_for_load(
            pg_collections={"language": Mock(), "vision": Mock()},
        )
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        with pytest.raises(AssertionError, match="exactly one active ProcessGroupCollection"):
            _run_pretrain_mimo(
                cfg=cfg,
                setup_output=setup_output,
                checkpoint_exists_return=True,
            )


# ---------------------------------------------------------------------------
# Tests: scheduler v1 fanout behavior
# ---------------------------------------------------------------------------


class TestSchedulerV1Fanout:
    """Verify scheduler state is loaded into first_scheduler and fanned out."""

    def test_scheduler_fanout_after_load(self):
        """After load, all schedulers should have the state of first_scheduler."""
        sched_a = MagicMock()
        sched_a.optimizer.param_groups = [{"lr": 1e-4}]
        sched_a.state_dict.return_value = {"step": 50, "lr": 0.001}

        sched_b = MagicMock()
        sched_b.optimizer.param_groups = [{"lr": 1e-4}]

        schedulers = {"language": sched_a, "vision": sched_b}
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")

        # Simulate load succeeding and setting step > 0 via side_effect.
        # load_checkpoint modifies global_state.train_state in-place, but
        # in mock context it doesn't. We need step to remain 0 so iterator
        # builder doesn't require train_state kwarg.
        _run_pretrain_mimo(cfg=cfg, schedulers=schedulers, checkpoint_exists_return=True)

        # sched_b should have received the fanout
        sched_b.load_state_dict.assert_called_once_with({"step": 50, "lr": 0.001})
        # sched_a should NOT have load_state_dict called by fanout (it's the source)
        sched_a.load_state_dict.assert_not_called()

    def test_no_fanout_when_single_scheduler(self):
        sched = MagicMock()
        sched.optimizer.param_groups = [{"lr": 1e-4}]
        sched.state_dict.return_value = {"step": 50}

        schedulers = {"language": sched}
        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        _run_pretrain_mimo(cfg=cfg, schedulers=schedulers, checkpoint_exists_return=True)

        # No fanout needed with single scheduler
        sched.load_state_dict.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: iterator resume semantics
# ---------------------------------------------------------------------------


class TestIteratorResumeSemanticsLoad:
    """Verify iterators are built after load and receive train_state when resuming."""

    def test_iterators_built_after_setup_not_during(self):
        """setup_mimo should be called with build_data_iterators_fn=None."""
        cfg = _make_pretrain_cfg()
        mocks = _run_pretrain_mimo(cfg=cfg)
        _, kwargs = mocks["setup_mimo"].call_args
        assert kwargs["build_data_iterators_fn"] is None

    def test_iterator_builder_called_without_train_state_when_not_resuming(self):
        cfg = _make_pretrain_cfg()
        build_fn = Mock(return_value=(iter([]), None))
        _run_pretrain_mimo(cfg=cfg, build_data_iterators_fn=build_fn)
        build_fn.assert_called_once()
        args, kwargs = build_fn.call_args
        assert "train_state" not in kwargs

    def test_iterator_builder_receives_train_state_mock(self):
        """When resuming (step > 0), builder receives train_state kwarg."""
        build_fn = MagicMock(return_value=(iter([]), None))
        # Give mock the train_state parameter so inspect.signature finds it

        def _sig_fn(cfg, mimo_infra, *, train_state=None):
            pass

        build_fn.__signature__ = inspect.signature(_sig_fn)

        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        setup_output = _make_setup_output_for_load(train_state_step=10, consumed_train_samples=500)

        _run_pretrain_mimo(
            cfg=cfg,
            setup_output=setup_output,
            checkpoint_exists_return=True,
            build_data_iterators_fn=build_fn,
        )

        build_fn.assert_called_once()
        _, kwargs = build_fn.call_args
        assert "train_state" in kwargs
        assert kwargs["train_state"].step == 10
        assert kwargs["train_state"].consumed_train_samples == 500

    def test_iterator_builder_fails_fast_if_no_train_state_param_on_resume(self):
        """Resuming with a builder that lacks train_state param raises RuntimeError."""

        def legacy_builder(cfg, mimo_infra):
            return (iter([]), None)

        build_fn = MagicMock(return_value=(iter([]), None))
        build_fn.__signature__ = inspect.signature(legacy_builder)

        cfg = _make_pretrain_cfg(load_path="/tmp/ckpt")
        setup_output = _make_setup_output_for_load(train_state_step=10)

        with pytest.raises(RuntimeError, match="build_data_iterators_fn does not accept"):
            _run_pretrain_mimo(
                cfg=cfg,
                setup_output=setup_output,
                checkpoint_exists_return=True,
                build_data_iterators_fn=build_fn,
            )


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

        def builder(cfg, mimo_infra, *, train_state=None):
            return (iter([]), None)

        build_fn = MagicMock(return_value=(iter([]), None))
        build_fn.__signature__ = inspect.signature(builder)

        mocks = _run_pretrain_mimo(
            cfg=cfg,
            setup_output=setup_output,
            checkpoint_exists_return=True,
            build_data_iterators_fn=build_fn,
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
            checkpoint_exists_return=True,
        )
        _, kwargs = mocks["train_mimo"].call_args
        assert kwargs["global_state"].train_state.floating_point_operations_so_far == 99999


# ---------------------------------------------------------------------------
# Tests: local checkpoint resume plumbing
# ---------------------------------------------------------------------------


class TestLocalCheckpointResumePlumbing:
    """Verify non-persistent local checkpoint intent triggers load."""

    def test_local_non_persistent_triggers_load(self):
        cfg = _make_pretrain_cfg(non_persistent_ckpt_type="local")
        mocks = _run_pretrain_mimo(cfg=cfg, checkpoint_exists_return=False)
        mocks["load_checkpoint"].assert_called_once()

    def test_global_non_persistent_triggers_load(self):
        cfg = _make_pretrain_cfg(non_persistent_ckpt_type="global")
        mocks = _run_pretrain_mimo(cfg=cfg, checkpoint_exists_return=False)
        mocks["load_checkpoint"].assert_called_once()


# ---------------------------------------------------------------------------
# Tests: no-checkpoint graceful fallback
# ---------------------------------------------------------------------------


class TestNoCheckpointGracefulFallback:
    """Verify load is not attempted and training starts from random init."""

    def test_no_load_no_crash(self):
        """When no checkpoint intent exists, load is skipped and training starts cleanly."""
        cfg = _make_pretrain_cfg()
        mocks = _run_pretrain_mimo(cfg=cfg, checkpoint_exists_return=False)
        mocks["load_checkpoint"].assert_not_called()
        mocks["train_mimo"].assert_called_once()

    def test_iterators_still_built_without_checkpoint(self):
        cfg = _make_pretrain_cfg()
        build_fn = Mock(return_value=(iter([]), None))
        _run_pretrain_mimo(cfg=cfg, build_data_iterators_fn=build_fn)
        build_fn.assert_called_once()


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
