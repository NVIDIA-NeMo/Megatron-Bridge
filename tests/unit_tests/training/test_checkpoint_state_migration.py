# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native checkpoint migrations run before restored metadata affects runtime state."""

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

import megatron.bridge.training.checkpointing as checkpointing
from megatron.bridge.training.state import TrainState
from tests.unit_tests.training.test_checkpointing import load_checkpoint_fixtures  # noqa: F401


pytestmark = pytest.mark.unit


@pytest.fixture(params=["torch_dist", "fsdp_dtensor", "local"])
def migration_resume(request, load_checkpoint_fixtures):
    fixtures = load_checkpoint_fixtures
    state, cfg = fixtures["mock_state"], fixtures["mock_cfg"]
    cfg.peft = None
    cfg.checkpoint.ckpt_format = "torch_dist" if request.param == "local" else request.param
    cfg.checkpoint.load_rng = False
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = False
    cfg.checkpoint.fully_parallel_save = False
    cfg.ddp.fp8_param_gather = cfg.ddp.fp4_param_gather = False
    cfg.scheduler.override_opt_param_scheduler = True
    cfg.optimizer.lr, cfg.optimizer.min_lr = 0.001, 0.0
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    scheduler = OptimizerParamScheduler(
        optimizer,
        0,
        0.001,
        0,
        400,
        1200,
        "linear",
        0,
        0,
        1200,
        "constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )
    saved_scheduler = scheduler.state_dict()
    saved_scheduler["num_steps"] = 600
    saved_train_state = TrainState(step=6, consumed_train_samples=0)
    state.train_state = TrainState()
    payload = {
        "model": {},
        "optimizer": optimizer.state_dict(),
        "opt_param_scheduler": saved_scheduler,
        "train_state_metadata": saved_train_state.state_dict(),
        "checkpoint_version": 3.1,
    }
    run_config = {
        "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
        "checkpoint": {"save_optim": True, "save_rng": False, "fully_parallel_save": False},
    }
    checkpoint_type = (
        checkpointing.CheckpointType.LOCAL if request.param == "local" else checkpointing.CheckpointType.GLOBAL
    )
    with ExitStack() as stack:
        replacements = {
            "checkpoint_exists": True,
            "is_hf_checkpoint_dir": False,
            "file_exists": True,
            "read_run_config": run_config,
            "read_train_state": saved_train_state,
            "update_num_microbatches": None,
            "_get_model_glu_interleave_sizes": (None, None),
            "_load_model_state_dict": None,
            "generate_state_dict": {"model": {}},
            "unwrap_model": fixtures["mock_model"],
            "get_pg_collection": Mock(),
            "hybrid_optimizer_state_loading": nullcontext(),
            "memory_efficient_precision_aware_optimizer_state_checkpointing": nullcontext(),
            "sync_hybrid_device_optimizer_fp32_master_copies": None,
        }
        mocks = {
            name: stack.enter_context(patch.object(checkpointing, name, return_value=value))
            for name, value in replacements.items()
        }
        mocks["base_load"] = stack.enter_context(
            patch.object(
                checkpointing,
                "_load_base_checkpoint",
                return_value=(payload, "/checkpoints/iter_0000006", False, checkpoint_type),
            )
        )
        stack.enter_context(patch.object(checkpointing.dist_checkpointing, "load_content_metadata", return_value={}))
        reader = stack.enter_context(patch.object(checkpointing, "_get_filesystem_reader"))
        reader.return_value.read_metadata.return_value.state_dict_metadata = {}
        stack.enter_context(patch.object(torch.distributed, "is_initialized", return_value=False))
        for logger in (checkpointing.wandb_utils, checkpointing.mlflow_utils, checkpointing.comet_utils):
            stack.enter_context(patch.object(logger, "on_load_checkpoint_success"))
        stack.enter_context(patch.object(checkpointing.fault_tolerance, "on_checkpoint_loaded"))
        local_manager = Mock()
        local_manager.load.return_value = (SimpleNamespace(common_state_dict=payload), None)
        yield SimpleNamespace(
            state=state,
            cfg=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            payload=payload,
            mocks=mocks,
            checkpoint_type=checkpoint_type,
            model=fixtures["mock_model"],
            context={"local_checkpoint_manager": local_manager},
        )


def _resume(case, **kwargs):
    return checkpointing.load_checkpoint(
        case.state,
        case.model,
        case.optimizer,
        case.scheduler,
        checkpointing_context=case.context,
        **kwargs,
    )


@pytest.mark.parametrize("with_hook", [False, True])
def test_migration_preserves_progress_before_microbatches_and_scheduler_override(migration_resume, with_hook):
    case = migration_resume
    original_train_state = case.state.train_state
    events = []

    def migrate(payload, train_state):
        assert train_state is case.state.train_state
        assert train_state is not original_train_state
        assert case.scheduler.num_steps == 0
        case.mocks["_load_model_state_dict"].assert_not_called()
        train_state.consumed_train_samples = payload["opt_param_scheduler"]["num_steps"]
        events.append("migration")

    hook = Mock(side_effect=migrate)
    case.mocks["update_num_microbatches"].side_effect = lambda **_: events.append("microbatches")
    _resume(case, checkpoint_state_migration_hook=hook if with_hook else None)
    expected_samples = 600 if with_hook else 0
    assert case.state.train_state.consumed_train_samples == expected_samples
    assert case.scheduler.num_steps == expected_samples
    assert case.optimizer.param_groups[0]["lr"] == pytest.approx(0.00075 if with_hook else 0.0)
    case.mocks["update_num_microbatches"].assert_called_once_with(consumed_samples=expected_samples, verbose=True)
    assert events == (["migration", "microbatches"] if with_hook else ["microbatches"])
    assert hook.call_count == int(with_hook)
    assert original_train_state.consumed_train_samples == 0


@pytest.mark.parametrize("mode", ["metadata_only", "no_optimizer", "override_disabled"])
def test_migration_is_independent_of_optimizer_loading(migration_resume, mode):
    case = migration_resume
    case.cfg.checkpoint.load_optim = mode != "no_optimizer"
    case.cfg.scheduler.override_opt_param_scheduler = mode != "override_disabled"

    def migrate(payload, train_state):
        train_state.consumed_train_samples = 600

    hook = Mock(side_effect=migrate)
    _resume(case, checkpoint_state_migration_hook=hook, skip_load_to_model_and_opt=mode == "metadata_only")
    hook.assert_called_once_with(case.payload, case.state.train_state)
    case.mocks["update_num_microbatches"].assert_called_once_with(consumed_samples=600, verbose=True)
    if mode == "no_optimizer":
        assert case.scheduler.num_steps == 0


@pytest.mark.parametrize("mode", ["finetune", "release", "missing", "hf_initialization"])
def test_migration_skips_non_resume_loads(migration_resume, mode):
    case = migration_resume
    hook = Mock()
    if mode == "finetune":
        case.cfg.checkpoint.finetune = True
        case.model[0].hide_loss_modules.return_value = nullcontext()
    elif mode == "release":
        case.mocks["base_load"].return_value = (case.payload, "/checkpoints/release", True, case.checkpoint_type)
    elif mode == "missing":
        case.mocks["base_load"].return_value = (None, "", False, case.checkpoint_type)
    else:
        case.cfg.checkpoint.load = None
        case.cfg.checkpoint.pretrained_checkpoint = "/pretrained"
        case.mocks["checkpoint_exists"].side_effect = lambda path: path == "/pretrained"
        case.mocks["is_hf_checkpoint_dir"].side_effect = lambda path: path == "/pretrained"
    with (
        patch.object(checkpointing, "_has_global_non_persistent_checkpoint", return_value=False),
        patch.object(checkpointing, "_load_hf_pretrained_checkpoint", return_value=(0, 0)),
    ):
        _resume(case, checkpoint_state_migration_hook=hook)
    hook.assert_not_called()


def test_migration_error_propagates_before_final_restoration_calls(migration_resume):
    case = migration_resume
    error = ValueError("unsupported client checkpoint metadata")
    hook = Mock(side_effect=error)
    with pytest.raises(ValueError, match="unsupported client checkpoint metadata") as exc:
        _resume(case, checkpoint_state_migration_hook=hook)
    assert exc.value is error
    hook.assert_called_once()
    case.mocks["update_num_microbatches"].assert_not_called()
    case.mocks["_load_model_state_dict"].assert_not_called()
    assert case.scheduler.num_steps == 0
    assert "max_lr" not in case.optimizer.param_groups[0]
