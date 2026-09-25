# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression coverage for optimizer, RNG, and dataloader checkpoint resumes."""

import random
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from megatron.core import dist_checkpointing, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

import megatron.bridge.training.checkpointing as checkpointing
from megatron.bridge.training.config import CheckpointConfig, OptimizerConfig
from megatron.bridge.training.state import TrainState
from tests.unit_tests.training.test_checkpointing import load_checkpoint_fixtures  # noqa: F401


@pytest.mark.parametrize(
    "mode", ["resume", "skip", "finetune", "no_optim", "release", "stub", "missing_optimizer", "load_error"]
)
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("precision", ["fp8", "fp4", "bf16"])
def test_quantized_resume_requires_successful_main_parameter_restore(
    load_checkpoint_fixtures, mode, precision, override
):
    fixtures = load_checkpoint_fixtures
    cfg, state = fixtures["mock_cfg"], fixtures["mock_state"]
    cfg.peft = None
    cfg.scheduler.override_opt_param_scheduler = override
    cfg.optimizer.lr, cfg.optimizer.min_lr = 0.02, 0.002
    fixtures["mock_model"][0].hide_loss_modules.return_value = nullcontext()
    cfg.checkpoint.load_rng = False
    cfg.checkpoint.load_optim = mode != "no_optim"
    cfg.checkpoint.finetune = mode == "finetune"
    cfg.ddp.fp8_param_gather = precision == "fp8"
    cfg.ddp.fp4_param_gather = precision == "fp4"
    state.train_state = TrainState(step=3, consumed_train_samples=48)
    optimizer, scheduler = fixtures["mock_optimizer"], fixtures["mock_scheduler"]
    optimizer.is_stub_optimizer = mode == "stub"
    optimizer.param_groups = [{"max_lr": 0.2, "min_lr": 0.1}]
    events = []
    optimizer.load_state_dict.side_effect = lambda *_: events.append("optimizer")
    optimizer.quantize_and_sync_model_params_from_main_params.side_effect = lambda: events.append("quantize")
    scheduler.load_state_dict.side_effect = lambda *_: events.append("scheduler")
    if mode == "load_error":
        optimizer.load_state_dict.side_effect = KeyError("missing main parameters")
    payload = {"model": {}, "optimizer": {}, "opt_param_scheduler": {}, "checkpoint_version": 3.0}
    run_config = {
        "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
        "checkpoint": {"save_optim": mode != "missing_optimizer", "save_rng": False, "fully_parallel_save": False},
    }
    with ExitStack() as stack:
        replacements = {
            "is_hf_checkpoint_dir": False,
            "is_checkpoint_iteration_directory": True,
            "file_exists": True,
            "read_run_config": run_config,
            "read_train_state": state.train_state,
            "update_num_microbatches": None,
            "_get_model_glu_interleave_sizes": (None, None),
            "_load_model_state_dict": None,
            "generate_state_dict": {"model": {}},
            "unwrap_model": fixtures["mock_model"],
        }
        for name, value in replacements.items():
            stack.enter_context(patch.object(checkpointing, name, return_value=value))
        stack.enter_context(patch.object(checkpointing.dist_checkpointing, "load_content_metadata", return_value={}))
        stack.enter_context(
            patch.object(
                checkpointing,
                "_load_base_checkpoint",
                return_value=(
                    payload,
                    "/checkpoints/iter_0000003",
                    mode == "release",
                    checkpointing.CheckpointType.GLOBAL,
                ),
            )
        )
        stack.enter_context(patch.object(torch.distributed, "is_initialized", return_value=False))
        for logger in (checkpointing.wandb_utils, checkpointing.mlflow_utils, checkpointing.comet_utils):
            stack.enter_context(patch.object(logger, "on_load_checkpoint_success"))
        kwargs = dict(pg_collection=Mock(), skip_load_to_model_and_opt=mode == "skip")
        if mode == "load_error":
            with pytest.raises(KeyError, match="missing main"):
                checkpointing._load_checkpoint_from_path(
                    "/checkpoints", state, fixtures["mock_model"], optimizer, scheduler, **kwargs
                )
        else:
            checkpointing._load_checkpoint_from_path(
                "/checkpoints", state, fixtures["mock_model"], optimizer, scheduler, **kwargs
            )
    expected = mode == "resume" and precision != "bf16"
    assert optimizer.quantize_and_sync_model_params_from_main_params.call_count == int(expected)
    applied_override = override and mode in ("resume",)
    if applied_override:
        assert optimizer.param_groups[0] == {"max_lr": 0.02, "min_lr": 0.002}
        assert scheduler.num_steps == 48
        scheduler.step.assert_called_once_with(increment=0)
    else:
        assert optimizer.param_groups[0] == {"max_lr": 0.2, "min_lr": 0.1}
        scheduler.step.assert_not_called()
    if expected:
        assert events == ["optimizer", "scheduler", "quantize"]


def test_scheduler_runtime_override_changes_next_lr_and_preserves_multipliers():
    groups = [
        {"params": [torch.nn.Parameter(torch.ones(1))], "max_lr": 0.2, "min_lr": 0.1, "lr_mult": 0.5},
        {"params": [torch.nn.Parameter(torch.ones(1))], "max_lr": 0.3, "min_lr": 0.1, "is_decoupled_lr": True},
    ]
    optimizer = torch.optim.SGD(groups, lr=0.2)
    scheduler = OptimizerParamScheduler(optimizer, 0, 0.2, 0.1, 0, 100, "linear", 0, 0, 100, "constant")
    scheduler.num_steps = 80
    cfg = SimpleNamespace(optimizer=OptimizerConfig(lr=0.02, min_lr=0.002, decoupled_lr=0.04, decoupled_min_lr=0.004))
    checkpointing._restore_scheduler_runtime_overrides(
        cfg, TrainState(consumed_train_samples=50), optimizer, scheduler
    )
    assert scheduler.num_steps == 50
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.011)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.022)
    assert optimizer.param_groups[0]["lr_mult"] == 0.5
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002 + 0.018 * 0.49)


@pytest.mark.parametrize("stub,missing_scheduler", [(True, False), (False, True)])
def test_scheduler_override_skips_unavailable_optimizer_or_scheduler(stub, missing_scheduler):
    optimizer = SimpleNamespace(is_stub_optimizer=stub, param_groups=[{"max_lr": 0.1}])
    scheduler = None if missing_scheduler else Mock()
    checkpointing._restore_scheduler_runtime_overrides(None, None, optimizer, scheduler)
    assert optimizer.param_groups == [{"max_lr": 0.1}]


def test_scheduler_override_preserves_unspecified_decoupled_bounds():
    optimizer = SimpleNamespace(param_groups=[{"is_decoupled_lr": True, "max_lr": 0.1, "min_lr": 0.01}])
    cfg = SimpleNamespace(optimizer=OptimizerConfig(lr=0.02, min_lr=0.002))
    checkpointing._restore_scheduler_runtime_overrides(cfg, TrainState(), optimizer, Mock())
    assert optimizer.param_groups[0]["max_lr"] == 0.1
    assert optimizer.param_groups[0]["min_lr"] == 0.01


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_rng_load_coordinates_follow_saved_layout(legacy, rank):
    current = ShardedObject("rng_state.module", [], (1, 1, 4), (0, 0, rank), 0)
    stored = (
        [ShardedObject(current.key, [], (1, 1), (0, 0), 0)]
        if legacy
        else [ShardedObject(current.key, [], (1, 1, 2), (0, 0, i), 0) for i in range(2)]
    )
    result = checkpointing._match_rng_state_metadata(current, {item.unique_key: item for item in stored})
    if legacy:
        assert result.global_shape == (1, 1)
        assert result.replica_id == rank
    elif rank >= 2:
        assert result is None
    else:
        assert result.global_shape == (1, 1, 2)
        assert result.global_offset == (0, 0, rank)


def test_rng_shrinking_world_keeps_fresh_state_without_partial_shard_access():
    current = ShardedObject("rng_state", [], (1, 1, 1), (0, 0, 0), 0)
    stored = [ShardedObject(current.key, [], (1, 1, 2), (0, 0, rank), 0) for rank in range(2)]
    assert checkpointing._match_rng_state_metadata(current, {item.unique_key: item for item in stored}) is None


def test_rng_legacy_layout_resolves_remote_storage_metadata(tmp_path):
    from megatron.core.msc_utils import MultiStorageClientFeature

    current = ShardedObject("rng_state", [], (1, 1, 2), (0, 0, 1), 0)
    legacy = ShardedObject("rng_state", [], (1, 1), (0, 0), 0)
    (tmp_path / ".metadata").touch()
    MultiStorageClientFeature.enable()
    try:
        with patch.object(checkpointing, "TorchDistLoadShardedStrategy") as strategy:
            strategy.return_value.load_sharded_metadata.return_value = {legacy.unique_key: legacy}
            result = checkpointing._align_rng_state_sharded_metadata(current, f"msc://default{tmp_path}")
        assert result.global_shape == (1, 1)
        assert result.replica_id == 1
        strategy.return_value.load_sharded_metadata.assert_called_once_with(f"msc://default{tmp_path}")
    finally:
        MultiStorageClientFeature.disable()


class _Group:
    def __init__(self, rank=0, size=1):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _Cursor:
    def __init__(self, cursor, rank_independent=False):
        self.cursor = cursor
        self.is_save_state_rank_independent = rank_independent
        self.save_calls = 0

    def save_state(self):
        self.save_calls += 1
        return {"cursor": self.cursor}

    def restore_state(self, state):
        self.cursor = state["cursor"]


@pytest.mark.parametrize("rank_independent", [False, True])
def test_loader_round_trip_uses_full_data_group_and_preserves_model_files(tmp_path, rank_independent):
    # Both GTP peers have replicate-DP rank zero but own different data.
    pg = SimpleNamespace(dp=_Group(), pp=_Group(), tp=_Group(), cp=_Group())
    directory = Path(checkpointing.get_checkpoint_name(str(tmp_path), 7))
    directory.mkdir()
    (directory / "model.pt").write_bytes(b"model checkpoint")
    (directory / "train_dataloader_dprank009.pt").write_bytes(b"stale loader")
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "barrier"),
        patch.object(checkpointing, "print_rank_0"),
    ):
        for rank in range(2):
            cursor = _Cursor(10 + rank, rank_independent)
            iterator = SimpleNamespace(iterable=cursor)
            group = _Group(rank, 2)
            checkpointing.maybe_save_dataloader_state(
                [], iterator, 7, str(tmp_path), pg_collection=pg, data_parallel_group=group
            )
            assert cursor.save_calls == int(rank == 0 or not rank_independent)
        for rank in range(2):
            cursor = _Cursor(-1, rank_independent)
            checkpointing.maybe_load_dataloader_state(
                SimpleNamespace(iterable=cursor),
                7,
                str(tmp_path),
                pg_collection=pg,
                data_parallel_group=_Group(rank, 2),
            )
            assert cursor.cursor == (10 if rank_independent else 10 + rank)
    assert (directory / "model.pt").read_bytes() == b"model checkpoint"
    assert not (directory / "train_dataloader_dprank009.pt").exists()
    assert len(list(directory.glob("train_dataloader_dprank*.pt"))) == (1 if rank_independent else 2)


@pytest.mark.parametrize(
    "fp8,fp4,recipe,saved_rows,expected",
    [
        (None, None, None, 12, False),
        ("e4m3", None, "mxfp8", 32, True),
        (None, "e2m1", None, 16, True),
        ("e4m3", None, "mxfp8", 9, False),
    ],
)
def test_gtp_load_only_allows_recognized_padding(fp8, fp4, recipe, saved_rows, expected):
    tensor = ShardedTensor.from_rank_offsets("weight", torch.zeros(10, 4))
    saved = ShardedTensor.from_rank_offsets("weight", torch.zeros(saved_rows, 4))
    cfg = SimpleNamespace(model=SimpleNamespace(fp8=fp8, fp4=fp4, fp8_recipe=recipe))
    ckpt_cfg = CheckpointConfig(fully_parallel_load=False)
    with (
        patch("megatron.core.dist_checkpointing.serialization.load_tensors_metadata", return_value={"weight": saved}),
        patch.object(checkpointing.dist_checkpointing, "load", return_value={}) as load,
    ):
        checkpointing._load_global_dist_base_checkpoint(
            "/checkpoints", ckpt_cfg, False, {"weight": tensor}, 1, False, pg_collection=SimpleNamespace(), cfg=cfg
        )
    assert tensor.allow_shape_mismatch is expected
    assert load.call_args.kwargs["strict"] == ckpt_cfg.dist_ckpt_strictness


def test_missing_required_checkpoint_exits_with_failure(tmp_path):
    cfg = CheckpointConfig(exit_on_missing_checkpoint=True)
    with pytest.raises(SystemExit) as error:
        checkpointing._load_base_checkpoint(str(tmp_path), cfg, pg_collection=SimpleNamespace())
    assert error.value.code == 1


def _rng_round_trip_worker(rank, rendezvous, checkpoint_dir, legacy, fully_parallel):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        singles = [torch.distributed.new_group([i]) for i in range(2)]
        pg = SimpleNamespace(pp=singles[rank], tp=singles[rank], dp_cp=torch.distributed.group.WORLD, ep=singles[rank])
        random.seed(100 + rank)
        np.random.seed(100 + rank)
        torch.manual_seed(100 + rank)
        torch.cuda.manual_seed(100 + rank)
        tracker = tensor_parallel.get_cuda_rng_tracker()
        tracker.reset()
        tracker.add("test", 1000 + rank)
        shard = checkpointing.get_rng_state(False, pg_collection=pg, module_name="decoder")
        if legacy:
            shard = replace(shard, global_shape=(1, 1), global_offset=(0, 0), replica_id=rank)
        from megatron.core.dist_checkpointing.strategies.fully_parallel import (
            FullyParallelLoadStrategyWrapper,
            FullyParallelSaveStrategyWrapper,
        )
        from megatron.core.dist_checkpointing.strategies.torch import (
            TorchDistLoadShardedStrategy,
            TorchDistSaveShardedStrategy,
        )

        save_strategy = TorchDistSaveShardedStrategy("torch_dist", 1)
        load_strategy = TorchDistLoadShardedStrategy()
        if fully_parallel:
            save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, pg.dp_cp)
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy, pg.dp_cp)
        dist_checkpointing.save({"rng_state": shard}, checkpoint_dir, sharded_strategy=save_strategy)
        expected = (random.random(), np.random.rand(), torch.rand(4), torch.rand(4, device="cuda").cpu())
        expected_all = [None, None]
        torch.distributed.all_gather_object(expected_all, expected)
        with tracker.fork("test"):
            expected_tracker = torch.rand(4, device="cuda").cpu()
        all_tracker = [None, None]
        torch.distributed.all_gather_object(all_tracker, expected_tracker)
        template = checkpointing.get_rng_state(False, pg_collection=pg, module_name="decoder")
        template = checkpointing._align_rng_state_sharded_metadata(template, checkpoint_dir)
        state = dist_checkpointing.load({"rng_state": template}, checkpoint_dir, sharded_strategy=load_strategy)[
            "rng_state"
        ][0]
        random.setstate(state["random_rng_state"])
        np.random.set_state(state["np_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"])
        tracker.set_states(state["rng_tracker_states"])
        actual = (random.random(), np.random.rand(), torch.rand(4), torch.rand(4, device="cuda").cpu())
        target = expected_all[0 if legacy else rank]
        assert actual[:2] == target[:2]
        torch.testing.assert_close(actual[2], target[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], target[3], rtol=0, atol=0)
        with tracker.fork("test"):
            torch.testing.assert_close(
                torch.rand(4, device="cuda").cpu(), all_tracker[0 if legacy else rank], rtol=0, atol=0
            )
        assert not torch.equal(expected_all[0][2], expected_all[1][2])

        from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict

        intermediate, _ = MCoreTensorAwareStateDict.from_state_dict(
            {"rng_state": shard}, algo="atomic", parallelization_group=pg.dp_cp
        )
        local_template = checkpointing.get_rng_state(False, pg_collection=pg, module_name="decoder")
        local_cfg = CheckpointConfig(non_persistent_ckpt_type="local", non_persistent_local_ckpt_algo="atomic")
        local_state, _, _, _ = checkpointing._load_non_persistent_base_checkpoint(
            "",
            local_cfg,
            False,
            {"rng_state": local_template},
            1,
            checkpointing_context={"local_checkpoint_manager": SimpleNamespace(load=lambda: (intermediate, "local"))},
            pg_collection=pg,
        )
        torch.testing.assert_close(local_state["rng_state"][0]["torch_rng_state"], shard.data[0]["torch_rng_state"])

    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.run_only_on("gpu")
@pytest.mark.parametrize("legacy,fully_parallel", [(False, False), (False, True), (True, True)])
def test_two_rank_rng_checkpoint_round_trip(tmp_path, legacy, fully_parallel):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    (tmp_path / "checkpoint").mkdir()
    torch.multiprocessing.spawn(
        _rng_round_trip_worker,
        args=(str(tmp_path / "rdzv"), str(tmp_path / "checkpoint"), legacy, fully_parallel),
        nprocs=2,
        join=True,
    )
