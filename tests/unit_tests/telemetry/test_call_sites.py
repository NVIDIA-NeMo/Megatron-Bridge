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

"""Tests that the startup, evaluation, and checkpoint call sites record the documented spans.

The spans of the training loop itself are covered end to end by
``tests/functional_tests/test_groups/training/test_telemetry.py``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from megatron.bridge import telemetry
from megatron.bridge.training.callbacks import CallbackManager
from megatron.bridge.training.checkpointing import maybe_finalize_async_save, save_checkpoint, schedule_async_save
from megatron.bridge.training.pretrain import _pretrain, pretrain
from megatron.bridge.training.train import save_checkpoint_and_time
from tests.unit_tests.telemetry.utils import clean_telemetry, telemetry_capture  # noqa: F401
from tests.unit_tests.training.test_checkpointing import save_checkpoint_fixtures  # noqa: F401
from tests.unit_tests.training.test_eval import _make_evaluate_state, _ModeTrackingModel, _run_evaluate


pytestmark = pytest.mark.unit


def _current_span_name() -> str | None:
    return getattr(trace.get_current_span(), "name", None)


class TestStartup:
    """pretrain installs telemetry before setup, traces setup, and flushes on every exit."""

    def test_telemetry_is_installed_before_training(self):
        calls = []
        config = MagicMock()
        config.inprocess_restart = None

        with (
            patch("megatron.bridge.training.pretrain.runtime_config_update"),
            patch("megatron.bridge.training.pretrain.GlobalState"),
            patch("megatron.bridge.training.pretrain.init_telemetry", side_effect=lambda cfg: calls.append(cfg)),
            patch("megatron.bridge.training.pretrain._pretrain", side_effect=lambda **_: calls.append("_pretrain")),
        ):
            pretrain(config, MagicMock())

        assert calls == [config, "_pretrain"]

    def test_failed_setup_is_recorded_and_flushed(self, telemetry_capture):
        capture = telemetry_capture("default")

        def failing_setup(*args, **kwargs):
            raise RuntimeError(f"setup failed in {_current_span_name()}")

        with (
            patch("megatron.bridge.training.pretrain.dist") as mock_dist,
            patch("megatron.bridge.training.pretrain.get_dataset_provider"),
            patch("megatron.bridge.training.pretrain.setup", side_effect=failing_setup),
            patch("megatron.bridge.training.pretrain.flush_telemetry") as mock_flush,
        ):
            mock_dist.is_initialized.return_value = True
            with pytest.raises(RuntimeError, match="setup failed in megatron.startup"):
                _pretrain(MagicMock(), MagicMock())

        assert capture.span_named("megatron.startup").status.status_code is StatusCode.ERROR
        mock_flush.assert_called_once_with()


class TestEvaluate:
    """evaluate records one span per pass, numbered by the training iteration."""

    @pytest.mark.parametrize(("is_test", "split"), [(False, "validation"), (True, "test")])
    def test_evaluate_span(self, telemetry_capture, is_test, split):
        capture = telemetry_capture("default")
        state = _make_evaluate_state(eval_iters=0)
        state.train_state.step = 40

        _run_evaluate(state=state, model=_ModeTrackingModel(), callback_manager=CallbackManager(), is_test=is_test)

        attributes = capture.span_named("megatron.evaluate").attributes
        assert attributes[telemetry.ITERATION] == 40
        assert attributes["megatron.evaluate.split"] == split


class TestCheckpointSpans:
    """Saving a checkpoint records the exposed save, the save, and their phases."""

    @staticmethod
    def _loop_state():
        timers = Mock(return_value=Mock())
        timers.log = Mock()
        return SimpleNamespace(
            timers=timers,
            energy_monitor=None,
            cfg=SimpleNamespace(
                ddp=SimpleNamespace(use_megatron_fsdp=False, overlap_param_gather=False),
                optimizer=SimpleNamespace(use_distributed_optimizer=False),
                model=SimpleNamespace(fp8=None, seq_length=1),
                logger=SimpleNamespace(log_progress=False),
                checkpoint=SimpleNamespace(async_save=False),
            ),
            train_state=SimpleNamespace(floating_point_operations_so_far=0, consumed_train_samples=0, step=30),
            start_time=0,
        )

    def test_exposed_save_wraps_the_checkpoint_manager(self, telemetry_capture):
        capture = telemetry_capture("default")
        saved_under = []
        checkpoint_manager = Mock()
        checkpoint_manager.save.side_effect = lambda *args, **kwargs: saved_under.append(_current_span_name())

        save_checkpoint_and_time(
            state=self._loop_state(),
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=1.0,
            checkpoint_manager=checkpoint_manager,
        )

        assert capture.span_named("megatron.checkpoint.exposed_save").attributes[telemetry.ITERATION] == 30
        assert saved_under == ["megatron.checkpoint.exposed_save"]

    def test_save_records_phases_with_attributes(self, telemetry_capture, save_checkpoint_fixtures, tmp_path):
        capture = telemetry_capture("default")
        state = save_checkpoint_fixtures["mock_state"]
        state.cfg.checkpoint.save = str(tmp_path)
        state.cfg.checkpoint.most_recent_k = -1
        pg_collection = Mock()
        pg_collection.expt_dp.rank.return_value = 0
        pg_collection.tp.rank.return_value = 0
        pg_collection.tp.size.return_value = 1
        pg_collection.pp.rank.return_value = 0
        pg_collection.pp.size.return_value = 1

        with (
            patch("megatron.bridge.training.checkpointing.dist_checkpointing.save", return_value=None),
            patch("megatron.bridge.training.checkpointing.TorchDistSaveShardedStrategy", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_rng_state", return_value=Mock()),
            patch("megatron.bridge.training.checkpointing.get_rerun_state_machine") as mock_rerun,
            patch("megatron.bridge.training.checkpointing._get_model_glu_interleave_sizes", return_value=(None, None)),
            patch(
                "megatron.bridge.training.checkpointing.generate_state_dict",
                return_value={"model": {"weight": Mock()}},
            ),
            patch(
                "megatron.bridge.training.checkpointing.unwrap_model",
                return_value=save_checkpoint_fixtures["mock_model"],
            ),
            patch("megatron.bridge.training.checkpointing.save_sharded_modelopt_state"),
            patch("megatron.bridge.training.checkpointing.maybe_save_dataloader_state"),
            patch("megatron.bridge.training.checkpointing.fault_tolerance"),
            patch("megatron.bridge.training.checkpointing.is_empty_async_queue", return_value=True),
            patch("megatron.bridge.training.checkpointing.get_rank_safe", return_value=0),
            patch("megatron.bridge.training.checkpointing.is_last_rank", return_value=False),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            mock_rerun.return_value.state_dict.return_value = {}
            save_checkpoint(
                state,
                save_checkpoint_fixtures["mock_model"],
                save_checkpoint_fixtures["mock_optimizer"],
                save_checkpoint_fixtures["mock_scheduler"],
                1000000,
                checkpointing_context={},
                pg_collection=pg_collection,
            )

        save_span = capture.span_named("megatron.checkpoint.save")
        assert dict(save_span.attributes) == {
            telemetry.ITERATION: 1000,
            "megatron.checkpoint.format": "torch_dist",
            "megatron.checkpoint.type": "global",
            "megatron.checkpoint.async": False,
            "megatron.checkpoint.non_persistent": False,
        }
        for phase in ("megatron.checkpoint.save.state_dict", "megatron.checkpoint.save.io_write"):
            assert capture.span_named(phase).parent.span_id == save_span.context.span_id

    def test_disabled_group_records_nothing(self, telemetry_capture):
        capture = telemetry_capture("job")

        save_checkpoint_and_time(
            state=self._loop_state(),
            model=[Mock()],
            optimizer=Mock(),
            opt_param_scheduler=Mock(),
            num_floating_point_operations_so_far=1.0,
            checkpoint_manager=Mock(),
        )

        assert capture.spans() == []


class TestAsyncCheckpointSpans:
    """Async saves record the hand-off to the worker and, while a save is pending, its finalization."""

    @staticmethod
    def _state(queue):
        return SimpleNamespace(async_calls_queue=queue)

    def test_schedule_records_call_index(self, telemetry_capture):
        capture = telemetry_capture("default")
        queue = Mock()
        queue.schedule_async_request.return_value = 7
        request = object()

        schedule_async_save(self._state(queue), request)

        queue.schedule_async_request.assert_called_once_with(request)
        assert capture.span_named("megatron.checkpoint.save.schedule").attributes["nvrx.call_idx"] == 7

    @pytest.mark.parametrize("blocking", [False, True])
    def test_finalize_is_traced_while_a_save_is_pending(self, telemetry_capture, blocking):
        capture = telemetry_capture("default")
        queue = Mock()
        queue.get_num_unfinalized_calls.return_value = 1
        queue.maybe_finalize_async_calls.return_value = [3]

        maybe_finalize_async_save(self._state(queue), SimpleNamespace(async_save=True), blocking=blocking)

        queue.maybe_finalize_async_calls.assert_called_once_with(blocking)
        attributes = capture.span_named("megatron.checkpoint.save.finalize").attributes
        assert attributes["megatron.checkpoint.finalize.blocking"] is blocking
        assert attributes["megatron.checkpoint.finalize.count"] == 1

    @pytest.mark.parametrize(("span_groups", "pending_calls"), [("default", 0), ("job", 1)])
    def test_finalize_is_not_traced_otherwise(self, telemetry_capture, span_groups, pending_calls):
        """No span when nothing is pending, the common per-iteration case, or when the group is off."""
        capture = telemetry_capture(span_groups)
        queue = Mock()
        queue.get_num_unfinalized_calls.return_value = pending_calls
        queue.maybe_finalize_async_calls.return_value = []

        maybe_finalize_async_save(self._state(queue), SimpleNamespace(async_save=True), terminate=True)

        queue.maybe_finalize_async_calls.assert_called_once_with(False)
        queue.close.assert_called_once_with()
        assert capture.spans() == []
