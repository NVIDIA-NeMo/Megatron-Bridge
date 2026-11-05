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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.training.callbacks import CallbackManager
from megatron.bridge.training.eval import evaluate, evaluate_and_print_results


pytestmark = pytest.mark.unit


def _make_state():
    return SimpleNamespace(
        wandb_logger=None,
        mlflow_logger=None,
        comet_logger=None,
        train_state=SimpleNamespace(step=0, consumed_train_samples=0),
        cfg=SimpleNamespace(logger=SimpleNamespace(log_validation_ppl_to_tensorboard=False)),
    )


class _ModeTrackingModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _StrictTimer:
    def __init__(self):
        self.started = False

    def start(self, barrier=False):
        assert not self.started, "timer has already been started"
        self.started = True

    def stop(self):
        assert self.started, "timer is not started"
        self.started = False


class _StrictTimers:
    def __init__(self):
        self.timer = _StrictTimer()

    def __call__(self, name, log_level=None):
        assert name == "evaluate"
        return self.timer

    def log(self, names):
        assert names == ["evaluate"]


def _make_evaluate_state(*, eval_iters, exit_duration_in_mins=None):
    timer = MagicMock()
    timers = MagicMock(return_value=timer)
    timers.log = MagicMock()
    return SimpleNamespace(
        timers=timers,
        start_time=0.0,
        train_state=SimpleNamespace(consumed_valid_samples=0),
        cfg=SimpleNamespace(
            validation=SimpleNamespace(
                eval_global_batch_size=1,
                eval_micro_batch_size=1,
                eval_iters=eval_iters,
            ),
            data_parallel_size=1,
            model=SimpleNamespace(
                seq_length=1,
                virtual_pipeline_model_parallel_size=None,
                moe_expert_rank_capacity_factor=None,
            ),
            dist=SimpleNamespace(use_decentralized_pg=True),
            optimizer=SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=False),
            ddp=SimpleNamespace(overlap_param_gather=False),
            dataset=SimpleNamespace(dataloader_type="single", seq_length=1),
            train=SimpleNamespace(
                empty_unused_memory_level=0,
                exit_duration_in_mins=exit_duration_in_mins,
            ),
        ),
    )


def _run_evaluate(*, state, model, callback_manager, is_test=False, timelimit_hit=False):
    pg_collection = SimpleNamespace(
        pp=SimpleNamespace(size=lambda: 1),
        dp=SimpleNamespace(size=lambda: 1),
        dp_cp=object(),
    )
    rerun_state_machine = MagicMock()
    done_cuda = MagicMock()
    done_cuda.item.return_value = int(timelimit_hit)

    with (
        patch("megatron.bridge.training.eval.prepare_forward_step_func", return_value=MagicMock()),
        patch("megatron.bridge.training.eval.get_pg_collection", return_value=pg_collection),
        patch("megatron.bridge.training.eval.get_model_config", return_value=SimpleNamespace()),
        patch("megatron.bridge.training.eval.get_rerun_state_machine", return_value=rerun_state_machine),
        patch("megatron.bridge.training.eval.is_full_iteration_cuda_graph", return_value=False),
        patch("megatron.bridge.training.eval.get_forward_backward_func", return_value=MagicMock(return_value=[{}])),
        patch("megatron.bridge.training.eval.is_pp_last_stage", return_value=False),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_start"),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_end"),
        patch("megatron.bridge.training.eval.time.time", return_value=120.0),
        patch("megatron.bridge.training.eval.torch.tensor", return_value=done_cuda),
        patch("megatron.bridge.training.eval.torch.distributed.all_reduce"),
    ):
        return evaluate(
            state=state,
            forward_step_func=MagicMock(),
            data_iterator=object(),
            model=[model],
            process_non_loss_data_func=None,
            config=SimpleNamespace(timers=state.timers),
            p2p_communicator=MagicMock(),
            callback_manager=callback_manager,
            is_test=is_test,
        )


@patch("megatron.bridge.training.eval.print_rank_last")
@patch("megatron.bridge.training.eval.evaluate")
def test_evaluate_and_print_results_returns_loss_dict(mock_evaluate, mock_print_rank_last):
    losses = {"lm loss": torch.tensor(1.0)}
    mock_evaluate.return_value = (losses, None, False)
    callback_manager = CallbackManager()

    result = evaluate_and_print_results(
        state=_make_state(),
        prefix="calibration",
        forward_step_func=MagicMock(),
        data_iterator=object(),
        model=[MagicMock()],
        config=SimpleNamespace(),
        write_to_tensorboard=False,
        callback_manager=callback_manager,
    )

    assert result is losses
    assert mock_evaluate.call_args.kwargs["callback_manager"] is callback_manager
    assert mock_print_rank_last.called


@patch("megatron.bridge.training.eval.print_rank_last")
@patch("megatron.bridge.training.eval.evaluate")
def test_evaluate_and_print_results_returns_none_on_timelimit(
    mock_evaluate,
    mock_print_rank_last,
):
    mock_evaluate.return_value = (None, None, True)
    callback_manager = CallbackManager()

    result = evaluate_and_print_results(
        state=_make_state(),
        prefix="calibration",
        forward_step_func=MagicMock(),
        data_iterator=object(),
        model=[MagicMock()],
        config=SimpleNamespace(),
        write_to_tensorboard=False,
        callback_manager=callback_manager,
    )

    assert result is None
    assert mock_evaluate.call_args.kwargs["callback_manager"] is callback_manager
    mock_print_rank_last.assert_not_called()


@pytest.mark.parametrize(
    ("is_test", "start_event", "end_event"),
    [
        (False, "on_eval_start", "on_eval_end"),
        (True, "on_test_start", "on_test_end"),
    ],
)
def test_evaluate_fires_lifecycle_callbacks_in_eval_mode(is_test, start_event, end_event):
    state = _make_evaluate_state(eval_iters=0)
    model = _ModeTrackingModel()
    callback_manager = CallbackManager()
    observed = []
    callback_manager.register(
        start_event,
        lambda context: observed.append((start_event, context.model[0].training, context.total_loss_dict)),
    )
    callback_manager.register(
        end_event,
        lambda context: observed.append((end_event, context.model[0].training, context.total_loss_dict)),
    )

    result = _run_evaluate(
        state=state,
        model=model,
        callback_manager=callback_manager,
        is_test=is_test,
    )

    assert result == ({}, None, False)
    assert observed == [
        (start_event, False, None),
        (end_event, False, {}),
    ]
    assert model.training


def test_evaluate_timelimit_fires_start_but_not_end_callback():
    state = _make_evaluate_state(eval_iters=1, exit_duration_in_mins=1)
    model = _ModeTrackingModel()
    callback_manager = CallbackManager()
    observed = []
    callback_manager.register(
        "on_eval_start",
        lambda context: observed.append(("on_eval_start", context.model[0].training)),
    )
    callback_manager.register(
        "on_eval_end",
        lambda context: observed.append(("on_eval_end", context.model[0].training)),
    )

    result = _run_evaluate(
        state=state,
        model=model,
        callback_manager=callback_manager,
        timelimit_hit=True,
    )

    assert result == (None, None, True)
    assert observed == [("on_eval_start", False)]


def test_evaluate_timelimit_releases_lifecycle_state_before_next_evaluation():
    state = _make_evaluate_state(eval_iters=1, exit_duration_in_mins=1)
    state.timers = _StrictTimers()
    model = _ModeTrackingModel()
    callback_manager = CallbackManager()

    validation_result = _run_evaluate(
        state=state,
        model=model,
        callback_manager=callback_manager,
        timelimit_hit=True,
    )
    test_result = _run_evaluate(
        state=state,
        model=model,
        callback_manager=callback_manager,
        is_test=True,
        timelimit_hit=True,
    )

    assert validation_result == (None, None, True)
    assert test_result == (None, None, True)
    assert not state.timers.timer.started
    assert model.training


def test_evaluate_uses_injected_eval_data_parallel_size_for_microbatches():
    """An injected eval process-group layout owns eval global-batch semantics."""
    state = _make_evaluate_state(eval_iters=1)
    state.cfg.validation.eval_global_batch_size = 4
    state.cfg.validation.eval_micro_batch_size = 1
    state.cfg.data_parallel_size = 2
    model = _ModeTrackingModel()
    forward_backward_func = MagicMock(return_value=[{}])
    rerun_state_machine = MagicMock()
    eval_pg_collection = SimpleNamespace(
        pp=SimpleNamespace(size=lambda: 1),
        dp=SimpleNamespace(size=lambda: 1),
        dp_cp=object(),
    )

    with (
        patch("megatron.bridge.training.eval.prepare_forward_step_func", return_value=MagicMock()),
        patch("megatron.bridge.training.eval.get_model_config", return_value=SimpleNamespace()),
        patch("megatron.bridge.training.eval.get_rerun_state_machine", return_value=rerun_state_machine),
        patch("megatron.bridge.training.eval.is_full_iteration_cuda_graph", return_value=False),
        patch("megatron.bridge.training.eval.get_forward_backward_func", return_value=forward_backward_func),
        patch("megatron.bridge.training.eval.is_pp_last_stage", return_value=False),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_start"),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_end"),
    ):
        evaluate(
            state=state,
            forward_step_func=MagicMock(),
            data_iterator=object(),
            model=[model],
            process_non_loss_data_func=None,
            config=SimpleNamespace(timers=state.timers),
            p2p_communicator=MagicMock(),
            pg_collection=eval_pg_collection,
        )

    assert forward_backward_func.call_args.kwargs["num_microbatches"] == 4


def test_evaluate_preserves_multimodule_data_parallel_accounting():
    """MegatronMIMO collections do not expose one shared DP process group."""

    class MultimodulePGCollection:
        pass

    class MultimoduleCommunicator:
        is_pp_last_stage = False

    state = _make_evaluate_state(eval_iters=1)
    state.cfg.validation.eval_global_batch_size = 4
    state.cfg.validation.eval_micro_batch_size = 1
    state.cfg.data_parallel_size = 2
    model = _ModeTrackingModel()
    forward_backward_func = MagicMock(return_value=[{}])
    rerun_state_machine = MagicMock()

    with (
        patch("megatron.bridge.training.eval.MultiModuleProcessGroupCollection", MultimodulePGCollection),
        patch("megatron.bridge.training.eval.MultiModulePipelineCommunicator", MultimoduleCommunicator),
        patch("megatron.bridge.training.eval.prepare_forward_step_func", return_value=MagicMock()),
        patch("megatron.bridge.training.eval.get_model_config", return_value=SimpleNamespace()),
        patch("megatron.bridge.training.eval.get_rerun_state_machine", return_value=rerun_state_machine),
        patch("megatron.bridge.training.eval.is_full_iteration_cuda_graph", return_value=False),
        patch(
            "megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving",
            forward_backward_func,
        ),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_start"),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_end"),
    ):
        evaluate(
            state=state,
            forward_step_func=MagicMock(),
            data_iterator=object(),
            model=[model],
            process_non_loss_data_func=None,
            config=SimpleNamespace(timers=state.timers),
            p2p_communicator=MultimoduleCommunicator(),
            pg_collection=MultimodulePGCollection(),
        )

    assert forward_backward_func.call_args.kwargs["num_microbatches"] == 2


def test_evaluate_runs_non_loss_collection_on_every_pipeline_rank():
    state = _make_evaluate_state(eval_iters=0)
    model = _ModeTrackingModel()
    forward_backward_func = MagicMock(return_value=[])
    rerun_state_machine = MagicMock()
    pg_collection = SimpleNamespace(
        pp=SimpleNamespace(size=lambda: 2),
        dp=SimpleNamespace(size=lambda: 1),
        dp_cp=object(),
    )

    with (
        patch("megatron.bridge.training.eval.prepare_forward_step_func", return_value=MagicMock()),
        patch("megatron.bridge.training.eval.get_model_config", return_value=SimpleNamespace()),
        patch("megatron.bridge.training.eval.get_rerun_state_machine", return_value=rerun_state_machine),
        patch("megatron.bridge.training.eval.is_full_iteration_cuda_graph", return_value=False),
        patch("megatron.bridge.training.eval.get_forward_backward_func", return_value=forward_backward_func),
        patch("megatron.bridge.training.eval.is_last_rank", return_value=False),
    ):
        result = evaluate(
            state=state,
            forward_step_func=MagicMock(),
            data_iterator=object(),
            model=[model],
            process_non_loss_data_func=MagicMock(),
            config=SimpleNamespace(timers=state.timers),
            p2p_communicator=MagicMock(),
            pg_collection=pg_collection,
        )

    assert result == ({}, [], False)
    forward_backward_func.assert_called_once()
    assert forward_backward_func.call_args.kwargs["collect_non_loss_data"] is True


def _run_evaluate_loss_reduction(loss_dicts, *, eval_iters=1, rank=1, remote=None, empty_loss_keys=None):
    """Drive evaluate() through its loss-aggregation path on the last pipeline stage."""
    state = _make_evaluate_state(eval_iters=eval_iters)
    pg_collection = SimpleNamespace(
        pp=SimpleNamespace(size=lambda: 2),
        dp=SimpleNamespace(size=lambda: 1),
        dp_cp=object(),
    )

    def reduce(value, **kwargs):
        assert kwargs["group"] is pg_collection.dp_cp
        if remote is not None:
            value.add_(value.new_tensor(remote))

    with (
        patch("megatron.bridge.training.eval.prepare_forward_step_func", return_value=MagicMock()),
        patch("megatron.bridge.training.eval.get_pg_collection", return_value=pg_collection),
        patch("megatron.bridge.training.eval.get_model_config", return_value=SimpleNamespace()),
        patch("megatron.bridge.training.eval.get_rerun_state_machine", return_value=MagicMock()),
        patch("megatron.bridge.training.eval.is_full_iteration_cuda_graph", return_value=False),
        patch(
            "megatron.bridge.training.eval.get_forward_backward_func",
            return_value=MagicMock(return_value=loss_dicts),
        ),
        patch("megatron.bridge.training.eval.is_pp_last_stage", return_value=rank == 1),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_start"),
        patch("megatron.bridge.training.eval.fault_tolerance.on_eval_step_end"),
        patch("megatron.bridge.training.eval.torch.distributed.all_reduce", side_effect=reduce),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("builtins.print") as printed,
    ):
        total_loss_dict, _, timelimit = evaluate(
            state=state,
            forward_step_func=MagicMock(),
            data_iterator=object(),
            model=[_ModeTrackingModel()],
            process_non_loss_data_func=None,
            config=SimpleNamespace(timers=state.timers),
            p2p_communicator=MagicMock(),
            callback_manager=CallbackManager(),
            empty_loss_keys=empty_loss_keys,
        )
    assert timelimit is False
    return total_loss_dict, [call.args[0] for call in printed.call_args_list]


def test_evaluate_all_masked_batch_reports_finite_zero():
    """A fully masked evaluation set leaves a zero denominator.

    Dividing by it produced NaN, which then propagated into the validation
    perplexity and every metric writer. Matches the training-side reduction.
    """
    total_loss_dict, messages = _run_evaluate_loss_reduction(
        [{"lm loss": torch.tensor([0.0, 0.0], dtype=torch.float, device="cuda")}]
    )

    value = total_loss_dict["lm loss"]
    assert torch.isfinite(value)
    assert value.item() == 0.0
    # The substituted zero is otherwise indistinguishable from a real loss.
    assert any("no unmasked tokens" in message for message in messages)


def test_evaluate_token_weighted_average_is_unchanged():
    """The guard must not disturb the normal token-weighted average."""
    total_loss_dict, messages = _run_evaluate_loss_reduction(
        [
            {"lm loss": torch.tensor([6.0, 2.0], dtype=torch.float, device="cuda")},
            {"lm loss": torch.tensor([9.0, 1.0], dtype=torch.float, device="cuda")},
        ]
    )

    assert total_loss_dict["lm loss"].item() == pytest.approx(5.0)
    assert not any("no unmasked tokens" in message for message in messages)


def test_evaluate_first_pipeline_stage_does_not_warn():
    total_loss_dict, messages = _run_evaluate_loss_reduction([], rank=0)

    assert total_loss_dict == {}
    assert not any("no unmasked tokens" in message for message in messages)


def test_evaluate_locally_masked_batch_uses_remote_tokens():
    total_loss_dict, messages = _run_evaluate_loss_reduction(
        [{"lm loss": torch.tensor([0.0, 0.0], device="cuda")}], remote=[12.0, 3.0]
    )

    assert total_loss_dict["lm loss"].item() == pytest.approx(4.0)
    assert not any("no unmasked tokens" in message for message in messages)


def test_evaluate_reports_empty_loss_keys():
    """Callers can tell a substituted 0.0 from a measured loss."""
    empty_loss_keys = set()
    _run_evaluate_loss_reduction(
        [
            {
                "lm loss": torch.tensor([0.0, 0.0], device="cuda"),
                "aux loss": torch.tensor([6.0, 2.0], device="cuda"),
            }
        ],
        empty_loss_keys=empty_loss_keys,
    )

    assert empty_loss_keys == {"lm loss"}


@pytest.mark.parametrize(
    ("local_losses", "rank", "remote"),
    [
        ([[6.0, 2.0]], 1, None),
        ([[0.0, 0.0]], 1, [12.0, 3.0]),
        ([], 0, None),
    ],
    ids=["healthy", "locally-masked-with-remote-tokens", "first-pipeline-stage"],
)
def test_evaluate_leaves_empty_loss_keys_unset_when_tokens_exist(local_losses, rank, remote):
    # Tensors are built here rather than in the parametrize list so collection stays CUDA-free.
    loss_dicts = [{"lm loss": torch.tensor(loss, device="cuda")} for loss in local_losses]
    empty_loss_keys = set()
    _run_evaluate_loss_reduction(loss_dicts, rank=rank, remote=remote, empty_loss_keys=empty_loss_keys)

    assert empty_loss_keys == set()


@patch("megatron.bridge.training.eval.is_last_rank", return_value=True)
@patch("megatron.bridge.training.eval.print_rank_last")
@patch("megatron.bridge.training.eval.evaluate")
def test_evaluate_and_print_results_skips_writers_for_empty_eval_set(
    mock_evaluate, mock_print_rank_last, mock_is_last_rank
):
    """An empty evaluation set must not plot as a perfect model (loss 0.0, perplexity 1.0)."""
    losses = {"empty loss": torch.tensor(0.0), "lm loss": torch.tensor(2.0)}

    def fake_evaluate(*args, empty_loss_keys, **kwargs):
        empty_loss_keys.add("empty loss")
        return losses, None, False

    mock_evaluate.side_effect = fake_evaluate
    state = _make_state()
    state.cfg.logger.log_validation_ppl_to_tensorboard = True
    state.tensorboard_logger = MagicMock()
    state.wandb_logger = MagicMock()
    state.mlflow_logger = MagicMock()
    state.comet_logger = MagicMock()

    result = evaluate_and_print_results(
        state=state,
        prefix="iteration 10",
        forward_step_func=MagicMock(),
        data_iterator=object(),
        model=[MagicMock()],
        config=SimpleNamespace(),
    )

    # The returned values and the console line are unchanged.
    assert result is losses
    printed = " ".join(str(call.args[0]) for call in mock_print_rank_last.call_args_list)
    assert "empty loss value" in printed
    assert "lm loss value" in printed

    writer_calls = str(
        state.tensorboard_logger.add_scalar.call_args_list
        + state.wandb_logger.log.call_args_list
        + state.mlflow_logger.log_metrics.call_args_list
        + state.comet_logger.log_metrics.call_args_list
    )
    assert "empty loss" not in writer_calls
    state.tensorboard_logger.add_scalar.assert_any_call("lm loss validation", 2.0, 0)
    state.tensorboard_logger.add_scalar.assert_any_call("lm loss validation ppl", pytest.approx(7.389056), 0)
    state.wandb_logger.log.assert_any_call({"lm loss validation": 2.0}, 0)
    state.mlflow_logger.log_metrics.assert_any_call({"val/lm loss": 2.0}, step=0)
    state.comet_logger.log_metrics.assert_any_call({"lm loss validation": 2.0}, step=0)
