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
            dataset=SimpleNamespace(dataloader_type="single"),
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
