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

from megatron.bridge.training.eval import evaluate_and_print_results


pytestmark = pytest.mark.unit


def _make_state():
    return SimpleNamespace(
        wandb_logger=None,
        mlflow_logger=None,
        comet_logger=None,
        train_state=SimpleNamespace(step=0, consumed_train_samples=0),
        cfg=SimpleNamespace(logger=SimpleNamespace(log_validation_ppl_to_tensorboard=False)),
    )


@patch("megatron.bridge.training.eval.print_rank_last")
@patch("megatron.bridge.training.eval.should_fire", return_value=False)
@patch("megatron.bridge.training.eval.evaluate")
def test_evaluate_and_print_results_returns_loss_dict(mock_evaluate, mock_should_fire, mock_print_rank_last):
    losses = {"lm loss": torch.tensor(1.0)}
    mock_evaluate.return_value = (losses, None, False)

    result = evaluate_and_print_results(
        state=_make_state(),
        prefix="calibration",
        forward_step_func=MagicMock(),
        data_iterator=object(),
        model=[MagicMock()],
        config=SimpleNamespace(),
        write_to_tensorboard=False,
    )

    assert result is losses
    assert mock_should_fire.called
    assert mock_print_rank_last.called


@patch("megatron.bridge.training.eval.print_rank_last")
@patch("megatron.bridge.training.eval.should_fire", return_value=False)
@patch("megatron.bridge.training.eval.evaluate")
def test_evaluate_and_print_results_returns_none_on_timelimit(
    mock_evaluate,
    mock_should_fire,
    mock_print_rank_last,
):
    mock_evaluate.return_value = (None, None, True)

    result = evaluate_and_print_results(
        state=_make_state(),
        prefix="calibration",
        forward_step_func=MagicMock(),
        data_iterator=object(),
        model=[MagicMock()],
        config=SimpleNamespace(),
        write_to_tensorboard=False,
    )

    assert result is None
    assert mock_should_fire.called
    mock_print_rank_last.assert_not_called()
