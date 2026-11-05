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

"""The Qwen3-VL step must attach the KD loss when distilling, or training silently skips KD."""

from functools import partial
from unittest.mock import Mock, patch

import modelopt.torch.distill as mtd
import torch

from megatron.bridge.models.qwen_vl import qwen3_vl_step
from megatron.bridge.training.post_training.distillation import create_kd_loss_function


_LOSS_MASK = torch.tensor([[1.0, 1.0, 0.0]])


def _loss_function_from(forward_step, model) -> partial:
    """Run ``forward_step`` far enough to capture the loss function it builds."""
    with (
        patch.object(qwen3_vl_step, "_forward_step_common", side_effect=lambda *a: a[-1]) as common,
        patch("megatron.bridge.training.post_training.distillation.unwrap_model", return_value=model),
    ):
        factory = forward_step(Mock(), iter([]), model)
        common.assert_called_once()
    return factory(_LOSS_MASK, model, False, False)


def test_modelopt_step_attaches_kd_loss_when_distilling():
    """The regression that matters: a step without this trains with plain CE and no warning."""
    loss_func = _loss_function_from(qwen3_vl_step.forward_step_modelopt, Mock(spec=mtd.DistillationModel))

    assert loss_func.func.__name__ == "loss_func_kd"
    assert loss_func.keywords["original_loss_fn"].func.__name__ == "masked_next_token_loss"


def test_modelopt_step_falls_back_to_plain_loss_without_a_distillation_model():
    loss_func = _loss_function_from(qwen3_vl_step.forward_step_modelopt, Mock())

    assert loss_func.func.__name__ == "masked_next_token_loss"


def test_plain_step_never_attaches_kd_loss():
    """`forward_step` stays pure CE even when handed a distilling model."""
    loss_func = _loss_function_from(qwen3_vl_step.forward_step, Mock(spec=mtd.DistillationModel))

    assert loss_func.func.__name__ == "masked_next_token_loss"


def test_kd_loss_function_carries_the_configured_checks():
    with patch(
        "megatron.bridge.training.post_training.distillation.unwrap_model",
        return_value=Mock(spec=mtd.DistillationModel),
    ):
        loss_func = create_kd_loss_function(_LOSS_MASK, Mock(), check_for_nan_in_loss=True, check_for_spiky_loss=False)

    original = loss_func.keywords["original_loss_fn"]
    assert original.keywords["check_for_nan_in_loss"] is True
    assert original.keywords["check_for_spiky_loss"] is False
    assert torch.equal(loss_func.keywords["loss_mask"], _LOSS_MASK)
