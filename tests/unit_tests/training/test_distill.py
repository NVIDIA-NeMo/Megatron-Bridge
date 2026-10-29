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

"""`distill()` hands its callbacks to `pretrain()`, as `finetune()` does."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from megatron.bridge.models.distillation_provider import DistillationProvider
from megatron.bridge.training.callbacks import Callback
from megatron.bridge.training.distill import distill
from megatron.bridge.training.gpt_step import forward_step_modelopt


def test_distill_passes_callbacks_to_pretrain():
    config = SimpleNamespace(model=Mock(spec=DistillationProvider, context_parallel_size=1))
    callbacks = [Callback()]
    with patch("megatron.bridge.training.distill.pretrain") as mock_pretrain:
        distill(config, callbacks=callbacks)
    mock_pretrain.assert_called_once_with(config, forward_step_modelopt, callbacks=callbacks)
