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

"""`distill()` must reject the GPT step for models that shard the sequence themselves."""

from types import SimpleNamespace

import pytest

from megatron.bridge.training.distill import _check_step_handles_context_parallelism
from megatron.bridge.training.gpt_step import forward_step_modelopt


def _config(
    *, position_embedding_type: str, context_parallel_size: int, distill_submodule: str | None = None
) -> SimpleNamespace:
    model = SimpleNamespace(
        position_embedding_type=position_embedding_type,
        context_parallel_size=context_parallel_size,
        distill_submodule=distill_submodule,
    )
    return SimpleNamespace(model=model)


def _other_step(*args, **kwargs):
    raise AssertionError("never called")


def test_gpt_step_rejected_for_a_whole_mrope_vlm_under_context_parallelism():
    with pytest.raises(ValueError, match="already-sharded batch"):
        _check_step_handles_context_parallelism(
            _config(position_embedding_type="mrope", context_parallel_size=2), forward_step_modelopt
        )


@pytest.mark.parametrize(
    ("position_embedding_type", "context_parallel_size", "forward_step_func"),
    [
        pytest.param("mrope", 1, forward_step_modelopt, id="mrope-without-cp"),
        pytest.param("rope", 2, forward_step_modelopt, id="rope-with-cp"),
        pytest.param("mrope", 2, _other_step, id="mrope-with-cp-and-a-model-specific-step"),
    ],
)
def test_allowed_combinations(position_embedding_type, context_parallel_size, forward_step_func):
    _check_step_handles_context_parallelism(
        _config(position_embedding_type=position_embedding_type, context_parallel_size=context_parallel_size),
        forward_step_func,
    )


def test_language_submodule_distillation_keeps_the_gpt_step():
    """QAD on a VLM distills only the language model, whose forward never re-shards the sequence."""
    _check_step_handles_context_parallelism(
        _config(position_embedding_type="mrope", context_parallel_size=2, distill_submodule="language_model"),
        forward_step_modelopt,
    )
