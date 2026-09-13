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

"""Unit tests for NemotronH 56B 256-GPU performance recipes."""

import pytest

from megatron.bridge.perf_recipes.nemotronh.b200.nemotronh import (
    nemotronh_56b_pretrain_256gpu_b200_bf16_config,
    nemotronh_56b_pretrain_256gpu_b200_fp8cs_config,
)
from megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh import (
    nemotronh_56b_pretrain_256gpu_gb300_bf16_config,
    nemotronh_56b_pretrain_256gpu_gb300_fp8cs_config,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("bf16_factory", "fp8cs_factory"),
    [
        (nemotronh_56b_pretrain_256gpu_b200_bf16_config, nemotronh_56b_pretrain_256gpu_b200_fp8cs_config),
        (nemotronh_56b_pretrain_256gpu_gb300_bf16_config, nemotronh_56b_pretrain_256gpu_gb300_fp8cs_config),
    ],
)
def test_nemotronh_56b_256gpu_bf16_global_batch_size(bf16_factory, fp8cs_factory):
    """256-GPU BF16 configs rescale global_batch_size to 768, matching FP8-CS and satisfying DP divisibility."""
    bf16_cfg = bf16_factory()
    fp8cs_cfg = fp8cs_factory()

    assert bf16_cfg.train.global_batch_size == 768
    assert bf16_cfg.train.global_batch_size == fp8cs_cfg.train.global_batch_size

    tp = bf16_cfg.model.tensor_model_parallel_size
    pp = bf16_cfg.model.pipeline_model_parallel_size
    cp = bf16_cfg.model.context_parallel_size
    dp = 256 // (tp * pp * cp)
    mbs = bf16_cfg.train.micro_batch_size

    assert dp == 128
    assert bf16_cfg.train.global_batch_size % (mbs * dp) == 0
