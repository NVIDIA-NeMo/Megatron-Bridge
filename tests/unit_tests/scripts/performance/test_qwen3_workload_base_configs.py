# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for Qwen3 performance workload presets."""

import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from configs.qwen.qwen3_workload_base_configs import (  # noqa: E402
    QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX_V1,
)


def test_qwen3_30b_gb200_natural_routing_uses_partial_cuda_graphs():
    config = QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX_V1

    assert config.cuda_graph_impl == "transformer_engine"
    assert config.cuda_graph_scope == ["attn", "moe_router", "moe_preprocess"]
