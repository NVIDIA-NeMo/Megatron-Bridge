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

"""Keep Megatron-Core's process-wide MoE metrics tracker isolated per test."""

import pytest


@pytest.fixture(autouse=True)
def _fresh_moe_metrics():
    """Every test gets an empty router-metrics tracker.

    The router load-balancing loss is enabled by default (training-side default), so each test's
    routers record metrics into the process-wide tracker; models with different layer counts would
    otherwise grow its per-layer tensors past their size and fail on the next test.
    """
    from megatron.core.transformer.moe import moe_logging

    saved = moe_logging.get_moe_metrics_tracker()
    moe_logging.set_moe_metrics_tracker(moe_logging.MoEMetricsTracker())
    yield
    moe_logging.set_moe_metrics_tracker(saved)
