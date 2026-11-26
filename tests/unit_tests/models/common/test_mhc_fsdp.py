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

import pytest
from megatron.training.models.dist_utils import _ddp_wrap

from megatron.bridge.models.common import unimodal


@pytest.mark.unit
def test_mhc_wrapping_uses_core_entry_point():
    """Core owns mHC unit collection, expert sharding, and replay boundaries."""
    assert unimodal._ddp_wrap is _ddp_wrap
