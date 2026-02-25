# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Override parent conftest fixtures that require the full megatron.bridge install.

Tests in this directory use direct file imports (importlib) to avoid the heavy
megatron.bridge.__init__ chain, so the parent conftest's autouse fixtures
(which import from megatron.bridge) must be replaced with no-ops.
"""

import pytest


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """No-op override: vocab_slice tests don't use checkpoint_utils caching."""
    yield
