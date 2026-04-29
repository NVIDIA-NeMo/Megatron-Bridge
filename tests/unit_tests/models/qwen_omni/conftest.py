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

import pytest


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Preserve cache clearing when available, otherwise degrade to a no-op.

    The repository-wide fixture imports training config modules that are
    currently incompatible with the local Megatron-Core version in this
    environment. When those imports work, keep the original cache-clearing
    behavior for isolation.
    """
    try:
        from megatron.bridge.training.utils.checkpoint_utils import read_run_config, read_train_state
    except (ImportError, ModuleNotFoundError):
        yield
        return

    read_run_config.cache_clear()
    read_train_state.cache_clear()

    yield

    read_run_config.cache_clear()
    read_train_state.cache_clear()
