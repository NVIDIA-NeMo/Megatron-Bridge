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
import logging
import os
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

import pytest
from megatron.core.msc_utils import MultiStorageClientFeature


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    """Cleanup local experiments folder"""
    # Asserts in fixture are not recommended, but I'd rather stop users from deleting expensive training runs
    assert not Path("./NeMo_experiments").exists()
    assert not Path("./nemo_experiments").exists()

    yield

    if Path("./NeMo_experiments").exists():
        rmtree("./NeMo_experiments", ignore_errors=True)
    if Path("./nemo_experiments").exists():
        rmtree("./nemo_experiments", ignore_errors=True)


@pytest.fixture(scope="function", autouse=True)
def disable_msc():
    """Disable MSC for the tests."""
    MultiStorageClientFeature.disable()


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear LRU cache before each test to ensure test isolation."""
    # Import the functions that use @lru_cache
    from megatron.bridge.training.utils.checkpoint_utils import read_run_config, read_train_state

    # Clear the cache before each test
    read_run_config.cache_clear()
    read_train_state.cache_clear()

    yield

    # Clear cache after each test as well
    read_run_config.cache_clear()
    read_train_state.cache_clear()


@pytest.fixture
def mock_distributed_environment():
    """Mock torch.distributed environment for testing."""
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
        patch("megatron.bridge.training.utils.checkpoint_utils.get_world_size_safe", return_value=1),
    ):
        yield


@pytest.fixture
def sample_config_data():
    """Provide sample configuration data for testing."""
    return {
        "model": {"type": "gpt", "layers": 24, "hidden_size": 1024, "attention_heads": 16},
        "training": {"learning_rate": 1e-4, "batch_size": 32, "max_steps": 10000, "warmup_steps": 1000},
        "optimizer": {"type": "adam", "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    }


@pytest.fixture
def sample_train_state_data():
    """Provide sample train state data for testing."""
    return {"iteration": 5000, "epoch": 10, "step": 50000, "learning_rate": 0.0001, "loss": 2.34}
