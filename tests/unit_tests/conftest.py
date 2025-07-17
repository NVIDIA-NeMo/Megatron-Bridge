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
import os
from pathlib import Path
from shutil import rmtree
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    For now:
        --cpu: use CPU during testing (DEFAULT: GPU)
        --use_local_test_data: use local test data/skip downloading from URL/GitHub (DEFAULT: False)
    """
    parser.addoption(
        "--cpu", action="store_true", help="pass that argument to use CPU during testing (DEFAULT: False = GPU)"
    )
    parser.addoption(
        "--with_downloads",
        action="store_true",
        help="pass this argument to active tests which download models from the cloud.",
    )


@pytest.fixture
def device(request):
    """Simple fixture returning string denoting the device [CPU | GPU]"""
    if request.config.getoption("--cpu"):
        return "CPU"
    else:
        return "GPU"


@pytest.fixture(autouse=True)
def run_only_on_device_fixture(request, device):
    """Fixture to skip tests based on the device"""
    if request.node.get_closest_marker("run_only_on"):
        if request.node.get_closest_marker("run_only_on").args[0] != device:
            pytest.skip("skipped on this device: {}".format(device))


@pytest.fixture(autouse=True)
def downloads_weights(request, device):
    """Fixture to validate if the with_downloads flag is passed if necessary"""
    if request.node.get_closest_marker("with_downloads"):
        if not request.config.getoption("--with_downloads"):
            pytest.skip(
                "To run this test, pass --with_downloads option. It will download (and cache) models from cloud."
            )


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


def pytest_configure(config):
    """
    Initial configuration of conftest.
    The function checks if test_data.tar.gz is present in tests/.data.
    If so, compares its size with github's test_data.tar.gz.
    If file absent or sizes not equal, function downloads the archive from github and unpacks it.
    """
    config.addinivalue_line(
        "markers",
        "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )
    config.addinivalue_line(
        "markers",
        "with_downloads: runs the test using data present in tests/.data",
    )


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
