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
import tempfile

import pytest
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data(tmp_path_factory):
    """Ensure test data is available in a temporary directory by downloading if necessary."""
    data_path = tmp_path_factory.mktemp("test_data")

    # Check if data directory exists and has content
    if not any(data_path.iterdir()):
        logger.info(f"Test data not found at {data_path}. Downloading...")

        try:
            # Download assets to data_path
            from tests.functional_tests.data.download_unit_tests_dataset import get_oldest_release_and_assets

            get_oldest_release_and_assets(assets_dir=str(data_path))

            logger.info("Test data downloaded successfully.")

        except ImportError as e:
            logger.info(f"Failed to import download function: {e}")
        except ValueError as e:
            logger.error(e)
            pytest.exit(f"Failed to download test data: {e}", returncode=1)
            # Don't fail the tests, just warn
        except Exception as e:
            logger.info(f"Failed to download test data: {e}")
            # Don't fail the tests, just warn
    else:
        logger.info(f"Test data already available at {data_path}")

    yield data_path


def pytest_configure(config):
    """Configure pytest markers for functional tests."""
    config.addinivalue_line("markers", "run_only_on(device): run test only on specified device (GPU, CPU)")
    config.addinivalue_line("markers", "pleasefixme: mark test as broken and needs fixing")


def pytest_runtest_setup(item):
    """Setup for each test run - check device requirements."""
    # Check for run_only_on marker
    marker = item.get_closest_marker("run_only_on")
    if marker:
        device = marker.args[0]
        if device == "GPU" and not torch.cuda.is_available():
            pytest.skip(f"Test requires {device} but it's not available")
        elif device == "CPU" and torch.cuda.is_available():
            # Optionally skip CPU tests if GPU is available
            pass


@pytest.fixture(scope="session")
def world_size():
    """Get the world size for distributed tests."""
    return int(os.environ.get("WORLD_SIZE", "1"))


@pytest.fixture(scope="session")
def local_rank():
    """Get the local rank for distributed tests."""
    return int(os.environ.get("LOCAL_RANK", "0"))


@pytest.fixture(scope="session")
def global_rank():
    """Get the global rank for distributed tests."""
    return int(os.environ.get("RANK", "0"))


@pytest.fixture(scope="function")
def tmp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return str(checkpoint_dir)


@pytest.fixture(scope="function")
def tmp_tensorboard_dir(tmp_path):
    """Create a temporary directory for tensorboard logs."""
    tensorboard_dir = tmp_path / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    return str(tensorboard_dir)


@pytest.fixture(scope="function")
def tmp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return str(data_dir)


@pytest.fixture(scope="session")
def shared_tmp_dir():
    """Create a shared temporary directory that persists across test session."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def reset_cuda():
    """Reset CUDA state between tests."""
    yield

    # Clear CUDA cache after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables to prevent test leakage."""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)
