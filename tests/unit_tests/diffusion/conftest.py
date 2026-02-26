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

import pytest
import torch


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    """
    parser.addoption(
        "--with_downloads",
        action="store_true",
        help="pass this argument to active tests which download models from the cloud.",
    )


@pytest.fixture(autouse=True)
def downloads_weights(request):
    """Fixture to validate if the with_downloads flag is passed if necessary"""
    if request.node.get_closest_marker("with_downloads"):
        if not request.config.getoption("--with_downloads"):
            pytest.skip(
                "To run this test, pass --with_downloads option. It will download (and cache) models from cloud."
            )


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
def check_gpu_requirements(request):
    """Fixture to skip tests that require GPU when CUDA is not available"""
    marker = request.node.get_closest_marker("run_only_on")
    if marker and "gpu" in [arg.lower() for arg in marker.args]:
        if not torch.cuda.is_available():
            pytest.skip("Test requires GPU but CUDA is not available")


def pytest_configure(config):
    """
    Initial configuration of conftest.

    Note: DFM uses the following pattern for CPU/GPU test separation:
    Tests don't use markers - GPU visibility is controlled by CUDA_VISIBLE_DEVICES
    in the shell scripts (L0_Unit_Tests_CPU.sh and L0_Unit_Tests_GPU.sh).
    """
    config.addinivalue_line(
        "markers",
        "with_downloads: runs the test using data present in tests/.data",
    )
    config.addinivalue_line(
        "markers",
        "pleasefixme: marks test as needing fixes (will be skipped in CI)",
    )
    config.addinivalue_line(
        "markers",
        "run_only_on: marks test to run only on specific hardware (CPU/GPU)",
    )
