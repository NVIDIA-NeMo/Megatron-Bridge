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

import importlib.util
import os
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.unit

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "models" / "diffusion_gemma"
PARITY_SCRIPTS = ("toy_parity.py", "toy_training_parity.py", "real_encoder_parity.py", "real_fp32_parity.py")


def _load_fp32_precision():
    spec = importlib.util.spec_from_file_location("fp32_precision", EXAMPLE_DIR / "fp32_precision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tf32_state():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    override = os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
    yield
    os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
    if override is not None:
        os.environ["NVIDIA_TF32_OVERRIDE"] = override
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def test_disable_tf32_also_disables_cublas_tf32(tf32_state):
    fp32_precision = _load_fp32_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    assert fp32_precision.disable_tf32() is True
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_torch_flags_alone_do_not_count_as_tf32_disabled(tf32_state):
    fp32_precision = _load_fp32_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    assert fp32_precision.tf32_disabled() is False


@pytest.mark.parametrize("script", PARITY_SCRIPTS)
def test_parity_scripts_disable_tf32_through_the_shared_helper(script):
    source = (EXAMPLE_DIR / script).read_text()

    assert "disable_tf32()" in source
    assert "allow_tf32 = False" not in source
