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

"""Strict FP32 setup shared by the DiffusionGemma parity examples."""

import os

import torch


def disable_tf32() -> bool:
    """Disable TF32 in PyTorch and in cuBLAS-backed libraries such as Transformer Engine."""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return tf32_disabled()


def tf32_disabled() -> bool:
    """Return whether TF32 is off for PyTorch matmul, cuDNN and cuBLAS-backed libraries."""
    return (
        os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32
    )
