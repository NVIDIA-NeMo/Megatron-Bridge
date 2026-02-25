# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from megatron.bridge.diffusion.data.common.diffusion_sample import DiffusionSample


def test_add():
    """Test __add__ method for DiffusionSample."""
    # Create two DiffusionSample instances with different seq_len_q
    sample1 = DiffusionSample(
        __key__="sample1",
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__=["default"],
        video=torch.randn(3, 8, 16, 16),
        context_embeddings=torch.randn(10, 512),
        seq_len_q=torch.tensor(100),
    )
    sample2 = DiffusionSample(
        __key__="sample2",
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__=["default"],
        video=torch.randn(3, 8, 16, 16),
        context_embeddings=torch.randn(10, 512),
        seq_len_q=torch.tensor(200),
    )

    # Test adding two DiffusionSample instances
    result = sample1 + sample2
    assert result == 300, f"Expected 300, got {result}"

    # Test adding DiffusionSample with an integer
    result = sample1 + 50
    assert result == 150, f"Expected 150, got {result}"


def test_radd():
    """Test __radd__ method for DiffusionSample."""
    # Create a DiffusionSample instance
    sample = DiffusionSample(
        __key__="sample",
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__=["default"],
        video=torch.randn(3, 8, 16, 16),
        context_embeddings=torch.randn(10, 512),
        seq_len_q=torch.tensor(100),
    )

    # Test reverse addition with an integer
    result = 50 + sample
    assert result == 150, f"Expected 150, got {result}"

    # Test sum() function which uses __radd__ (starting with 0)
    samples = [
        DiffusionSample(
            __key__="sample1",
            __restore_key__=(),
            __subflavor__=None,
            __subflavors__=["default"],
            video=torch.randn(3, 8, 16, 16),
            context_embeddings=torch.randn(10, 512),
            seq_len_q=torch.tensor(10),
        ),
        DiffusionSample(
            __key__="sample2",
            __restore_key__=(),
            __subflavor__=None,
            __subflavors__=["default"],
            video=torch.randn(3, 8, 16, 16),
            context_embeddings=torch.randn(10, 512),
            seq_len_q=torch.tensor(20),
        ),
        DiffusionSample(
            __key__="sample3",
            __restore_key__=(),
            __subflavor__=None,
            __subflavors__=["default"],
            video=torch.randn(3, 8, 16, 16),
            context_embeddings=torch.randn(10, 512),
            seq_len_q=torch.tensor(30),
        ),
    ]
    result = sum(samples)
    assert result == 60, f"Expected 60, got {result}"


def test_lt():
    """Test __lt__ method for DiffusionSample."""
    # Create two DiffusionSample instances with different seq_len_q
    sample1 = DiffusionSample(
        __key__="sample1",
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__=["default"],
        video=torch.randn(3, 8, 16, 16),
        context_embeddings=torch.randn(10, 512),
        seq_len_q=torch.tensor(100),
    )
    sample2 = DiffusionSample(
        __key__="sample2",
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__=["default"],
        video=torch.randn(3, 8, 16, 16),
        context_embeddings=torch.randn(10, 512),
        seq_len_q=torch.tensor(200),
    )

    # Test comparing two DiffusionSample instances
    assert sample1 < sample2, "Expected sample1 < sample2"
    assert not (sample2 < sample1), "Expected not (sample2 < sample1)"

    # Test comparing DiffusionSample with an integer
    assert sample1 < 150, "Expected sample1 < 150"
    assert not (sample1 < 50), "Expected not (sample1 < 50)"

    # Test sorting a list of DiffusionSample instances
    samples = [sample2, sample1]
    sorted_samples = sorted(samples)
    assert sorted_samples[0].seq_len_q.item() == 100, "Expected first element to have seq_len_q=100"
    assert sorted_samples[1].seq_len_q.item() == 200, "Expected second element to have seq_len_q=200"
