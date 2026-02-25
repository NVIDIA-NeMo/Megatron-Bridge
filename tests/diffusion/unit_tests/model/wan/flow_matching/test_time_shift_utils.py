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

import torch

from megatron.bridge.diffusion.models.wan.flow_matching.time_shift_utils import (
    compute_density_for_timestep_sampling,
    get_flow_match_loss_weight,
    time_shift,
)


def test_time_shift_constant_linear_sqrt_bounds_and_monotonic():
    t_small = torch.tensor(0.1, dtype=torch.float32)
    t_large = torch.tensor(0.9, dtype=torch.float32)
    seq_len = 512

    # constant
    s_small = time_shift(t_small, image_seq_len=seq_len, shift_type="constant", constant=3.0)
    s_large = time_shift(t_large, image_seq_len=seq_len, shift_type="constant", constant=3.0)
    assert 0.0 <= s_small.item() <= 1.0
    assert 0.0 <= s_large.item() <= 1.0
    assert s_large > s_small

    # linear
    s_small = time_shift(t_small, image_seq_len=seq_len, shift_type="linear", base_shift=0.5, max_shift=1.15)
    s_large = time_shift(t_large, image_seq_len=seq_len, shift_type="linear", base_shift=0.5, max_shift=1.15)
    assert 0.0 <= s_small.item() <= 1.0
    assert 0.0 <= s_large.item() <= 1.0
    assert s_large > s_small

    # sqrt
    s_small = time_shift(t_small, image_seq_len=seq_len, shift_type="sqrt")
    s_large = time_shift(t_large, image_seq_len=seq_len, shift_type="sqrt")
    assert 0.0 <= s_small.item() <= 1.0
    assert 0.0 <= s_large.item() <= 1.0
    assert s_large > s_small


def test_compute_density_for_timestep_sampling_modes_and_ranges():
    batch_size = 16
    for mode in ["uniform", "logit_normal", "mode"]:
        u = compute_density_for_timestep_sampling(mode, batch_size=batch_size, logit_mean=0.0, logit_std=1.0)
        assert u.shape == (batch_size,)
        assert torch.all((0.0 <= u) & (u <= 1.0))


def test_get_flow_match_loss_weight_simple_cases():
    sigma = torch.zeros(5, dtype=torch.float32)
    w = get_flow_match_loss_weight(sigma, shift=3.0)
    assert torch.allclose(w, torch.ones_like(w))

    sigma = torch.ones(5, dtype=torch.float32)
    w = get_flow_match_loss_weight(sigma, shift=2.0)
    assert torch.allclose(w, torch.full_like(sigma, 3.0))
