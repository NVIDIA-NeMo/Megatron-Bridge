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

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from megatron.bridge.models.bagel.bagel_step import bagel_loss
from megatron.bridge.models.bagel.data.batch import _attention_metadata
from megatron.bridge.models.bagel.data.external import BagelRNGIterator
from megatron.bridge.models.bagel.diffusion import BagelDiffusionScheduler
from megatron.bridge.models.bagel.provider import BagelModelProvider


class RandomIterator:
    """Expose global RNG draws through a stateful test iterator."""

    def __init__(self) -> None:
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.position += 1
        return random.random(), float(np.random.rand()), float(torch.rand(()))

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def _draw_process_rng() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def test_rng_iterator_is_deterministic_and_does_not_change_process_rng():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    process_states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    expected_process_draw = _draw_process_rng()
    random.setstate(process_states[0])
    np.random.set_state(process_states[1])
    torch.set_rng_state(process_states[2])

    first = BagelRNGIterator(RandomIterator(), 42)
    second = BagelRNGIterator(RandomIterator(), 42)
    assert next(first) == next(second)
    assert _draw_process_rng() == expected_process_draw

    state = first.state_dict()
    expected_suffix = next(first)
    restored = BagelRNGIterator(RandomIterator(), 0)
    restored.load_state_dict(state)
    assert next(restored) == expected_suffix


def test_attention_metadata_preserves_sample_boundaries():
    first = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])
    second = torch.zeros((1, 1))
    allowed, sample_ids, starts, offsets, lengths = _attention_metadata([first, second])
    reconstructed = torch.zeros((3, 3), dtype=torch.bool)
    for query in range(3):
        for key in range(3):
            if sample_ids[query] == sample_ids[key]:
                index = offsets[query] + (query - starts[query]) * lengths[query] + key - starts[query]
                reconstructed[query, key] = allowed[index]
    assert torch.equal(
        reconstructed,
        torch.tensor([[True, False, False], [True, True, False], [False, False, True]]),
    )


def test_diffusion_scheduler_uses_shifted_linear_interpolation():
    scheduler = BagelDiffusionScheduler(
        bagel_repo="/unused",
        vae_path="/unused",
        timestep_shift=2.0,
    )
    timesteps = torch.tensor([0.0, float("-inf")])
    shifted = scheduler.shift_timesteps(timesteps)
    assert torch.allclose(shifted, torch.tensor([2 / 3, 0.0]))

    clean = torch.tensor([[2.0], [3.0]])
    with patch("torch.randn_like", return_value=torch.ones_like(clean)):
        noisy, target = scheduler.add_noise(clean, shifted)
    assert torch.allclose(noisy, torch.tensor([[4 / 3], [3.0]]))
    assert torch.equal(target, torch.tensor([[-1.0]]))


def test_bagel_loss_matches_official_ce_and_mse_normalization():
    with (
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.all_reduce"),
    ):
        loss, tokens, metrics = bagel_loss(
            torch.tensor([2.0, 4.0]),
            loss_mask=torch.tensor([1.0, 0.0, 2.0]),
            mse_loss=torch.tensor([[1.0, 3.0], [0.0, 0.0]]),
            mse_loss_mask=torch.tensor([1.0, 0.0]),
            dp_cp_group=object(),
            ce_weight=1.0,
            mse_weight=2.0,
            ce_loss_reweighting=False,
        )
    assert loss.item() == pytest.approx(7.0)
    assert tokens.item() == 1
    assert torch.equal(metrics["ce"], torch.tensor([6.0, 2.0]))
    assert torch.equal(metrics["mse"], torch.tensor([2.0, 1.0]))


def test_provider_rejects_unvalidated_model_parallel_topology():
    provider = BagelModelProvider(tensor_model_parallel_size=2)
    with pytest.raises(ValueError, match="TP=PP=CP=1"):
        provider.finalize()
