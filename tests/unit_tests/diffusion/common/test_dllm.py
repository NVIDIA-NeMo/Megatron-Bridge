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

"""Unit tests for the shared block-diffusion sampling primitives in ``dllm``.

These primitives are model-agnostic and used by every dLLM generation loop
(NemotronLabsDiffusion, LLaDA1.5). All tests run on CPU with plain tensors —
no model, no checkpoint, no GPU.
"""

import pytest
import torch

from megatron.bridge.diffusion.common.dllm import (
    add_gumbel_noise,
    compute_block_bias,
    get_num_transfer_tokens,
    get_transfer_index,
)


pytestmark = [pytest.mark.unit]


class TestComputeBlockBias:
    def test_shape_and_dtype(self):
        n = 4
        bias = compute_block_bias(block_size=2, max_seq_length=n, dtype=torch.float32, device="cpu")
        assert bias.shape == (1, 1, 2 * n, 2 * n)
        assert bias.dtype == torch.float32

    def test_allowed_zero_disallowed_neg_inf(self):
        bias = compute_block_bias(block_size=2, max_seq_length=4, dtype=torch.float32, device="cpu")
        vals = torch.unique(bias)
        # exactly two levels: 0 (allowed) and finfo.min (disallowed)
        assert torch.tensor(0.0) in vals
        assert torch.finfo(torch.float32).min in vals
        assert set(v.item() for v in vals) <= {0.0, torch.finfo(torch.float32).min}

    def test_no_fully_masked_query_row(self):
        # every query row must have at least one allowed (0) key -- else softmax is NaN
        bias = compute_block_bias(block_size=2, max_seq_length=4, dtype=torch.float32, device="cpu")[0, 0]
        allowed = bias == 0.0
        assert bool(allowed.any(dim=1).all())

    def test_x0_half_is_causal(self):
        # bottom-right n×n quadrant (x0 attending x0) must be lower-triangular allowed
        n = 4
        bias = compute_block_bias(block_size=2, max_seq_length=n, dtype=torch.float32, device="cpu")[0, 0]
        x0 = bias[n:, n:] == 0.0
        assert torch.equal(x0, torch.tril(torch.ones(n, n, dtype=torch.bool)))


class TestAddGumbelNoise:
    def test_zero_temperature_is_identity(self):
        logits = torch.randn(2, 4, 10)
        assert torch.equal(add_gumbel_noise(logits, temperature=0), logits)

    def test_nonzero_temperature_changes_values(self):
        logits = torch.randn(2, 4, 10)
        out = add_gumbel_noise(logits, temperature=1.0)
        assert not torch.equal(out, logits)

    def test_nonzero_temperature_preserves_shape(self):
        logits = torch.randn(3, 5, 7)
        assert add_gumbel_noise(logits, temperature=0.5).shape == logits.shape

    def test_nonzero_temperature_output_is_positive(self):
        logits = torch.randn(2, 8, 32)
        result = add_gumbel_noise(logits, temperature=1.0)
        assert (result > 0).all()

    def test_nonzero_temperature_can_change_argmax(self):
        torch.manual_seed(1)
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, 0] = 10.0
        results = [torch.argmax(add_gumbel_noise(logits.clone(), temperature=2.0)).item() for _ in range(20)]
        assert len(set(results)) > 1


class TestGetNumTransferTokens:
    def test_evenly_divisible(self):
        mask = torch.ones(1, 10, dtype=torch.bool)
        n = get_num_transfer_tokens(mask, steps=5)
        assert n.shape == (1, 5)
        assert (n == 2).all()

    def test_total_equals_mask_count(self):
        mask = torch.ones(2, 7, dtype=torch.bool)
        n = get_num_transfer_tokens(mask, steps=3)
        assert n.sum(dim=1).tolist() == [7, 7]

    def test_remainder_distributed_to_first_steps(self):
        mask = torch.ones(1, 7, dtype=torch.bool)
        n = get_num_transfer_tokens(mask, steps=3)
        # 7 = 3 + 2 + 2; remainder (1) added to the first step
        assert n[0].tolist() == [3, 2, 2]

    def test_respects_partial_mask(self):
        # Only 4 of 10 positions masked -> total transfers must equal 4.
        mask = torch.tensor([[True, True, False, True, False, True, False, False, False, False]])
        n = get_num_transfer_tokens(mask, steps=2)
        assert int(n.sum()) == 4

    def test_single_step(self):
        mask = torch.ones(1, 6, dtype=torch.bool)
        n = get_num_transfer_tokens(mask, steps=1)
        assert n.shape == (1, 1)
        assert int(n[0, 0]) == 6


class TestGetTransferIndex:
    def _setup(self, batch=2, seq=6, vocab=20):
        logits = torch.randn(batch, seq, vocab)
        x = torch.zeros(batch, seq, dtype=torch.long)
        # First three positions masked in row 0, first two in row 1.
        mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
            dtype=torch.bool,
        )
        return logits, x, mask

    def test_output_shapes(self):
        logits, x, mask = self._setup()
        nt = torch.tensor([2, 1])
        x0, ti = get_transfer_index(logits, 0.0, "low_confidence", mask, x, nt)
        assert x0.shape == x.shape
        assert ti.shape == x.shape
        assert ti.dtype == torch.bool

    def test_exactly_n_transferred_per_row(self):
        logits, x, mask = self._setup()
        nt = torch.tensor([2, 1])
        _, ti = get_transfer_index(logits, 0.0, "low_confidence", mask, x, nt)
        assert int(ti[0].sum()) == 2
        assert int(ti[1].sum()) == 1

    def test_transfer_only_from_masked_positions(self):
        logits, x, mask = self._setup()
        nt = torch.tensor([2, 1])
        _, ti = get_transfer_index(logits, 0.0, "low_confidence", mask, x, nt)
        assert int((ti & ~mask).sum()) == 0

    def test_unmasked_positions_keep_original_x(self):
        logits, x, mask = self._setup()
        x = x + 7  # distinct sentinel for unmasked positions
        nt = torch.tensor([2, 1])
        x0, _ = get_transfer_index(logits, 0.0, "low_confidence", mask, x, nt)
        assert (x0[~mask] == 7).all()

    def test_random_remasking_runs(self):
        logits, x, mask = self._setup()
        nt = torch.tensor([1, 1])
        _, ti = get_transfer_index(logits, 0.0, "random", mask, x, nt)
        assert ti.dtype == torch.bool

    def test_neg_entropy_runs(self):
        logits, x, mask = self._setup()
        nt = torch.tensor([1, 1])
        _, ti = get_transfer_index(logits, 0.0, "low_confidence", mask, x, nt, neg_entropy=True)
        assert int(ti[0].sum()) == 1

    def test_invalid_remasking_raises(self):
        logits, x, mask = self._setup()
        nt = torch.tensor([1, 1])
        with pytest.raises(NotImplementedError):
            get_transfer_index(logits, 0.0, "bogus", mask, x, nt)

    def test_threshold_overrides_count(self):
        # With a threshold, all masked positions above it transfer regardless of nt.
        logits, x, mask = self._setup()
        nt = torch.tensor([1, 1])
        # threshold below any softmax prob => all masked positions transfer
        _, ti = get_transfer_index(logits, 0.0, "low_confidence", mask, x, nt, threshold=-1.0)
        assert int(ti[0].sum()) == int(mask[0].sum())
