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

"""Unit tests for _get_qwen3_vl_padding_multiple in qwen3_vl_step.py.

The function computes a sequence-length padding multiple that satisfies:
  - TP/CP alignment (base requirement for all Qwen3-VL runs)
  - FP8 alignment (lcm with 16 when use_fp8_padding=True)
  - HybridEP 128-chunk alignment ((batch_size * target_len) / local_token_divisor % 128 == 0)

These tests exercise representative combinations without requiring GPU or
distributed initialization.
"""

import math

import pytest

from megatron.bridge.models.qwen_vl.qwen3_vl_step import _get_qwen3_vl_padding_multiple


def _hybridep_alignment_ok(batch_size: int, seq_len: int, divisible_by: int, cp_size: int, tp_size: int, sp: bool):
    """Verify the HybridEP 128-chunk invariant holds for a padded sequence length."""
    target_len = math.ceil(seq_len / divisible_by) * divisible_by
    local_divisor = cp_size * (tp_size if sp else 1)
    local_tokens = (batch_size * target_len) // local_divisor
    return local_tokens % 128 == 0


class TestGetQwen3VLPaddingMultiple:
    """Tests for _get_qwen3_vl_padding_multiple."""

    def test_returns_int(self):
        result = _get_qwen3_vl_padding_multiple(
            batch_size=2, tp_size=1, cp_size=1,
            use_fp8_padding=False, use_hybridep=False, sequence_parallel=False,
        )
        assert isinstance(result, int)

    @pytest.mark.parametrize("tp_size", [1, 2, 4])
    def test_baseline_tp_only(self, tp_size):
        d = _get_qwen3_vl_padding_multiple(
            batch_size=2, tp_size=tp_size, cp_size=1,
            use_fp8_padding=False, use_hybridep=False, sequence_parallel=False,
        )
        assert d == tp_size

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_baseline_with_cp(self, cp_size):
        d = _get_qwen3_vl_padding_multiple(
            batch_size=2, tp_size=1, cp_size=cp_size,
            use_fp8_padding=False, use_hybridep=False, sequence_parallel=False,
        )
        assert d == cp_size * 2

    def test_fp8_padding_lcm_with_16(self):
        d = _get_qwen3_vl_padding_multiple(
            batch_size=2, tp_size=1, cp_size=1,
            use_fp8_padding=True, use_hybridep=False, sequence_parallel=False,
        )
        assert d % 16 == 0

    @pytest.mark.parametrize("tp_size", [1, 2, 4])
    def test_fp8_padding_with_tp(self, tp_size):
        d = _get_qwen3_vl_padding_multiple(
            batch_size=2, tp_size=tp_size, cp_size=1,
            use_fp8_padding=True, use_hybridep=False, sequence_parallel=False,
        )
        assert d % tp_size == 0
        assert d % 16 == 0

    @pytest.mark.parametrize(
        "batch_size,tp_size,cp_size,sp",
        [
            (2, 1, 1, False),
            (4, 1, 1, False),
            (2, 2, 1, True),
            (2, 4, 1, True),
            (2, 1, 2, False),
            (2, 2, 2, True),
            (1, 8, 1, True),
            (8, 1, 1, False),
        ],
    )
    def test_hybridep_alignment(self, batch_size, tp_size, cp_size, sp):
        d = _get_qwen3_vl_padding_multiple(
            batch_size=batch_size, tp_size=tp_size, cp_size=cp_size,
            use_fp8_padding=True, use_hybridep=True, sequence_parallel=sp,
        )
        for cur_len in [100, 500, 1000, 2048, 3500, 4096]:
            assert _hybridep_alignment_ok(batch_size, cur_len, d, cp_size, tp_size, sp), (
                f"HybridEP alignment failed: batch_size={batch_size}, cur_len={cur_len}, "
                f"divisible_by={d}, tp={tp_size}, cp={cp_size}, sp={sp}"
            )

    def test_hybridep_reproduces_original_failure_case(self):
        """The original failure: tp=1, ep=8, pp=1, mbs=4, no SP.

        Without the fix, divisible_by=16 (fp8 only), producing target_len
        such that total_tokens=7104 which is not divisible by 128.
        With the fix, divisible_by must ensure 128-alignment.
        """
        d = _get_qwen3_vl_padding_multiple(
            batch_size=4, tp_size=1, cp_size=1,
            use_fp8_padding=True, use_hybridep=True, sequence_parallel=False,
        )
        target_len = math.ceil(1776 / d) * d
        total_tokens = 4 * target_len
        assert total_tokens % 128 == 0, (
            f"Original failure case not fixed: divisible_by={d}, "
            f"target_len={target_len}, total_tokens={total_tokens}"
        )

    def test_hybridep_false_does_not_change_baseline(self):
        d_off = _get_qwen3_vl_padding_multiple(
            batch_size=2, tp_size=1, cp_size=1,
            use_fp8_padding=True, use_hybridep=False, sequence_parallel=False,
        )
        d_baseline = _get_qwen3_vl_padding_multiple(
            batch_size=2, tp_size=1, cp_size=1,
            use_fp8_padding=True, use_hybridep=False, sequence_parallel=True,
        )
        assert d_off == d_baseline

    def test_divisible_by_is_multiple_of_base(self):
        for tp in [1, 2, 4]:
            for cp in [1, 2]:
                base = tp * cp * 2 if cp > 1 else tp
                d = _get_qwen3_vl_padding_multiple(
                    batch_size=2, tp_size=tp, cp_size=cp,
                    use_fp8_padding=True, use_hybridep=True, sequence_parallel=True,
                )
                assert d % base == 0

