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

"""Sequence-parallel boundaries of the Shensi stream path, simulated on one process."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.shensi import modeling_shensi
from megatron.bridge.models.shensi.modeling_shensi import ShensiStreamExpansion
from megatron.bridge.models.shensi.shensi_hybrid import ShensiDecoderStack


class _FakeTP:
    """A two-rank TP group viewed from rank 0."""

    def size(self) -> int:
        return 2

    def rank(self) -> int:
        return 0


def test_mask_2d_accepts_the_full_sequence_under_sp() -> None:
    stack = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True), tp_group=_FakeTP())
    embeds = torch.zeros(4, 1, 8)  # this rank's sequence chunk
    mask = torch.ones(1, 8, dtype=torch.bool)  # full sequence
    accepted = ShensiDecoderStack._mask_2d(stack, mask, embeds)
    assert accepted.shape == (1, 8)
    # No mask means every (full-sequence) position is valid.
    default = ShensiDecoderStack._mask_2d(stack, None, embeds)
    assert default.shape == (1, 8)
    with pytest.raises(ValueError, match="does not match"):
        ShensiDecoderStack._mask_2d(stack, torch.ones(1, 6, dtype=torch.bool), embeds)


def test_stream_expansion_conv_reads_the_full_sequence(monkeypatch) -> None:
    cfg = SimpleNamespace(hidden_size=4, hc_conv_kernels=[2], sequence_parallel=True)
    module = ShensiStreamExpansion(cfg)
    conv = module.convs[0]
    with torch.no_grad():
        conv.weight.copy_(torch.ones_like(conv.weight) / conv.kernel_size[0])
        conv.bias.zero_()

    local = torch.arange(3 * 4, dtype=torch.float32).view(3, 1, 4)
    other = local + 100.0
    full = torch.cat([local, other], dim=0)

    monkeypatch.setattr(modeling_shensi, "_tp_group", lambda: _FakeTP())
    monkeypatch.setattr(
        modeling_shensi,
        "gather_from_sequence_parallel_region",
        lambda tensor, group=None: full,
    )
    monkeypatch.setattr(
        modeling_shensi,
        "scatter_to_sequence_parallel_region",
        lambda tensor, group=None: tensor[: local.shape[0]],
    )

    streams = module(local)
    expected = conv(full.permute(1, 2, 0))[..., : full.shape[0]].permute(2, 0, 1)
    torch.testing.assert_close(streams[:, :, 1], expected[: local.shape[0]])
    torch.testing.assert_close(streams[:, :, 0], local)
