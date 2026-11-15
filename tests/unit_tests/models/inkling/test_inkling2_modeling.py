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

"""CPU unit tests for the Inkling2 layers' process groups and the model's attention-mask check."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.inkling.modeling_inkling2 import layers, ops
from megatron.bridge.models.inkling.modeling_inkling2.model import Inkling2Model, reject_document_masks


pytestmark = pytest.mark.unit


class _Group:
    """Process group stand-in of a given size."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _causal_mask(batch: int, seq: int) -> torch.Tensor:
    """Megatron-Core's causal mask, in which True hides a key."""
    return torch.ones(seq, seq, dtype=torch.bool).triu(1).expand(batch, 1, seq, seq).clone()


def _document_reset_mask(seq: int, eod: int) -> torch.Tensor:
    """The GPT dataset's mask with ``reset_attention_mask``: tokens after the EOD at ``eod`` see only their document."""
    mask = _causal_mask(1, seq)
    mask[..., eod + 1 :, : eod + 1] = True
    return mask


def test_moe_short_convolution_gathers_and_scatters_over_the_layer_tp_group(monkeypatch):
    calls = []

    def gather(tensor, tensor_parallel_output_grad=True, group=None):
        calls.append(("gather", group, tensor_parallel_output_grad))
        return tensor

    def scatter(tensor, group=None):
        calls.append(("scatter", group))
        return tensor

    monkeypatch.setattr(layers, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(layers, "scatter_to_sequence_parallel_region", scatter)
    tp_group = _Group(2)
    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=True, inkling2_sconv_kernel_size=4, inkling2_compile_ops=False),
        pg_collection=SimpleNamespace(tp=tp_group, cp=_Group(1)),
        mlp_conv_weight=torch.randn(8, 1, 4),
    )
    output = torch.randn(6, 2, 8)

    result = layers.Inkling2MoELayer._mlp_conv(layer, output, None)

    # Megatron-Core would fall back to the global TP group without an explicit group.
    assert calls == [("gather", tp_group, False), ("scatter", tp_group)]
    positions = ops.document_positions(6, 2, output.device, None)
    torch.testing.assert_close(result, ops.static_short_conv(output, layer.mlp_conv_weight.squeeze(1), positions))


@pytest.mark.parametrize(
    "mask",
    [
        None,
        _causal_mask(2, 8),
        # Padding at the end hides keys that no real token sees under causal attention.
        _causal_mask(2, 8).index_fill(-1, torch.arange(6, 8), True),
        # The image-SFT collate's [b, s] padding mask.
        torch.ones(2, 8, dtype=torch.long),
    ],
    ids=["none", "causal", "padding_at_the_end", "padding_mask_2d"],
)
def test_masks_that_causal_attention_applies_are_accepted(mask):
    reject_document_masks(mask, None)


@pytest.mark.parametrize(
    "mask",
    [_document_reset_mask(8, eod=2), _causal_mask(1, 8).index_fill(-1, torch.arange(2), True)],
    ids=["document_reset", "left_padding"],
)
def test_masks_that_hide_earlier_tokens_are_rejected(mask):
    with pytest.raises(ValueError, match="separates documents only through packed sequences"):
        reject_document_masks(mask, None)


def test_packed_batch_separates_documents_through_packed_seq_params():
    reject_document_masks(_document_reset_mask(8, eod=2), packed_seq_params=object())


@pytest.mark.parametrize("another_rank_sees_a_reset", [False, True])
def test_context_parallel_ranks_reach_one_verdict(monkeypatch, another_rank_sees_a_reset):
    cp_group = _Group(2)
    reductions = []

    def all_reduce(tensor, op=None, group=None):
        reductions.append((group, op))
        if another_rank_sees_a_reset:
            tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    # This rank's query rows all precede the boundary, so only later rows, on another rank, see the reset.
    local_rows = _document_reset_mask(8, eod=5)[..., :4, :]

    if another_rank_sees_a_reset:
        with pytest.raises(ValueError, match="packed sequences"):
            reject_document_masks(local_rows, None, cp_group)
    else:
        reject_document_masks(local_rows, None, cp_group)
    assert reductions == [(cp_group, torch.distributed.ReduceOp.MAX)]


def test_model_forward_rejects_a_document_reset_mask():
    model = SimpleNamespace(pg_collection=SimpleNamespace(cp=_Group(1)))
    with pytest.raises(ValueError, match="packed sequences"):
        Inkling2Model.forward(
            model, input_ids=torch.zeros(1, 8, dtype=torch.long), attention_mask=_document_reset_mask(8, eod=2)
        )
