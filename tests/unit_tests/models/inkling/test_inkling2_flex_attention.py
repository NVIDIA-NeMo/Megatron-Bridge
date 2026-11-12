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

"""Unit tests for the FlexAttention paths of the Inkling2 relative-sink attention (CUDA).

The FlexAttention paths apply the sink after the kernel, through the log-normalizer; the
eager reference keeps the sink as one more softmax logit. Both must agree on the output
and on every gradient, including those of the relative logits and the sink.
"""

import pytest
import torch

from megatron.bridge.models.inkling.modeling_inkling2 import ops
from megatron.bridge.models.inkling.modeling_inkling2.attention import (
    _eager_relative_sink_attention,
    context_parallel_relative_sink_attention,
    relative_sink_attention,
)
from megatron.bridge.models.inkling.modeling_inkling2.context_parallel import context_parallel_layout


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention runs on CUDA"),
]
CUDA = torch.device("cuda")
HEADS, KV_HEADS, DIM, EXTENT, TOTAL = 4, 2, 64, 128, 512
# Packed documents whose lengths divide by twice the CP size; the 256-token document is longer
# than the relative-logit extent, so it has keys both inside and beyond the learned distances.
PACKED = [0, 128, 384, 512]
# BF16 inputs and kernel outputs against an FP32 reference.
TOLERANCE = 2e-2


class _Group:
    """Stand-in for a CP process group; layouts read only its size and rank."""

    def __init__(self, size: int, rank: int) -> None:
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def _inputs(seed: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device=CUDA).manual_seed(seed)

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=CUDA) * scale

    tensors = [
        randn(TOTAL, 1, HEADS, DIM, scale=0.5).bfloat16(),
        randn(TOTAL, 1, KV_HEADS, DIM, scale=0.5).bfloat16(),
        randn(TOTAL, 1, KV_HEADS, DIM).bfloat16(),
        randn(1, HEADS, TOTAL, EXTENT).bfloat16(),
        randn(1, HEADS, TOTAL, scale=2.0),
    ]
    return tensors, randn(TOTAL, 1, HEADS, DIM)


def _run(attention, tensors: list[torch.Tensor], grad: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    leaves = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    output = attention(*leaves)
    output.float().backward(grad)
    return output.detach().float(), [leaf.grad.float() for leaf in leaves]


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    error = float((actual - expected).norm() / expected.norm())
    assert error < TOLERANCE, f"{name}: relative error {error:.2e}"


def _assert_matches(actual, expected) -> None:
    (out, grads), (ref_out, ref_grads) = actual, expected
    _assert_close(out, ref_out, "output")
    for name, grad, ref_grad in zip(("query", "key", "value", "rel_logits", "sink"), grads, ref_grads):
        _assert_close(grad, ref_grad, f"{name} gradient")


@pytest.mark.parametrize("window", [None, EXTENT])
@pytest.mark.parametrize("packed", [None, PACKED])
def test_flex_attention_matches_the_eager_reference(window, packed):
    cu_seqlens = None if packed is None else torch.tensor(packed, dtype=torch.int32, device=CUDA)
    tensors, grad = _inputs(seed=1)
    docs = ops.document_ids(TOTAL, CUDA, cu_seqlens)

    def flex(query, key, value, rel_logits, sink):
        return relative_sink_attention(
            query, key, value, rel_logits, sink, window=window, cu_seqlens=cu_seqlens, out_dtype=torch.float32
        )

    def eager(query, key, value, rel_logits, sink):
        return _eager_relative_sink_attention(
            query, key, value, rel_logits, sink, window=window, docs=docs, out_dtype=torch.float32
        )

    _assert_matches(_run(flex, tensors, grad), _run(eager, tensors, grad))


@pytest.mark.parametrize("window", [None, EXTENT])
@pytest.mark.parametrize("packed", [None, PACKED])
def test_context_parallel_flex_attention_matches_the_eager_reference(window, packed):
    cp_size = 2
    cu_seqlens = None if packed is None else torch.tensor(packed, dtype=torch.int32, device=CUDA)
    tensors, grad = _inputs(seed=2)
    docs = ops.document_ids(TOTAL, CUDA, cu_seqlens)
    groups = [_Group(cp_size, rank) for rank in range(cp_size)]
    layouts = [
        context_parallel_layout(
            group, local_tokens=TOTAL // cp_size, batch=1, device=CUDA, width=4, cu_seqlens=cu_seqlens
        )
        for group in groups
    ]
    for layout in layouts:
        index = layout.local_index

        def flex(query, key, value, rel_logits, sink):
            return context_parallel_relative_sink_attention(
                query.index_select(0, index),
                key,
                value,
                rel_logits.index_select(2, index),
                sink.index_select(2, index),
                window=window,
                layout=layout,
                out_dtype=torch.float32,
            )

        def eager(query, key, value, rel_logits, sink):
            full = _eager_relative_sink_attention(
                query, key, value, rel_logits, sink, window=window, docs=docs, out_dtype=torch.float32
            )
            return full.index_select(0, index)

        local_grad = grad.index_select(0, index)
        _assert_matches(_run(flex, tensors, local_grad), _run(eager, tensors, local_grad))


def test_output_dtype_defaults_to_the_query_dtype():
    tensors, _ = _inputs(seed=3)
    default = relative_sink_attention(*tensors, window=EXTENT)
    fp32 = relative_sink_attention(*tensors, window=EXTENT, out_dtype=torch.float32)
    assert default.dtype == torch.bfloat16
    assert fp32.dtype == torch.float32
    _assert_close(default.float(), fp32, "bf16 output")
