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

"""Unit tests for the MiMo-V2-Flash modeling layer.

These tests target the ``forward`` override on
:class:`MiMoV2FlashTEDotProductAttention` that applies
``attention_value_scale`` to the value tensor before the attention kernel.

The override is exercised without instantiating the TransformerEngine-backed
parent class: we call the unbound method against a mock ``self`` and patch the
parent's ``forward`` to capture what V actually gets passed in. This keeps the
tests CPU-only and avoids any dependency on TransformerEngine.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashTEDotProductAttention,
)


_SUPER_FORWARD_PATH = "megatron.bridge.models.mimo_v2_flash.modeling_mimo_v2_flash.TEDotProductAttention.forward"


def _make_instance(scale):
    """Build a real ``MiMoV2FlashTEDotProductAttention`` instance without running ``__init__``.

    The parent ``TEDotProductAttention`` requires a CUDA build of TransformerEngine
    to initialize, which we don't want to pull into a CPU unit test. ``object.__new__``
    skips ``__init__`` entirely; we then plant the single attribute the override
    reads (``_attention_value_scale``) directly on the instance. ``super()`` inside
    the override still resolves correctly because the instance's class chain is real.
    """
    instance = object.__new__(MiMoV2FlashTEDotProductAttention)
    instance._attention_value_scale = scale
    return instance


def _invoke_forward(scale, value):
    """Call the override and return the V (and kwargs) passed to super().

    Patches the parent ``TEDotProductAttention.forward`` so we never hit TE and
    can inspect exactly what the override forwards upstream.
    """
    instance = _make_instance(scale)

    query = torch.zeros_like(value)
    key = torch.zeros_like(value)
    attention_mask = None
    attn_mask_type = MagicMock()

    captured = {}

    def fake_super_forward(_self, query, key, value, attention_mask, attn_mask_type, **kwargs):
        captured["value"] = value
        captured["kwargs"] = kwargs
        return torch.zeros(1)

    with patch(_SUPER_FORWARD_PATH, autospec=True, side_effect=fake_super_forward) as mock_super:
        out = instance.forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            extra_kwarg="passthrough",
        )

    return captured, mock_super, out


class TestAttentionValueScaleForward:
    """The forward override must multiply V by ``attention_value_scale``.

    Regression coverage for the bug where ``_attention_value_scale`` was read
    from the HF config but silently dropped on the forward path, causing the
    attention output to be off by ~1/scale relative to the HF reference.
    """

    def test_scale_applied_to_value(self):
        scale = 0.707
        v = torch.randn(2, 4, 8, 64)
        captured, _, _ = _invoke_forward(scale, v)
        torch.testing.assert_close(captured["value"], v * scale)

    @pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0])
    def test_scale_various_values(self, scale):
        v = torch.randn(1, 2, 4, 16)
        captured, _, _ = _invoke_forward(scale, v)
        torch.testing.assert_close(captured["value"], v * scale)

    def test_none_scale_passes_value_through_unchanged(self):
        v = torch.randn(2, 4, 8, 64)
        captured, _, _ = _invoke_forward(None, v)
        # When scale is None we expect the exact same tensor object — no
        # allocation, no copy, no scaling. ``is`` is intentional.
        assert captured["value"] is v

    def test_value_not_mutated_in_place(self):
        """Scaling must not mutate the caller's V buffer in place."""
        v = torch.randn(2, 4, 8, 64)
        original = v.clone()
        _invoke_forward(0.5, v)
        torch.testing.assert_close(v, original)

    def test_query_and_key_are_unchanged(self):
        v = torch.randn(2, 4, 8, 64)
        instance = _make_instance(0.707)
        q = torch.randn_like(v)
        k = torch.randn_like(v)
        q_ref = q.clone()
        k_ref = k.clone()

        seen = {}

        def fake_super_forward(_self, query, key, value, *_args, **_kwargs):
            seen["q"] = query
            seen["k"] = key
            return torch.zeros(1)

        with patch(_SUPER_FORWARD_PATH, autospec=True, side_effect=fake_super_forward):
            instance.forward(q, k, v, None, MagicMock())

        # Q and K must reach super() untouched.
        assert seen["q"] is q
        assert seen["k"] is k
        torch.testing.assert_close(q, q_ref)
        torch.testing.assert_close(k, k_ref)

    def test_extra_kwargs_forwarded(self):
        v = torch.randn(1, 2, 4, 16)
        captured, mock_super, _ = _invoke_forward(0.707, v)
        assert captured["kwargs"].get("extra_kwarg") == "passthrough"
        assert mock_super.call_count == 1

    def test_return_value_propagated_from_super(self):
        v = torch.randn(1, 2, 4, 16)
        instance = _make_instance(0.707)
        sentinel = torch.full((3, 3), 42.0)

        with patch(_SUPER_FORWARD_PATH, autospec=True, return_value=sentinel):
            out = instance.forward(torch.zeros_like(v), torch.zeros_like(v), v, None, MagicMock())

        assert out is sentinel
