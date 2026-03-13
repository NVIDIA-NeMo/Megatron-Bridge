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

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.diffusion.models.common.dit_attention import (
    DiTCrossAttention,
    DiTCrossAttentionSubmodules,
    DiTSelfAttention,
)


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    hidden_size=64,
    num_attention_heads=4,
    num_query_groups=4,
    layernorm_across_heads=False,
    test_mode=False,
    layernorm_epsilon=1e-6,
    add_bias_linear=False,
    crossattn_emb_size=None,
):
    """Build a lightweight mock TransformerConfig with the attributes needed by DiT attention."""
    cfg = MagicMock()
    cfg.hidden_size = hidden_size
    cfg.kv_channels = hidden_size // num_attention_heads
    cfg.num_attention_heads = num_attention_heads
    cfg.num_query_groups = num_query_groups
    cfg.layernorm_across_heads = layernorm_across_heads
    cfg.test_mode = test_mode
    cfg.layernorm_epsilon = layernorm_epsilon
    cfg.normalization = "LayerNorm"
    cfg.add_bias_linear = add_bias_linear
    cfg.init_method = MagicMock()
    if crossattn_emb_size is not None:
        cfg.crossattn_emb_size = crossattn_emb_size
    else:
        del cfg.crossattn_emb_size
    return cfg


def _make_self_attn_submodules(q_layernorm=None, k_layernorm=None):
    """Build SelfAttentionSubmodules with required positional args stubbed."""
    sub = SelfAttentionSubmodules(
        linear_qkv=MagicMock(),
        core_attention=MagicMock(),
    )
    sub.q_layernorm = q_layernorm
    sub.k_layernorm = k_layernorm
    return sub


def _init_module_and_attrs(obj, config, hidden_size_per_head=16, query_proj_size=64, kv_proj_size=64):
    """Initialize nn.Module internals on obj, then set attributes the child __init__ expects."""
    nn.Module.__init__(obj)
    obj.config = config
    obj.hidden_size_per_attention_head = hidden_size_per_head
    obj.query_projection_size = query_proj_size
    obj.kv_projection_size = kv_proj_size


def _new_with_module_init(cls):
    """Create an instance via __new__ and initialize nn.Module internals only."""
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    return obj


# ---------------------------------------------------------------------------
# DiTCrossAttentionSubmodules
# ---------------------------------------------------------------------------


class TestDiTCrossAttentionSubmodules:
    """Tests for the DiTCrossAttentionSubmodules dataclass."""

    def test_all_fields_default_to_none(self):
        sub = DiTCrossAttentionSubmodules()
        for f in fields(sub):
            assert getattr(sub, f.name) is None

    def test_field_names(self):
        expected = {"linear_q", "linear_kv", "core_attention", "linear_proj", "q_layernorm", "k_layernorm"}
        actual = {f.name for f in fields(DiTCrossAttentionSubmodules)}
        assert actual == expected

    def test_custom_values(self):
        sentinel_q = MagicMock()
        sentinel_kv = MagicMock()
        sub = DiTCrossAttentionSubmodules(linear_q=sentinel_q, linear_kv=sentinel_kv)
        assert sub.linear_q is sentinel_q
        assert sub.linear_kv is sentinel_kv
        assert sub.core_attention is None


# ---------------------------------------------------------------------------
# DiTSelfAttention
# ---------------------------------------------------------------------------


class TestDiTSelfAttentionInit:
    """Tests for DiTSelfAttention __init__ (layernorm overrides)."""

    def _run_init(self, config, submodules, mock_super_init):
        """Construct a DiTSelfAttention, mocking the parent __init__ to set up nn.Module."""
        attn = DiTSelfAttention.__new__(DiTSelfAttention)
        _init_module_and_attrs(attn, config)
        mock_super_init.side_effect = lambda *a, **kw: None
        DiTSelfAttention.__init__(attn, config, submodules, layer_number=1, attn_mask_type=AttnMaskType.no_mask)
        return attn

    @patch("megatron.bridge.diffusion.models.common.dit_attention.SelfAttention.__init__", return_value=None)
    @patch(
        "megatron.bridge.diffusion.models.common.dit_attention.build_module",
        side_effect=lambda *a, **kw: nn.Identity(),
    )
    def test_layernorms_built_when_submodules_present(self, mock_build, mock_super_init):
        config = _make_config(layernorm_across_heads=False)
        submodules = _make_self_attn_submodules(q_layernorm=MagicMock(), k_layernorm=MagicMock())

        attn = self._run_init(config, submodules, mock_super_init)

        assert attn.q_layernorm is not None
        assert attn.k_layernorm is not None
        assert mock_build.call_count == 2

    @patch("megatron.bridge.diffusion.models.common.dit_attention.SelfAttention.__init__", return_value=None)
    def test_layernorms_none_when_submodules_absent(self, mock_super_init):
        config = _make_config()
        submodules = _make_self_attn_submodules(q_layernorm=None, k_layernorm=None)

        attn = self._run_init(config, submodules, mock_super_init)

        assert attn.q_layernorm is None
        assert attn.k_layernorm is None

    @patch("megatron.bridge.diffusion.models.common.dit_attention.SelfAttention.__init__", return_value=None)
    @patch(
        "megatron.bridge.diffusion.models.common.dit_attention.build_module",
        side_effect=lambda *a, **kw: nn.Identity(),
    )
    def test_layernorm_across_heads_uses_full_projection_size(self, mock_build, mock_super_init):
        config = _make_config(layernorm_across_heads=True)
        submodules = _make_self_attn_submodules(q_layernorm=MagicMock(), k_layernorm=MagicMock())

        attn = self._run_init(config, submodules, mock_super_init)

        assert attn.layernorm_across_heads is True
        q_call_kwargs = mock_build.call_args_list[0].kwargs
        assert q_call_kwargs["hidden_size"] == 64  # query_projection_size
        k_call_kwargs = mock_build.call_args_list[1].kwargs
        assert k_call_kwargs["hidden_size"] == 64  # kv_projection_size

    @patch("megatron.bridge.diffusion.models.common.dit_attention.SelfAttention.__init__", return_value=None)
    @patch(
        "megatron.bridge.diffusion.models.common.dit_attention.build_module",
        side_effect=lambda *a, **kw: nn.Identity(),
    )
    def test_layernorm_per_head_uses_head_size(self, mock_build, mock_super_init):
        config = _make_config(layernorm_across_heads=False)
        submodules = _make_self_attn_submodules(q_layernorm=MagicMock(), k_layernorm=MagicMock())

        attn = self._run_init(config, submodules, mock_super_init)

        q_call_kwargs = mock_build.call_args_list[0].kwargs
        assert q_call_kwargs["hidden_size"] == 16  # hidden_size_per_attention_head
        k_call_kwargs = mock_build.call_args_list[1].kwargs
        assert k_call_kwargs["hidden_size"] == 16


class TestDiTSelfAttentionGetQKV:
    """Tests for DiTSelfAttention.get_query_key_value_tensors (TP=1)."""

    def _make_attn(self, hidden_size=64, num_heads=4, num_groups=4, with_layernorm=False, across_heads=False):
        """Build a DiTSelfAttention instance with mocked internals for TP=1."""
        head_dim = hidden_size // num_heads
        heads_per_group = num_heads // num_groups

        attn = _new_with_module_init(DiTSelfAttention)
        attn.config = _make_config(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_query_groups=num_groups,
            layernorm_across_heads=across_heads,
            test_mode=False,
        )
        attn.num_query_groups_per_partition = num_groups
        attn.num_attention_heads_per_partition = num_heads
        attn.hidden_size_per_attention_head = head_dim
        attn.query_projection_size = hidden_size
        attn.kv_projection_size = num_groups * head_dim
        attn.layernorm_across_heads = across_heads

        qkv_dim = num_groups * (heads_per_group + 2) * head_dim
        linear_qkv = MagicMock()
        linear_qkv.side_effect = lambda x: (torch.randn(*x.shape[:-1], qkv_dim), None)
        attn.linear_qkv = linear_qkv

        if with_layernorm:
            attn.q_layernorm = nn.Identity()
            attn.k_layernorm = nn.Identity()
        else:
            attn.q_layernorm = None
            attn.k_layernorm = None

        return attn

    @patch("megatron.bridge.diffusion.models.common.dit_attention.parallel_state")
    def test_output_shapes(self, mock_ps):
        mock_ps.get_tensor_model_parallel_world_size.return_value = 1
        attn = self._make_attn(hidden_size=64, num_heads=4, num_groups=4)
        seq_len, batch = 10, 2
        hidden = torch.randn(seq_len, batch, 64)
        q, k, v = attn.get_query_key_value_tensors(hidden)

        assert q.shape == (seq_len, batch, 4, 16)
        assert k.shape == (seq_len, batch, 4, 16)
        assert v.shape == (seq_len, batch, 4, 16)

    @patch("megatron.bridge.diffusion.models.common.dit_attention.parallel_state")
    def test_with_layernorm_per_head(self, mock_ps):
        mock_ps.get_tensor_model_parallel_world_size.return_value = 1
        spy_q = MagicMock(side_effect=lambda x: x)
        spy_k = MagicMock(side_effect=lambda x: x)

        attn = self._make_attn(hidden_size=64, num_heads=4, with_layernorm=False, across_heads=False)
        # Assign non-Module spy mocks via object.__setattr__ isn't needed since MagicMock isn't nn.Module
        attn.q_layernorm = spy_q
        attn.k_layernorm = spy_k

        q, k, v = attn.get_query_key_value_tensors(torch.randn(8, 2, 64))
        spy_q.assert_called_once()
        spy_k.assert_called_once()

    @patch("megatron.bridge.diffusion.models.common.dit_attention.parallel_state")
    def test_without_layernorm(self, mock_ps):
        mock_ps.get_tensor_model_parallel_world_size.return_value = 1
        attn = self._make_attn(hidden_size=64, num_heads=4, with_layernorm=False)
        q, k, v = attn.get_query_key_value_tensors(torch.randn(8, 2, 64))
        assert q is not None and k is not None and v is not None

    @patch("megatron.bridge.diffusion.models.common.dit_attention.parallel_state")
    def test_gqa_shapes(self, mock_ps):
        """Group Query Attention: fewer KV groups than query heads."""
        mock_ps.get_tensor_model_parallel_world_size.return_value = 1
        attn = self._make_attn(hidden_size=64, num_heads=4, num_groups=2)
        q, k, v = attn.get_query_key_value_tensors(torch.randn(6, 3, 64))
        assert q.shape == (6, 3, 4, 16)
        assert k.shape == (6, 3, 2, 16)
        assert v.shape == (6, 3, 2, 16)


# ---------------------------------------------------------------------------
# DiTCrossAttention
# ---------------------------------------------------------------------------


class TestDiTCrossAttentionInit:
    """Tests for DiTCrossAttention __init__ (layernorm + linear_kv overrides)."""

    def _run_init(self, config, submodules, mock_super_init):
        """Construct a DiTCrossAttention, mocking the parent __init__ to set up nn.Module."""
        attn = DiTCrossAttention.__new__(DiTCrossAttention)
        _init_module_and_attrs(attn, config)
        mock_super_init.side_effect = lambda *a, **kw: None
        DiTCrossAttention.__init__(attn, config, submodules, layer_number=1, attn_mask_type=AttnMaskType.no_mask)
        return attn

    @patch("megatron.bridge.diffusion.models.common.dit_attention.CrossAttention.__init__", return_value=None)
    @patch(
        "megatron.bridge.diffusion.models.common.dit_attention.build_module",
        side_effect=lambda *a, **kw: nn.Identity(),
    )
    def test_layernorms_and_linear_kv_built(self, mock_build, mock_super_init):
        config = _make_config(layernorm_across_heads=False, add_bias_linear=False)
        submodules = DiTCrossAttentionSubmodules(
            q_layernorm=MagicMock(), k_layernorm=MagicMock(), linear_kv=MagicMock()
        )

        attn = self._run_init(config, submodules, mock_super_init)

        assert attn.q_layernorm is not None
        assert attn.k_layernorm is not None
        # 3 calls: q_layernorm, k_layernorm, linear_kv
        assert mock_build.call_count == 3

    @patch("megatron.bridge.diffusion.models.common.dit_attention.CrossAttention.__init__", return_value=None)
    @patch(
        "megatron.bridge.diffusion.models.common.dit_attention.build_module",
        side_effect=lambda *a, **kw: nn.Identity(),
    )
    def test_layernorms_none_when_absent(self, mock_build, mock_super_init):
        config = _make_config()
        submodules = DiTCrossAttentionSubmodules(q_layernorm=None, k_layernorm=None, linear_kv=MagicMock())

        attn = self._run_init(config, submodules, mock_super_init)

        assert attn.q_layernorm is None
        assert attn.k_layernorm is None
        # Only linear_kv built
        assert mock_build.call_count == 1

    @patch("megatron.bridge.diffusion.models.common.dit_attention.CrossAttention.__init__", return_value=None)
    @patch(
        "megatron.bridge.diffusion.models.common.dit_attention.build_module",
        side_effect=lambda *a, **kw: nn.Identity(),
    )
    def test_linear_kv_uses_crossattn_emb_size(self, mock_build, mock_super_init):
        config = _make_config(hidden_size=64, crossattn_emb_size=128)
        submodules = DiTCrossAttentionSubmodules(q_layernorm=None, k_layernorm=None, linear_kv=MagicMock())

        self._run_init(config, submodules, mock_super_init)

        call_args = mock_build.call_args_list[0]
        assert call_args[0][1] == 128  # crossattn_emb_size used

    @patch("megatron.bridge.diffusion.models.common.dit_attention.CrossAttention.__init__", return_value=None)
    @patch(
        "megatron.bridge.diffusion.models.common.dit_attention.build_module",
        side_effect=lambda *a, **kw: nn.Identity(),
    )
    def test_linear_kv_falls_back_to_hidden_size(self, mock_build, mock_super_init):
        config = _make_config(hidden_size=64)
        submodules = DiTCrossAttentionSubmodules(q_layernorm=None, k_layernorm=None, linear_kv=MagicMock())

        self._run_init(config, submodules, mock_super_init)

        call_args = mock_build.call_args_list[0]
        assert call_args[0][1] == 64  # config.hidden_size used as fallback


class TestDiTCrossAttentionGetQKV:
    """Tests for DiTCrossAttention.get_query_key_value_tensors (TP=1)."""

    def _make_attn(self, with_layernorm=False, across_heads=False):
        attn = _new_with_module_init(DiTCrossAttention)
        attn.config = _make_config(
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=4,
            layernorm_across_heads=across_heads,
        )
        attn.hidden_size_per_attention_head = 16
        attn.layernorm_across_heads = across_heads

        if with_layernorm:
            attn.q_layernorm = nn.Identity()
            attn.k_layernorm = nn.Identity()
        else:
            attn.q_layernorm = None
            attn.k_layernorm = None

        return attn

    @patch("megatron.bridge.diffusion.models.common.dit_attention.parallel_state")
    @patch("megatron.bridge.diffusion.models.common.dit_attention.CrossAttention.get_query_key_value_tensors")
    def test_delegates_to_super(self, mock_super_qkv, mock_ps):
        mock_ps.get_tensor_model_parallel_world_size.return_value = 1
        q = torch.randn(8, 2, 4, 16)
        k = torch.randn(8, 2, 4, 16)
        v = torch.randn(8, 2, 4, 16)
        mock_super_qkv.return_value = (q, k, v)

        attn = self._make_attn(with_layernorm=False)
        hidden = torch.randn(8, 2, 64)
        kv_states = torch.randn(8, 2, 64)

        q_out, k_out, v_out = attn.get_query_key_value_tensors(hidden, kv_states)

        mock_super_qkv.assert_called_once()
        assert torch.equal(q_out, q)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    @patch("megatron.bridge.diffusion.models.common.dit_attention.parallel_state")
    @patch("megatron.bridge.diffusion.models.common.dit_attention.CrossAttention.get_query_key_value_tensors")
    def test_applies_per_head_layernorm(self, mock_super_qkv, mock_ps):
        mock_ps.get_tensor_model_parallel_world_size.return_value = 1
        q = torch.randn(8, 2, 4, 16)
        k = torch.randn(8, 2, 4, 16)
        v = torch.randn(8, 2, 4, 16)
        mock_super_qkv.return_value = (q, k, v)

        spy_q = MagicMock(side_effect=lambda x: x)
        spy_k = MagicMock(side_effect=lambda x: x)

        attn = self._make_attn(with_layernorm=False, across_heads=False)
        # MagicMock is not nn.Module, so PyTorch __setattr__ stores it in __dict__ directly
        attn.q_layernorm = spy_q
        attn.k_layernorm = spy_k

        attn.get_query_key_value_tensors(torch.randn(8, 2, 64), torch.randn(8, 2, 64))

        spy_q.assert_called_once()
        spy_k.assert_called_once()

    @patch("megatron.bridge.diffusion.models.common.dit_attention.parallel_state")
    @patch("megatron.bridge.diffusion.models.common.dit_attention.CrossAttention.get_query_key_value_tensors")
    def test_output_shapes_preserved(self, mock_super_qkv, mock_ps):
        mock_ps.get_tensor_model_parallel_world_size.return_value = 1
        q = torch.randn(6, 3, 4, 16)
        k = torch.randn(6, 3, 4, 16)
        v = torch.randn(6, 3, 4, 16)
        mock_super_qkv.return_value = (q, k, v)

        attn = self._make_attn(with_layernorm=True, across_heads=False)
        q_out, k_out, v_out = attn.get_query_key_value_tensors(torch.randn(6, 3, 64), torch.randn(6, 3, 64))

        assert q_out.shape == (6, 3, 4, 16)
        assert k_out.shape == (6, 3, 4, 16)
        assert v_out.shape == (6, 3, 4, 16)
