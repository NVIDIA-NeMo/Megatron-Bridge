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

"""Tests for the Gemma2 TransformerEngine attention path.

Three tiers, gated as narrowly as each actually requires:

**CPU-only**, so these always execute. The config rewrite Gemma2TEDotProductAttention
performs, the provider's backend forcing and softcap validation, and the layer-spec selection.
They exist because everything here used to sit behind the softcap gate below and was therefore
skipped in CI, which is how an inverted sliding-window layer parity once shipped.

**CUDA only.** Forward-output parity against the unfused Gemma2DotProductAttention oracle, and
SWA masking. Both run with the cap disabled: masking, the Gemma2 attention scale and the output
layout are all observable uncapped, and mcore asserts ``_te_dpa_supports_softcap`` only when
``attn_logit_softcapping`` is set. So these need a GPU but not a softcap-capable build.

**CUDA and a softcap-capable TE.** Only ``test_softcap_applied_pre_softmax``, which is the one
assertion that genuinely cannot be made without the fused kernel.

Run with:
    uv run pytest tests/unit_tests/models/gemma/test_gemma2_te_attention.py
"""

import datetime
import math
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType

from megatron.bridge.models.gemma.gemma2_provider import (
    Gemma2DotProductAttention,
    Gemma2FlexDotProductAttention,
    Gemma2ModelProvider,
    Gemma2TEDotProductAttention,
    gemma2_layer_spec,
)


try:
    # Megatron-LM plumbs attn_logit_softcapping to TE's `softcap` kwarg and exports this probe
    # alongside it. Its absence means 3rdparty/Megatron-LM predates that change, so the TE path
    # cannot carry a cap yet and these parity tests would compare two uncapped runs.
    from megatron.core.extensions.transformer_engine import _te_dpa_supports_softcap
except Exception:  # pragma: no cover - import guard
    _te_dpa_supports_softcap = False


_HAVE_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _HAVE_CUDA, reason="Requires a CUDA device")
requires_te_flash_softcap = pytest.mark.skipif(
    not (_HAVE_CUDA and _te_dpa_supports_softcap),
    reason=(
        "Requires CUDA, a Megatron-LM with attn_logit_softcapping plumbed to TE "
        "(NVIDIA/Megatron-LM#6590 plus a 3rdparty/Megatron-LM bump), and a TransformerEngine "
        "build whose DotProductAttention accepts softcap (NVIDIA/TransformerEngine#3391)"
    ),
)

# bf16 flash attention vs an eager fp-accumulated reference: loose but meaningful tolerances.
_RTOL = 2e-2
_ATOL = 2e-2

# Small attention shape kept intentionally light for CI GPUs.
_NUM_HEADS = 4
_HEAD_DIM = 64
# Deliberately != _HEAD_DIM. Gemma2 scales by 1/sqrt(query_pre_attn_scalar), so if these were
# equal the Gemma2 scale would coincide with the default 1/sqrt(head_dim) and dropping the
# scale logic entirely would still pass every test here.
_QUERY_PRE_ATTN_SCALAR = 256  # scale = 1/sqrt(256) = 1/16, vs 1/sqrt(64) = 1/8 for head_dim
_SOFTCAP = 50.0


def _make_config(window_size=(3, 0), softcap=_SOFTCAP):
    """Build a Gemma2ModelProvider usable directly as the attention TransformerConfig.

    A small sliding window (default (3, 0)) keeps the seq>window masking test tractable; the
    production Gemma2 window is (4095, 0).
    """
    provider = Gemma2ModelProvider(
        num_layers=2,
        hidden_size=_NUM_HEADS * _HEAD_DIM,
        num_attention_heads=_NUM_HEADS,
        num_query_groups=_NUM_HEADS,  # no GQA — keep the oracle repeat_interleave a no-op
        kv_channels=_HEAD_DIM,
        query_pre_attn_scalar=_QUERY_PRE_ATTN_SCALAR,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        masked_softmax_fusion=False,
        attention_softmax_in_fp32=True,
        apply_query_key_layer_scaling=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    # window_size and attn_logit_softcapping are real Gemma2ModelProvider fields (guarded setattr).
    assert hasattr(provider, "window_size")
    assert hasattr(provider, "attn_logit_softcapping")
    provider.window_size = window_size
    provider.attn_logit_softcapping = softcap
    return provider


def _qkv(seq, batch, *, device, dtype=torch.bfloat16, scale=1.0, seed=0):
    """Random sbhd query/key/value tensors: [seq, batch, num_heads, head_dim]."""
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (seq, batch, _NUM_HEADS, _HEAD_DIM)
    q = torch.randn(shape, device=device, dtype=dtype, generator=gen) * scale
    k = torch.randn(shape, device=device, dtype=dtype, generator=gen) * scale
    v = torch.randn(shape, device=device, dtype=dtype, generator=gen) * scale
    return q, k, v


@pytest.mark.unit
class TestGemma2TEDotProductAttentionConfigRewrite:
    """CPU-only checks on the config Gemma2TEDotProductAttention hands to TransformerEngine.

    Unlike the parity tests below, these need no GPU and no softcap-capable TE build, so they
    actually execute in CI. Gemma2TEDotProductAttention does all of its work before calling
    super().__init__, so patching the base captures everything it computed. Patching is not
    merely a convenience: mcore asserts `_te_dpa_supports_softcap` whenever
    attn_logit_softcapping is set, so real construction cannot run against a TE without softcap.
    """

    @staticmethod
    def _captured_init(provider, layer_number, **kwargs):
        """Construct the TE attention class with the base stubbed, returning the captured kwargs."""
        with patch.object(Gemma2TEDotProductAttention.__bases__[0], "__init__", return_value=None) as init:
            Gemma2TEDotProductAttention(
                config=provider,
                layer_number=layer_number,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                **kwargs,
            )
        return init.call_args.kwargs

    @pytest.mark.parametrize(
        "layer_number, expect_window",
        [(1, True), (2, False), (3, True), (4, False)],
    )
    def test_sliding_window_on_odd_layers_only(self, layer_number, expect_window):
        """SWA belongs on odd 1-indexed layers; even layers are full causal.

        HuggingFace builds layer_types as "sliding_attention" when (i + 1) % 2 is truthy over its
        0-indexed i, and mcore's layer_number is i + 1, so HF layer 0 is mcore layer 1. This
        mirrors Gemma2DotProductAttention, which applies the window when layer_number % 2 == 1.
        Getting this backwards does not crash; it silently trains a different model.
        """
        window = (4095, 0)
        provider = _make_config(window_size=window)
        captured = self._captured_init(provider, layer_number)
        if expect_window:
            assert captured["config"].window_size == window, (
                f"layer_number={layer_number} is odd and must use sliding-window attention"
            )
        else:
            assert captured["config"].window_size is None, (
                f"layer_number={layer_number} is even and must use full causal attention"
            )

    def test_layer_number_zero_is_clamped_to_one(self):
        """max(1, layer_number) means layer 0 is treated as layer 1, so it slides."""
        provider = _make_config(window_size=(4095, 0))
        assert self._captured_init(provider, 0)["config"].window_size == (4095, 0)

    def test_softmax_scale_uses_query_pre_attn_scalar_not_head_dim(self):
        """Gemma2 scales by 1/sqrt(query_pre_attn_scalar), which is not 1/sqrt(head_dim).

        The fixture keeps the two deliberately different, so dropping the scale logic entirely
        would fail here rather than silently coinciding with the TE default.
        """
        provider = _make_config()
        captured = self._captured_init(provider, 1)
        assert _QUERY_PRE_ATTN_SCALAR != _HEAD_DIM, "fixture must keep these distinct"
        assert captured["softmax_scale"] == pytest.approx(1.0 / math.sqrt(_QUERY_PRE_ATTN_SCALAR))
        assert captured["softmax_scale"] != pytest.approx(1.0 / math.sqrt(_HEAD_DIM))

    def test_explicit_softmax_scale_is_honored(self):
        """An explicit softmax_scale from SelfAttention must not be overwritten."""
        provider = _make_config()
        captured = self._captured_init(provider, 1, softmax_scale=0.123)
        assert captured["softmax_scale"] == pytest.approx(0.123)

    def test_provider_config_is_not_mutated(self):
        """The per-layer rewrite happens on a deep copy.

        Without the copy, constructing an even layer would write window_size=None onto the shared
        provider and every later layer would inherit it.
        """
        window = (4095, 0)
        provider = _make_config(window_size=window)
        self._captured_init(provider, 2)  # even layer clears window_size on its own copy
        assert provider.window_size == window
        assert self._captured_init(provider, 1)["config"] is not provider

    def test_softcap_survives_the_deep_copy(self):
        """attn_logit_softcapping must reach TE; mcore turns it into the `softcap` kwarg."""
        provider = _make_config(softcap=_SOFTCAP)
        assert self._captured_init(provider, 1)["config"].attn_logit_softcapping == _SOFTCAP


@pytest.mark.unit
class TestGemma2ModelProviderAttentionConstraints:
    """CPU-only checks on the provider's backend forcing and softcap validation."""

    def test_default_leaves_backend_untouched(self):
        """The opt-in flag defaults off, and the default path keeps the inherited backend."""
        provider = _make_config()
        assert provider.use_transformer_engine_attention is False
        assert provider.attention_backend is AttnBackend.auto

    def test_enabling_the_flag_forces_flash_backend(self):
        """cuDNN cannot serve Gemma2 (head_dim=256, no softcap), so the TE path requires flash."""
        provider = _make_config()
        provider.use_transformer_engine_attention = True
        provider.__post_init__()
        assert provider.attention_backend is AttnBackend.flash

    def test_finalize_reapplies_constraints_after_overrides(self):
        """apply_overrides_and_finalize() never re-runs __post_init__.

        Without the finalize() override, enabling the flag through the canonical override path
        would select Gemma2TEDotProductAttention while leaving attention_backend at auto.
        """
        provider = _make_config()
        assert provider.attention_backend is AttnBackend.auto
        provider.apply_overrides_and_finalize(overrides={"use_transformer_engine_attention": True})
        assert provider.attention_backend is AttnBackend.flash

    @pytest.mark.parametrize("bad_cap", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_softcap_rejected_at_construction(self, bad_cap):
        """0.0 divides by zero, a negative cap is applied as its absolute value because tanh is
        odd, and nan is truthy so it slips past `if not scale` and yields all-NaN scores."""
        provider = _make_config()
        provider.attn_logit_softcapping = bad_cap
        with pytest.raises(ValueError, match="positive finite"):
            provider.__post_init__()

    @pytest.mark.parametrize("bad_cap", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_softcap_rejected_through_the_override_path(self, bad_cap):
        """The override path must validate too; before finalize() existed it accepted anything."""
        provider = _make_config()
        with pytest.raises(ValueError, match="positive finite"):
            provider.apply_overrides_and_finalize(overrides={"attn_logit_softcapping": bad_cap})

    def test_none_softcap_is_allowed(self):
        """None disables the cap, as HuggingFace spells it."""
        provider = _make_config(softcap=None)
        provider.__post_init__()
        assert provider.attn_logit_softcapping is None
        provider.apply_overrides_and_finalize(overrides={"attn_logit_softcapping": None})
        assert provider.attn_logit_softcapping is None


@pytest.mark.unit
class TestGemma2LayerSpecSelection:
    """CPU-only checks that the opt-in flag picks the right core-attention class."""

    def test_flag_selects_the_transformer_engine_path(self):
        provider = _make_config()
        provider.use_transformer_engine_attention = True
        spec = gemma2_layer_spec(provider)
        assert spec.submodules.self_attention.submodules.core_attention is Gemma2TEDotProductAttention

    def test_default_keeps_the_unfused_flex_path(self):
        """The no-regression default: anyone not opting in is untouched."""
        provider = _make_config()
        spec = gemma2_layer_spec(provider)
        assert spec.submodules.self_attention.submodules.core_attention is Gemma2FlexDotProductAttention


@requires_cuda
class TestGemma2TEDotProductAttentionParity:
    """Parity of the TE flash path against the unfused Gemma2 oracle.

    Gated on CUDA only. Masking, the Gemma2 attention scale and output layout are all
    observable with the cap disabled, and mcore asserts _te_dpa_supports_softcap only when
    attn_logit_softcapping is set, so these run against any TE with flash attention. Only
    the softcap test below additionally needs a softcap-capable build.
    """

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29513")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.cuda.set_device(0)
            dist.init_process_group(
                backend="nccl",
                world_size=1,
                rank=0,
                timeout=datetime.timedelta(minutes=10),
            )

    @classmethod
    def teardown_class(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setup_method(self):
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        parallel_state.destroy_model_parallel()

    @staticmethod
    def _pg_collection():
        return ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )

    def _make_oracle(self, config, layer_number):
        return Gemma2DotProductAttention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        ).cuda()

    def _make_te(self, config, layer_number):
        return Gemma2TEDotProductAttention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=self._pg_collection(),
        ).cuda()

    @pytest.mark.parametrize(
        "layer_number, label",
        [(1, "odd/SWA"), (2, "even/causal")],
    )
    def test_forward_output_parity(self, layer_number, label):
        """TE flash output must match the unfused oracle on both odd (SWA) and even (causal) layers."""
        seq, batch = 16, 2
        # Uncapped: this test is about masking, scale and layout, none of which need the cap.
        config = _make_config(window_size=(3, 0), softcap=None)
        oracle = self._make_oracle(config, layer_number)
        te = self._make_te(config, layer_number)

        q, k, v = _qkv(seq, batch, device="cuda", seed=7)
        oracle_out = oracle.forward(query=q, key=k, value=v, attention_mask=None, attn_mask_type=AttnMaskType.causal)
        te_out = te.forward(query=q, key=k, value=v, attention_mask=None, attn_mask_type=AttnMaskType.causal)

        assert te_out.shape == oracle_out.shape
        torch.testing.assert_close(
            te_out.float(),
            oracle_out.float(),
            rtol=_RTOL,
            atol=_ATOL,
            msg=f"TE flash path diverged from the unfused Gemma2 oracle on the {label} layer",
        )

    def test_swa_excludes_tokens_beyond_window_only_on_odd_layers(self):
        """Perturbing a key/value beyond the SWA window must leave an odd-layer last-query output
        unchanged, while an even-layer (full causal) output must change.

        Sliding window attention is applied on odd layers only (1-indexed), matching HuggingFace's
        layer_types where 0-indexed layer 0 is "sliding_attention". Each layer is checked against
        the unfused oracle as well as against the absolute convention, so that this test cannot be
        satisfied by a TE path that is merely self-consistent while disagreeing with the oracle.
        """
        seq, batch = 8, 1
        window_left = 3
        # Uncapped: pure masking behaviour, independent of the softcap.
        config = _make_config(window_size=(window_left, 0), softcap=None)

        q, k, v = _qkv(seq, batch, device="cuda", seed=11)
        # Perturb a token strictly outside the last query's window: position 0 is far past for the
        # last query (index seq-1) whose window only covers [seq-1-window_left, seq-1].
        k_pert = k.clone()
        v_pert = v.clone()
        k_pert[0] += 5.0
        v_pert[0] += 5.0

        def last_query(attn, key, value):
            out = attn.forward(query=q, key=key, value=value, attention_mask=None, attn_mask_type=AttnMaskType.causal)
            return out[-1].float()  # [batch, hidden]

        def reacts_to_out_of_window_token(attn):
            base = last_query(attn, k, v)
            pert = last_query(attn, k_pert, v_pert)
            return not torch.allclose(pert, base, rtol=_RTOL, atol=_ATOL)

        for layer_number, is_swa in ((1, True), (2, False)):
            label = "odd/SWA" if is_swa else "even/causal"
            te_reacts = reacts_to_out_of_window_token(self._make_te(config, layer_number))
            oracle_reacts = reacts_to_out_of_window_token(self._make_oracle(config, layer_number))

            assert te_reacts == oracle_reacts, (
                f"TE path disagrees with the unfused oracle on the {label} layer: TE "
                f"{'attends to' if te_reacts else 'ignores'} a token beyond the sliding window "
                f"while the oracle {'attends to' if oracle_reacts else 'ignores'} it"
            )
            # A SWA layer must ignore the out-of-window token; a causal layer must attend to it.
            assert te_reacts == (not is_swa), (
                f"{label} layer (layer_number={layer_number}) has the wrong masking: expected "
                f"{'SWA to exclude' if is_swa else 'full causal to include'} tokens beyond the window"
            )

    @requires_te_flash_softcap
    def test_softcap_applied_pre_softmax(self):
        """With large logits, the TE path must saturate via 50*tanh (matching the oracle) and differ
        from an uncapped TE path — proving the softcap is applied pre-softmax."""
        seq, batch = 16, 1
        layer_number = 2  # even layer -> full causal, isolating the softcap from SWA masking
        config = _make_config(window_size=(3, 0), softcap=_SOFTCAP)

        # Large scale drives pre-softmax logits well past the +/-50 softcap saturation range.
        q, k, v = _qkv(seq, batch, device="cuda", scale=8.0, seed=17)

        oracle = self._make_oracle(config, layer_number)  # applies 50*tanh softcap
        te_capped = self._make_te(config, layer_number)

        # Uncapped reference: identical config/scale but softcap disabled. Built through the same
        # Gemma2 subclass as the capped arm, so that a None cap being reinstated on this path
        # would show up here rather than being sidestepped by using the base class.
        uncapped_config = _make_config(window_size=(3, 0), softcap=_SOFTCAP)
        uncapped_config.attn_logit_softcapping = None
        te_uncapped = self._make_te(uncapped_config, layer_number)

        oracle_out = oracle.forward(
            query=q, key=k, value=v, attention_mask=None, attn_mask_type=AttnMaskType.causal
        ).float()
        capped_out = te_capped.forward(
            query=q, key=k, value=v, attention_mask=None, attn_mask_type=AttnMaskType.causal
        ).float()
        uncapped_out = te_uncapped.forward(
            query=q, key=k, value=v, attention_mask=None, attn_mask_type=AttnMaskType.causal
        ).float()

        # Softcapped TE flash matches the softcapped oracle.
        torch.testing.assert_close(
            capped_out,
            oracle_out,
            rtol=_RTOL,
            atol=_ATOL,
            msg="Softcapped TE flash path must match the softcapped Gemma2 oracle under large logits",
        )
        # And the softcap must actually change the result vs. an uncapped path.
        assert not torch.allclose(capped_out, uncapped_out, rtol=_RTOL, atol=_ATOL), (
            "Softcap (50*tanh) must alter attention vs. the uncapped path under large logits"
        )
