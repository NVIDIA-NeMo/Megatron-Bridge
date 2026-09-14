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

"""Numerical parity tests for Gemma2TEDotProductAttention.

These compare the TransformerEngine flash-attention path (Gemma2TEDotProductAttention)
against the unfused Gemma2DotProductAttention oracle on small tensors, covering:

* per-layer forward-output parity (odd/SWA layer and even/causal layer),
* SWA masking: tokens beyond the window are excluded only on odd layers,
* the tanh softcap being applied pre-softmax.

They require a real GPU and a TransformerEngine build whose DotProductAttention exposes a
``softcap`` argument, so they are skipped cleanly otherwise. They are not run in CI on CPU.

Run with:
    uv run pytest tests/unit_tests/models/gemma/test_gemma2_te_attention.py
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gemma.gemma2_provider import (
    Gemma2DotProductAttention,
    Gemma2ModelProvider,
    Gemma2TEDotProductAttention,
)


try:
    # Megatron-LM plumbs attn_logit_softcapping to TE's `softcap` kwarg and exports this probe
    # alongside it. Its absence means 3rdparty/Megatron-LM predates that change, so the TE path
    # cannot carry a cap yet and these parity tests would compare two uncapped runs.
    from megatron.core.extensions.transformer_engine import _te_dpa_supports_softcap
except Exception:  # pragma: no cover - import guard
    _te_dpa_supports_softcap = False


_HAVE_CUDA = torch.cuda.is_available()
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


@requires_te_flash_softcap
class TestGemma2TEDotProductAttentionParity:
    """Parity of the TE flash path against the unfused Gemma2 oracle."""

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
        config = _make_config(window_size=(3, 0))
        oracle = self._make_oracle(config, layer_number)
        te = self._make_te(config, layer_number)

        q, k, v = _qkv(seq, batch, device="cuda", seed=7)
        oracle_out = oracle.forward(query=q, key=k, value=v, attention_mask=None)
        te_out = te.forward(query=q, key=k, value=v, attention_mask=None)

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
        config = _make_config(window_size=(window_left, 0))

        q, k, v = _qkv(seq, batch, device="cuda", seed=11)
        # Perturb a token strictly outside the last query's window: position 0 is far past for the
        # last query (index seq-1) whose window only covers [seq-1-window_left, seq-1].
        k_pert = k.clone()
        v_pert = v.clone()
        k_pert[0] += 5.0
        v_pert[0] += 5.0

        def last_query(attn, key, value):
            out = attn.forward(query=q, key=key, value=value, attention_mask=None)
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

        oracle_out = oracle.forward(query=q, key=k, value=v, attention_mask=None).float()
        capped_out = te_capped.forward(query=q, key=k, value=v, attention_mask=None).float()
        uncapped_out = te_uncapped.forward(query=q, key=k, value=v, attention_mask=None).float()

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
