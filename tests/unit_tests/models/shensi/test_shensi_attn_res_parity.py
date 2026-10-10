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

"""AttnRes numerics against the Hugging Face reference (CPU, float32, bit-exact)."""

import pytest
import torch

from megatron.bridge.models.shensi.modeling_shensi import ShensiAttentionResidual


HIDDEN, RANK, LAYERS, EPS = 64, 16, 4, 1e-6


def _hf_attention_residual():
    """The HF reference module (skips the test when the shensi HF reference is absent or unusable).

    The reference lives in the (unreleased) shensi transformers implementation: it is neither in the
    released upstream package nor guaranteed to be constructible by the installed huggingface_hub
    (the HF config uses sentinel defaults for fields it does not support, which strict dataclass
    validation rejects). Both cases skip with the reason rather than failing the test run.
    """
    pytest.importorskip("transformers.models.shensi.configuration_shensi")
    pytest.importorskip("transformers.models.shensi.modeling_shensi")
    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiAttentionResidual

    try:
        cfg = ShensiConfig(
            hidden_size=HIDDEN,
            routed_expert_hidden_size=RANK,
            num_hidden_layers=LAYERS,
            rms_norm_eps=EPS,
            vocab_size=32,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "no usable reference here"
        pytest.skip(f"the installed HF ShensiConfig cannot be built: {type(exc).__name__}: {exc}")
    return ShensiAttentionResidual(cfg).float().eval()


def _megatron_attention_residual(hf: torch.nn.Module) -> torch.nn.Module:
    """The Megatron-Core module, loaded with the HF weights that have a matching name and shape."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=HIDDEN,
        routed_expert_hidden_size=RANK,
        num_layers=LAYERS,
        num_hidden_layers=LAYERS,
        layernorm_epsilon=EPS,
        attn_res_pp_state_transfer=False,
        recompute_granularity=None,
        fp32_keep=None,
        bf16=False,
        fp16=False,
        params_dtype=torch.float32,
        bf16_attention=False,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        fp8=None,
        fp8_param=False,
    )
    module = ShensiAttentionResidual(cfg).float().eval()
    src, dst = hf.state_dict(), module.state_dict()
    shared = {k: v for k, v in src.items() if k in dst and src[k].shape == dst[k].shape}
    module.load_state_dict(shared, strict=False)
    # Both sides must expose the same parameter names, otherwise the comparison is vacuous.
    assert set(shared) == set(dst), f"unaligned keys: {sorted(set(dst) - set(shared))}"
    return module


@pytest.mark.parametrize("with_delta", [True, False])
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
def test_attention_residual_matches_hf(with_delta, num_blocks):
    hf = _hf_attention_residual()
    ours = _megatron_attention_residual(hf)

    torch.manual_seed(1)
    prefix = torch.randn(2, 5, HIDDEN)
    delta = torch.randn(2, 5, HIDDEN) if with_delta else None
    blocks = torch.randn(2, 5, 3, HIDDEN)
    output_norm_weight = torch.randn(HIDDEN)

    with torch.no_grad():
        # The read path samples its value rows from the global RNG, so both sides need the
        # same RNG state for a bit-exact comparison.
        torch.manual_seed(7)
        hf_out, hf_updated = hf(prefix, delta, blocks, output_norm_weight, num_blocks)
        torch.manual_seed(7)
        ours_out, ours_updated = ours(prefix, delta, blocks, output_norm_weight, num_blocks)

    assert torch.equal(hf_out, ours_out), f"max abs diff={float((hf_out - ours_out).abs().max()):.3e}"
    assert torch.equal(hf_updated, ours_updated), f"max abs diff={float((hf_updated - ours_updated).abs().max()):.3e}"


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_slot_layout_matches_the_stacked_layout(num_blocks):
    """The per-slot blocks the state carries read out exactly like the reference's stacked tensor.

    Everything the read touches must be layout-independent: the same draws, the same rows in the
    same order, the same metric and the same weighted retrieval. Only the summation order inside
    the two replacements for the reference's matmuls may differ, so the comparison is numerical
    with a one-ulp-scale bound.
    """
    hf = _hf_attention_residual()
    ours = _megatron_attention_residual(hf)

    torch.manual_seed(3)
    prefix = torch.randn(2, 5, HIDDEN)
    delta = torch.randn(2, 5, HIDDEN)
    stacked = torch.randn(2, 5, 3, HIDDEN)
    slots = list(stacked.unbind(-2))

    with torch.no_grad():
        torch.manual_seed(11)
        stacked_out, stacked_updated = ours(prefix, delta, stacked, None, num_blocks)
        torch.manual_seed(11)
        slot_out, slot_updated = ours(prefix, delta, slots, None, num_blocks)

    # The write path does not touch the blocks at all, so this side is exact.
    assert torch.equal(stacked_updated, slot_updated)
    diff = float((stacked_out - slot_out).abs().max())
    assert diff < 1e-6, f"max abs diff={diff:.3e}"


def test_every_block_receives_gradient_in_both_layouts():
    """The retrieval is differentiable: each block slot is a live activation, in either layout."""
    hf = _hf_attention_residual()

    torch.manual_seed(5)
    prefix = torch.randn(2, 5, HIDDEN)
    delta = torch.randn(2, 5, HIDDEN)
    base = torch.randn(2, 5, 3, HIDDEN)

    grads = []
    for as_slots in (False, True):
        stacked = base.clone().requires_grad_(True)
        blocks = [s.detach().requires_grad_(True) for s in stacked.unbind(-2)] if as_slots else stacked
        ours = _megatron_attention_residual(hf)
        torch.manual_seed(13)
        out, _ = ours(prefix, delta, blocks, None, 3)
        out.sum().backward()
        per_block = [s.grad for s in blocks] if as_slots else [stacked.grad[..., k, :] for k in range(3)]
        assert all(g is not None and float(g.abs().sum()) > 0 for g in per_block), "a block lost its gradient"
        grads.append(per_block)

    for from_stacked, from_slots in zip(*grads):
        # The two layouts reassociate the same sums, so gradients agree to rounding scale.
        diff = float((from_stacked - from_slots).abs().max())
        assert torch.allclose(from_stacked, from_slots, atol=1e-6, rtol=1e-5), f"max abs grad diff={diff:.3e}"
