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


HIDDEN, RANK, LAYERS, READ_HEADS, EPS = 64, 16, 4, 8, 1e-6


def _hf_attention_residual():
    """The HF reference module (skips the test when the shensi HF implementation is absent)."""
    pytest.importorskip("transformers.models.shensi.modeling_shensi")
    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiAttentionResidual

    cfg = ShensiConfig(
        hidden_size=HIDDEN,
        routed_expert_hidden_size=RANK,
        num_hidden_layers=LAYERS,
        attn_res_read_heads=READ_HEADS,
        rms_norm_eps=EPS,
        vocab_size=32,
    )
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
        attn_res_read_heads=READ_HEADS,
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
        hf_out, hf_updated = hf(prefix, delta, blocks, output_norm_weight, num_blocks)
        ours_out, ours_updated = ours(prefix, delta, blocks, output_norm_weight, num_blocks)

    assert torch.equal(hf_out, ours_out), f"max abs diff={float((hf_out - ours_out).abs().max()):.3e}"
    assert torch.equal(hf_updated, ours_updated), f"max abs diff={float((hf_updated - ours_updated).abs().max()):.3e}"
