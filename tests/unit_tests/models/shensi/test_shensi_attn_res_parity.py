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

"""Attention-residual parity against the HF reference (layout transposed to seq-first)."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.shensi.modeling_shensi import ShensiAttentionResidual


HIDDEN, RANK, LAYERS, EPS = 64, 8, 4, 1e-6


def _hf_module():
    pytest.importorskip("transformers.models.shensi.configuration_shensi")
    pytest.importorskip("transformers.models.shensi.modeling_shensi")
    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiAttentionResidual as HFResidual

    try:
        cfg = ShensiConfig(
            hidden_size=HIDDEN,
            attn_res_rank=RANK,
            num_hidden_layers=LAYERS,
            rms_norm_eps=EPS,
            vocab_size=32,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "no usable reference here"
        pytest.skip(f"the installed HF ShensiConfig cannot be built: {type(exc).__name__}: {exc}")
    return HFResidual(cfg).float().eval()


def _ours(hf: torch.nn.Module):
    cfg = SimpleNamespace(
        hidden_size=HIDDEN,
        attn_res_rank=RANK,
        num_layers=LAYERS,
        layernorm_epsilon=EPS,
    )
    module = ShensiAttentionResidual(cfg).float().eval()
    src, dst = hf.state_dict(), module.state_dict()
    shared = {k: v for k, v in src.items() if k in dst and src[k].shape == dst[k].shape}
    module.load_state_dict(shared, strict=False)
    assert set(shared) == set(dst), f"unaligned keys: {sorted(set(dst) - set(shared))}"
    return module


@pytest.mark.parametrize("with_delta", [True, False])
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
def test_attention_residual_matches_hf(with_delta, num_blocks):
    hf = _hf_module()
    ours = _ours(hf)

    torch.manual_seed(1)
    # HF uses [batch, seq, ...]; the Bridge module uses [seq, batch, ...].
    prefix = torch.randn(2, 5, HIDDEN)
    delta = torch.randn(2, 5, HIDDEN) if with_delta else None
    blocks = torch.randn(2, 5, 3, HIDDEN)

    with torch.no_grad():
        reference, _ = hf(prefix, delta, blocks, num_blocks)
        actual, _ = ours(
            prefix.transpose(0, 1),
            delta.transpose(0, 1) if delta is not None else None,
            blocks.transpose(0, 1),
            num_blocks,
        )
    torch.testing.assert_close(actual.transpose(0, 1), reference, atol=1e-6, rtol=1e-5)
