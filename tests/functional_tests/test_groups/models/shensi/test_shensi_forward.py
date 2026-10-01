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

"""Forward pass smoke test: the tiny converted model must run and produce finite logits.

This is the test that covers the hash layers' token-id plumbing: `ShensiHashMLP` gates its
output with `deepemb(input_ids)`, token ids do not travel through `GPTModel.forward` into the
transformer block, and `ShensiModel.forward` therefore publishes them for the hash layers.
"""

import os

import pytest
import torch

from megatron.bridge import AutoBridge

from . import make_tiny_hf_model


SEQ_LEN = 16

# TransformerEngine-FL / FlagGems segfault on SM120 (the flagos backend) in this
# environment, so the vendored CUDA kernels are selected here as well.
os.environ.setdefault("TE_FL_PREFER", "vendor")


def _tiny_converted_pair(tmp_path):
    """Tiny HF checkpoint → Megatron model with the HF weights loaded."""
    hf_model = make_tiny_hf_model()
    hf_dir = tmp_path / "shensi_tiny_forward"
    hf_model.save_pretrained(hf_dir, safe_serialization=False)

    bridge = AutoBridge.from_hf_pretrained(hf_dir)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    # No fused DSA kernels in the test environment: the PyTorch fallback path is used.
    provider.dsa_kernel_backend = "none"
    # The indexer rotation takes bf16 inputs; fp32 weights are cast during loading.
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)
    return hf_model, provider, model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="forward pass needs CUDA")
def test_tiny_forward_produces_finite_logits(tmp_path):
    _, provider, model = _tiny_converted_pair(tmp_path)

    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)

    with torch.no_grad():
        # The sparse attention takes an implicit causal mask: attention_mask must be None.
        logits = model[0](tokens, position_ids, attention_mask=None)

    assert logits.shape[0] == 1 and logits.shape[1] == SEQ_LEN, logits.shape
    assert torch.isfinite(logits).all(), "forward produced NaN/Inf"


@torch.no_grad()
def _forward_logits(model, tokens, attention_mask=None):
    position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    return model[0](tokens, position_ids, attention_mask=attention_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="forward pass needs CUDA")
def test_right_padding_keeps_the_valid_prefix_close(tmp_path):
    """A dropped right-padding mask still leaves the valid prefix numerically sane.

    Bit-exactness is not claimed: the sparse attention takes no mask, so the pads still take
    part in the compressed blocks and perturb the prefix slightly. What is guaranteed is that
    causality holds: a pad is only visible to later pads. Rollout replies are right-padded,
    which is what makes that path work; left padding is rejected, see the next case.
    """
    _, provider, model = _tiny_converted_pair(tmp_path)
    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")

    base = _forward_logits(model, tokens, attention_mask=None)

    pad_len = 8
    padded_tokens = torch.cat([tokens, torch.zeros(1, pad_len, dtype=torch.long, device="cuda")], dim=1)
    mask = torch.cat([torch.ones_like(tokens), torch.zeros(1, pad_len, dtype=torch.long, device="cuda")], dim=1)
    padded = _forward_logits(model, padded_tokens, attention_mask=mask)

    assert padded.shape[1] == SEQ_LEN + pad_len
    assert torch.isfinite(padded).all(), "padding produced NaN/Inf"
    diff = (base - padded[:, :SEQ_LEN]).abs().max().item()
    # Tiny randomly initialized logits are of order 1e-2, so this bound is generous while
    # still catching a blow-up caused by the pads.
    assert diff < 1e-2, f"right padding moved the valid prefix too far: max|delta|={diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="forward pass needs CUDA")
def test_left_padding_is_rejected(tmp_path):
    """Left padding puts pads before valid tokens: it must raise instead of computing nonsense."""
    _, provider, model = _tiny_converted_pair(tmp_path)
    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")

    pad_len = 4
    padded_tokens = torch.cat([torch.zeros(1, pad_len, dtype=torch.long, device="cuda"), tokens], dim=1)
    mask = torch.cat([torch.zeros(1, pad_len, dtype=torch.long, device="cuda"), torch.ones_like(tokens)], dim=1)
    with pytest.raises(ValueError, match="not right-padded"):
        _forward_logits(model, padded_tokens, attention_mask=mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Megatron side needs CUDA")
def test_hf_forward_matches_megatron_within_bfloat16_noise(tmp_path) -> None:
    """The whole stack agrees with the HF reference: fp32 HF on CPU vs bf16 Megatron on CUDA.

    The HF reference runs in fp32 (loading it as bf16 mixes dtypes inside its fp32-kept modules and
    raises), so the comparison carries the bf16 rounding of the Megatron side. A structural
    mismatch — a weight mapped to the wrong module, a missing contraction — moves the logits by
    whole units, far outside the tolerance below, which is why this is worth asserting.
    """
    from safetensors.torch import load_file  # noqa: F401  (kept next to the other conversion imports)

    hf_model, provider, model = _tiny_converted_pair(tmp_path)
    hf_model = hf_model.float().eval()
    unwrapped = model[0].module if hasattr(model[0], "module") else model[0]

    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN))
    with torch.no_grad():
        reference = hf_model(tokens).logits.float()
        got = unwrapped(tokens.cuda(), torch.arange(SEQ_LEN, device="cuda").unsqueeze(0), attention_mask=None)

    got = got.float().cpu()
    assert got.shape == reference.shape
    assert torch.isfinite(got).all()
    max_diff = float((reference - got).abs().max())
    assert max_diff < 5e-2, f"HF and Megatron logits differ by {max_diff:.3e}"
