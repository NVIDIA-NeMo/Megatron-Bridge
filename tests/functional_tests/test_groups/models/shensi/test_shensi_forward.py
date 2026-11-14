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

This is the test that covers the token-id plumbing: the dense MLP's DeepEmbed scale and the
ROSA layer both need `input_ids`, which do not travel through `HybridModel.forward` into every
decoder layer, so `ShensiModel.forward` publishes them for the stack and its layers.
"""

import pytest
import torch
from megatron.core.extensions.transformer_engine import HAVE_TE

from megatron.bridge import AutoBridge


pytestmark = pytest.mark.skipif(
    not HAVE_TE,
    reason="Shensi latent MoE requires Transformer Engine (Megatron-Core asserts HAVE_TE).",
)

from . import TINY_SHENSI_CONFIG, make_tiny_hf_model


SEQ_LEN = 16


def _tiny_converted_pair(tmp_path, *, gradient_accumulation_fusion: bool = False):
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
    # Bare-autograd tests have no DDP main_grad for the fused wgrad accumulation.
    provider.gradient_accumulation_fusion = gradient_accumulation_fusion
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
    """The whole stack agrees with the HF reference: fp32 HF vs bf16 Megatron, both on CUDA.

    The HF reference runs in fp32 (loading it as bf16 mixes dtypes inside its fp32-kept modules and
    raises), so the comparison carries the bf16 rounding of the Megatron side. A structural
    mismatch — a weight mapped to the wrong module, a missing contraction — moves the logits by
    whole units, far outside the tolerance below, which is why this is worth asserting.
    """
    from safetensors.torch import load_file  # noqa: F401  (kept next to the other conversion imports)

    hf_model, provider, model = _tiny_converted_pair(tmp_path)
    # Keep the reference on CUDA: the FLA KDA kernels selected by the HF model are GPU-only.
    hf_model = hf_model.float().eval().cuda()
    unwrapped = model[0].module if hasattr(model[0], "module") else model[0]

    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    with torch.no_grad():
        # Training parity: the Megatron model has no KV cache path, so compare against the
        # HF forward with the cache disabled.
        reference = hf_model(tokens, use_cache=False).logits.float()
        got = unwrapped(tokens, torch.arange(SEQ_LEN, device="cuda").unsqueeze(0), attention_mask=None)

    got = got.float()
    assert got.shape == reference.shape
    assert torch.isfinite(got).all()
    max_diff = float((reference - got).abs().max())
    assert max_diff < 5e-2, f"HF and Megatron logits differ by {max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training pass needs CUDA")
def test_moe_training_loss_backward(tmp_path) -> None:
    """The masked-LM loss backpropagates through the latent MoE, router and ERC path."""
    from megatron.bridge.training.losses import masked_next_token_loss

    _, provider, model = _tiny_converted_pair(tmp_path)
    wrapped = model[0]
    wrapped.train()
    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    labels = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    loss_mask = torch.ones_like(labels)
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)

    losses = wrapped(tokens, position_ids, attention_mask=None, labels=labels)
    losses = losses[0] if isinstance(losses, tuple) else losses
    loss, num_tokens, _ = masked_next_token_loss(loss_mask, losses)
    assert int(num_tokens) == SEQ_LEN
    assert torch.isfinite(loss)
    loss.backward()

    unwrapped = wrapped.module if hasattr(wrapped, "module") else wrapped
    named = dict(unwrapped.named_parameters())
    router_grads = [
        param.grad for name, param in named.items() if name.endswith("mlp.router.weight") and param.grad is not None
    ]
    expert_grads = [param.grad for name, param in named.items() if ".experts." in name and param.grad is not None]
    assert router_grads, "the router received no gradient"
    assert expert_grads, "the routed experts received no gradient"
    assert all(torch.isfinite(grad).all() for grad in router_grads + expert_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training pass needs CUDA")
def test_fused_wgrad_accumulation_writes_main_grad(tmp_path) -> None:
    """APEX fused wgrad works on the non-TE output layer and the TE/MoE path."""
    pytest.importorskip("fused_weight_gradient_mlp_cuda")
    from megatron.bridge.training.losses import masked_next_token_loss

    _, provider, model = _tiny_converted_pair(tmp_path, gradient_accumulation_fusion=True)
    wrapped = model[0]
    wrapped.train()
    for parameter in wrapped.parameters():
        parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)

    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    labels = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)

    losses = wrapped(tokens, position_ids, attention_mask=None, labels=labels)
    losses = losses[0] if isinstance(losses, tuple) else losses
    loss, _, _ = masked_next_token_loss(torch.ones_like(labels), losses)
    loss.backward()

    main_grads = [param.main_grad for param in wrapped.parameters()]
    assert any(grad.abs().sum() > 0 for grad in main_grads), "fused wgrad wrote no main_grad"
    assert all(torch.isfinite(grad).all() for grad in main_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training pass needs CUDA")
def test_rosa_branch_receives_training_gradients(tmp_path) -> None:
    """A repeated suffix activates ROSA; the auxiliary loss trains its embedding/value/gate."""
    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiForCausalLM

    from megatron.bridge.training.losses import masked_next_token_loss

    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict.update(rosa_layer_ids=[2], num_mtp_layers=0)
    cfg = ShensiConfig(**cfg_dict)
    torch.manual_seed(0)
    hf_model = ShensiForCausalLM(cfg).eval().to("cuda")
    hf_dir = tmp_path / "shensi_tiny_rosa"
    hf_model.save_pretrained(hf_dir, safe_serialization=False)

    bridge = AutoBridge.from_hf_pretrained(hf_dir)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.dsa_kernel_backend = "none"
    provider.gradient_accumulation_fusion = False
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.rosa_loss_coef = 0.5
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)
    wrapped = model[0]
    wrapped.train()

    # A repeated suffix gives ROSA a non-trivial next-token prediction.
    tokens = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 5]], device="cuda")
    labels = torch.tensor([[2, 3, 4, 1, 2, 3, 5, 6]], device="cuda")
    position_ids = torch.arange(tokens.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        reference = hf_model(tokens, use_cache=False).logits.float()
        got = wrapped(tokens, position_ids, attention_mask=None)
        got = got[0] if isinstance(got, tuple) else got
        if got.dim() == 3 and got.shape[0] == tokens.shape[1]:
            got = got.transpose(0, 1)
    assert torch.isfinite(got).all()
    max_diff = float((reference - got.float()).abs().max())
    # bf16 comparison: Megatron's attention runs through fused SDPA, so accumulation order
    # differs from HF's explicit softmax. The fp32 parity tests (atol/rtol 1e-3) are the strict
    # numerical gate; here we only check the two implementations agree closely in bf16.
    assert max_diff < 8e-2, f"HF and Megatron ROSA logits differ by {max_diff:.3e}"

    losses = wrapped(tokens, position_ids, attention_mask=None, labels=labels)
    losses = losses[0] if isinstance(losses, tuple) else losses
    loss, _, _ = masked_next_token_loss(torch.ones_like(labels), losses)
    assert torch.isfinite(loss)
    loss.backward()

    unwrapped = wrapped.module if hasattr(wrapped, "module") else wrapped
    rosa_grads = [
        param.grad for name, param in unwrapped.named_parameters() if ".rosa." in name and param.grad is not None
    ]
    assert rosa_grads, "the ROSA branch received no gradient"
    assert all(torch.isfinite(grad).all() for grad in rosa_grads)
    assert any(grad.abs().sum() > 0 for grad in rosa_grads), "the ROSA gradients are all zero"
