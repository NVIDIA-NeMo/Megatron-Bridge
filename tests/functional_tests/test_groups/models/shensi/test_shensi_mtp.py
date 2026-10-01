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

"""MTP depth on a multi-stream model.

The family keeps the mHC streams inside the decoder and contracts them in the model post-process, so
an MTP depth has to contract the streams of its own prediction layer ahead of the MTP final norm and
the shared head. The tiny geometry here runs multi-stream (hc_mult = 16, four active streams), which
is the shape that used to reach that norm as a flat n-stream tensor.
"""

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from megatron.bridge import AutoBridge
from megatron.bridge.models.shensi.modeling_shensi import ShensiMultiTokenPredictionLayer

from . import make_tiny_hf_model


SEQ_LEN = 16

os.environ.setdefault("TE_FL_PREFER", "vendor")


def _tiny_mtp_checkpoint(tmp_path, mtp_layers: int) -> "Path":
    """Tiny checkpoint carrying the MTP depths the way the released checkpoints do.

    The HF reference keeps the MTP depths out of its module tree (it ignores ``mtp.*`` when it
    loads), so the tiny checkpoint grows a ``mtp.{k}`` subtree here: the wrapper norms, the fused
    embedding-hidden projection and its final norm, plus a copy of the last decoder layer, which is
    what an MTP depth's inner layer holds.
    """
    from . import TINY_SHENSI_CONFIG

    hf_model = make_tiny_hf_model(mtp_layers=mtp_layers)
    hf_dir = tmp_path / "shensi_tiny_mtp"
    hf_model.save_pretrained(hf_dir, safe_serialization=False)
    if not mtp_layers:
        return hf_dir

    hidden = int(TINY_SHENSI_CONFIG["hidden_size"])
    state = dict(load_file(hf_dir / "model.safetensors"))
    # The reference ties the routed experts and the router to the first MoE layer, so the MTP depth
    # takes its inner layer from a layer that carries them.
    prefixes = [f"model.layers.{i}." for i in range(int(TINY_SHENSI_CONFIG["num_hidden_layers"]))]
    source = next((p for p in prefixes if any(k.startswith(p + "mlp.gate.weight") for k in state)), prefixes[-1])
    inner = {key[len(source) :]: tensor.clone() for key, tensor in state.items() if key.startswith(source)}
    assert "mlp.gate.weight" in inner, sorted(inner)
    generator = torch.Generator().manual_seed(0)
    for depth in range(mtp_layers):
        prefix = f"mtp.{depth}."
        state[prefix + "enorm.weight"] = torch.randn(hidden, generator=generator) * 0.02
        state[prefix + "hnorm.weight"] = torch.randn(hidden, generator=generator) * 0.02
        state[prefix + "norm.weight"] = torch.randn(hidden, generator=generator) * 0.02
        state[prefix + "eh_proj.weight"] = torch.randn(hidden, 2 * hidden, generator=generator) * 0.02
        for suffix, tensor in inner.items():
            state[prefix + suffix] = tensor.clone()
    save_file(state, hf_dir / "model.safetensors")
    return hf_dir


def _tiny_mtp_pair(tmp_path, mtp_layers: int = 1):
    """Tiny multi-stream checkpoint with MTP depths → Megatron model with the weights loaded."""
    hf_dir = _tiny_mtp_checkpoint(tmp_path, mtp_layers)

    bridge = AutoBridge.from_hf_pretrained(hf_dir)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.dsa_kernel_backend = "none"
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)
    return bridge, provider, model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="forward pass needs CUDA")
def test_mtp_layer_contracts_the_streams_of_its_prediction_layer(tmp_path) -> None:
    """The MTP depth carries its own stream contract, and a forward plus backward runs."""
    bridge, provider, model = _tiny_mtp_pair(tmp_path)
    assert provider.num_residual_streams > 1, "this test is about the multi-stream MTP path"

    unwrapped = model[0].module if hasattr(model[0], "module") else model[0]
    mtp_layers = list(getattr(unwrapped, "mtp").layers)
    assert len(mtp_layers) == 1, [type(layer).__name__ for layer in mtp_layers]
    mtp_layer = mtp_layers[0]
    assert isinstance(mtp_layer, ShensiMultiTokenPredictionLayer)
    assert getattr(mtp_layer, "stream_contract", None) is not None

    torch.manual_seed(0)
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)

    # The forward reaches the MTP final norm, which the contract has to feed a single hidden state.
    logits = model[0](tokens, position_ids, attention_mask=None)
    assert torch.isfinite(logits).all(), "forward produced NaN/Inf"
    logits.float().sum().backward()
    grads = [p.grad for p in mtp_layer.mtp_model_layer.parameters() if p.grad is not None]
    assert grads, "the MTP depth received no gradient at all"
    assert all(torch.isfinite(grad).all() for grad in grads), "the MTP depth received a non-finite gradient"

    # The MTP weights of the checkpoint reached the mcore model rather than staying at init.
    source = load_file(tmp_path / "shensi_tiny_mtp" / "model.safetensors")
    loaded = unwrapped.mtp.layers[0].enorm.weight.detach().float().cpu()
    assert torch.allclose(loaded, source["mtp.0.enorm.weight"].float(), atol=1e-2)
    assert torch.allclose(
        unwrapped.mtp.layers[0].eh_proj.weight.detach().float().cpu(),
        source["mtp.0.eh_proj.weight"].float(),
        atol=1e-2,
    )

    # And they travel back out under the checkpoint's own names.
    exported = dict(bridge.export_hf_weights(model, cpu=True, show_progress=False))
    assert "mtp.0.norm.weight" in exported and "mtp.0.eh_proj.weight" in exported
    assert torch.allclose(exported["mtp.0.enorm.weight"].float(), source["mtp.0.enorm.weight"].float(), atol=1e-2)

    # The contract turns the prediction layer's flat n-stream output into one hidden state.
    state = mtp_layer.mtp_model_layer.attn_res_state
    streams = torch.randn(SEQ_LEN, 1, provider.num_residual_streams * provider.hidden_size, device="cuda")
    contracted = mtp_layer.stream_contract(streams)
    assert contracted.shape == (SEQ_LEN, 1, provider.hidden_size)
    assert torch.isfinite(contracted).all()
    assert state.prefix_sum is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="forward pass needs CUDA")
def test_mtp_off_keeps_the_decoder_output_flat(tmp_path) -> None:
    """Without MTP depths the model has no MTP block to contract for."""
    _, provider, model = _tiny_mtp_pair(tmp_path, mtp_layers=0)
    assert getattr(model[0].module if hasattr(model[0], "module") else model[0], "mtp", None) is None
    tokens = torch.randint(0, provider.vocab_size, (1, SEQ_LEN), device="cuda")
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0)
    with torch.no_grad():
        logits = model[0](tokens, position_ids, attention_mask=None)
    assert torch.isfinite(logits).all()
