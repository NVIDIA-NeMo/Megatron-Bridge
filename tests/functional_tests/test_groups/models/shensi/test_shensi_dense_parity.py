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

"""Dense-only HF <-> Megatron logits parity.

The latent MoE path needs Transformer Engine, but the stream loop, indexed attention, dense MLP,
hyper-connections and attention residuals do not. This test keeps every layer dense and builds a
model with Megatron-Core's local backend when TE is absent, so the core forward stays covered.
"""

import pytest
import torch

from megatron.bridge import AutoBridge

from . import TINY_SHENSI_CONFIG


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Megatron model is built on CUDA")
def test_dense_only_hf_matches_megatron(tmp_path):
    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiForCausalLM

    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict.update(
        num_hidden_layers=4,
        num_mtp_layers=0,
        mlp_layer_types=["dense"] * 4,
        first_k_dense_replace=4,
        layer_types=["indexed_attention"] * 4,
        attn_res_block_layer_types=["block_write_layer"] * 4,
        attn_res_block_size=1,
        rosa_layer_ids=[],
        indexer_types=None,
    )
    cfg = ShensiConfig(**cfg_dict)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hf_model = ShensiForCausalLM(cfg).eval().to(device)
    hf_model.save_pretrained(tmp_path, safe_serialization=False)

    bridge = AutoBridge.from_hf_pretrained(tmp_path)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    # Structural parity: keep the Megatron side in fp32 like the HF reference.
    provider.bf16 = False
    provider.fp16 = False
    provider.params_dtype = torch.float32
    # Bare-autograd tests have no DDP main_grad for the fused wgrad accumulation.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()
    bridge.load_hf_weights([megatron_model])

    tokens = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, 8), dtype=torch.bool, device=device)
    with torch.no_grad():
        reference = hf_model(tokens, attention_mask=attention_mask, use_cache=False).logits.float()
        logits = megatron_model(tokens, position_ids, attention_mask=attention_mask)
        logits = logits[0] if isinstance(logits, tuple) else logits
        if logits.dim() == 3 and logits.shape[0] == 8:
            logits = logits.transpose(0, 1)
    assert torch.isfinite(logits).all()
    # The local backend matches exactly; Transformer Engine's fp32 GEMMs default to TF32, so
    # allow the same 1e-3 budget as the mixed-attention test when TE is installed.
    torch.testing.assert_close(logits.float(), reference, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Megatron model is built on CUDA")
def test_indexed_linear_mixed_with_rosa_matches_hf(tmp_path):
    """KDA linear attention, ROSA and a shared indexer across linear layers must match HF."""
    pytest.importorskip("fla")

    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiForCausalLM

    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict.update(
        num_hidden_layers=4,
        num_mtp_layers=0,
        mlp_layer_types=["dense"] * 4,
        first_k_dense_replace=4,
        attn_res_block_layer_types=[
            "block_write_layer",
            "block_read_layer",
            "block_write_layer",
            "block_read_layer",
        ],
        layer_types=["indexed_attention", "linear_attention", "indexed_attention", "linear_attention"],
        rosa_layer_ids=[1, 3],
        rosa_reset_token_id=2,
        indexer_types=None,
    )
    cfg = ShensiConfig(**cfg_dict)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hf_model = ShensiForCausalLM(cfg).eval().to(device)
    hf_model.save_pretrained(tmp_path, safe_serialization=False)

    bridge = AutoBridge.from_hf_pretrained(tmp_path)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    # Structural parity: keep the Megatron side in fp32 like the HF reference.
    provider.bf16 = False
    provider.fp16 = False
    provider.params_dtype = torch.float32
    # Bare-autograd tests have no DDP main_grad for the fused wgrad accumulation.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)[0].eval()
    bridge.load_hf_weights([megatron_model])

    tokens = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, 8), dtype=torch.bool, device=device)
    with torch.no_grad():
        reference = hf_model(tokens, attention_mask=attention_mask, use_cache=False).logits.float()
        logits = megatron_model(tokens, position_ids, attention_mask=attention_mask)
        logits = logits[0] if isinstance(logits, tuple) else logits
        if logits.dim() == 3 and logits.shape[0] == 8:
            logits = logits.transpose(0, 1)
    assert torch.isfinite(logits).all()
    # HF computes KDA with its pure-PyTorch chunk kernel while Megatron-Core uses FLA.
    torch.testing.assert_close(logits.float(), reference, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Megatron model is built on CUDA")
def test_dense_only_training_loss_backward(tmp_path):
    """The language-model loss path produces finite gradients through the whole dense model."""
    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiForCausalLM

    from megatron.bridge.training.losses import masked_next_token_loss

    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict.update(
        num_hidden_layers=4,
        num_mtp_layers=0,
        mlp_layer_types=["dense"] * 4,
        first_k_dense_replace=4,
        layer_types=["indexed_attention"] * 4,
        attn_res_block_layer_types=["block_write_layer"] * 4,
        attn_res_block_size=1,
        rosa_layer_ids=[],
        indexer_types=None,
    )
    cfg = ShensiConfig(**cfg_dict)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hf_model = ShensiForCausalLM(cfg).eval().to(device)
    hf_model.save_pretrained(tmp_path, safe_serialization=False)

    bridge = AutoBridge.from_hf_pretrained(tmp_path)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    # Structural parity: keep the Megatron side in fp32 like the HF reference.
    provider.bf16 = False
    provider.fp16 = False
    provider.params_dtype = torch.float32
    # Bare-autograd tests have no DDP main_grad for the fused wgrad accumulation.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0]
    bridge.load_hf_weights([model])
    model.train()

    tokens = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    labels = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, 8), dtype=torch.bool, device=device)
    loss_mask = torch.ones((1, 8), dtype=torch.int, device=device)

    losses = model(tokens, position_ids, attention_mask=attention_mask, labels=labels)
    losses = losses[0] if isinstance(losses, tuple) else losses
    loss, num_tokens, _ = masked_next_token_loss(loss_mask, losses)
    assert int(num_tokens) == 8
    assert torch.isfinite(loss)
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads, "the training loss produced no gradients"
    assert all(torch.isfinite(grad).all() for grad in grads)
