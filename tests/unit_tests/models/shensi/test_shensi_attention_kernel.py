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

"""The fused attention path at a sequence length where the fused kernels are selected."""

import torch

from .test_shensi_train_loop import _provider, distributed  # noqa: F401 - fixtures


def test_attention_forward_backward_at_training_length(distributed):
    """A 1k-token bf16 pass exercises the fused kernel (a 1-head KV must be broadcast).

    The tiny functional tests run 8-16 tokens, where the math backend hides the fact that the
    fused kernels reject a key/value head count of one; this pins the training-length path.
    """
    from megatron.core.transformer.module import Float16Module

    from megatron.bridge.models.shensi.modeling_shensi import ShensiIndexedAttention

    provider = _provider(0)
    # Bare autograd (no DDP buffer): fused wgrad needs `main_grad`, so keep it off here.
    provider.gradient_accumulation_fusion = False
    # The training stack wraps the module: Float16Module casts the bridge-owned fp32 modules
    # (embeddings, nn.Linear) to the parameter dtype before the TE linears see them.
    module = Float16Module(provider, ShensiIndexedAttention(provider, layer_number=1)).cuda().train()

    tokens = torch.randint(1, provider.vocab_size, (1, 1024), device="cuda")
    hidden = torch.randn(1024, 1, provider.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output, _ = module(hidden, attention_mask=None, input_ids=tokens)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()

    output.float().square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
