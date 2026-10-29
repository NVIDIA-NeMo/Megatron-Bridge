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

"""Functional test for LLaDA1.5: real Megatron GPTModel forward + block-diffusion generate.

Unlike the unit tests (which mock the model), this builds a **tiny, randomly
initialized** LLaDA1.5 Megatron model on GPU and exercises the *real*
``LLaDA15TEDotProductAttention`` (Transformer Engine kernel, bidirectional
``no_mask`` and padding ``arbitrary`` mask paths) and the *real*
``generate_block_diffusion`` loop end-to-end.

No checkpoint and no HuggingFace download — the model is constructed directly
from a toy ``LLaDA15ModelProvider`` config with random weights. Single GPU.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.diffusion.models.llada15.inference_llada15 import _unwrap, generate_block_diffusion
from megatron.bridge.diffusion.models.llada15.llada15_attention import LLaDA15TEDotProductAttention
from megatron.bridge.diffusion.models.llada15.llada15_provider import LLaDA15ModelProvider


# Tiny toy dims — small enough to build and run quickly on a single GPU.
TOY = dict(
    num_layers=2,
    hidden_size=128,
    num_attention_heads=4,
    num_query_groups=4,
    kv_channels=32,
    ffn_hidden_size=256,
    vocab_size=256,
    seq_length=64,
)
MASK_ID = 255  # within toy vocab; argmax over random logits rarely lands here exactly


def _init_distributed():
    """Initialize a single-rank process group + model parallel for in-process testing.

    Sets torchrun env defaults so the test runs under a plain ``pytest`` launcher
    (no external ``torchrun`` wrapper needed).
    """
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(42)


def _build_toy_model():
    provider = LLaDA15ModelProvider(
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=False,
        rotary_base=500000.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        **TOY,
    )
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)
    return model[0].eval() if isinstance(model, list) else model.eval()


@pytest.mark.run_only_on("GPU")
class TestLLaDA15ToyGenerate:
    """End-to-end GPU smoke of the real attention + generation, no checkpoint."""

    def test_attention_is_te_backed(self):
        _init_distributed()
        model = _build_toy_model()
        # Every layer's core attention must be the TE-backed LLaDA1.5 attention.
        # _unwrap drops the Float16Module/DDP wrapper added by provide_distributed_model.
        for layer in _unwrap(model).decoder.layers:
            assert isinstance(layer.self_attention.core_attention, LLaDA15TEDotProductAttention)

    def test_forward_shape(self):
        _init_distributed()
        model = _build_toy_model()
        ids = torch.randint(0, TOY["vocab_size"], (1, 16), device="cuda")
        pos = torch.arange(16, device="cuda").unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=ids, position_ids=pos, attention_mask=None)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        assert logits.shape[0] == 1 and logits.shape[1] == 16
        assert logits.shape[2] >= TOY["vocab_size"]  # vocab may be padded for TP

    def test_generate_shape_and_prompt_preserved(self):
        _init_distributed()
        model = _build_toy_model()
        prompt = torch.randint(0, TOY["vocab_size"] - 1, (1, 4), device="cuda")
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=8,
            block_length=4,
            steps=8,
            mask_token_id=MASK_ID,
            eos_token_id=None,
        )
        assert out.shape == (1, 4 + 8)
        assert torch.equal(out[0, :4], prompt[0])
        # All generated positions should be unmasked.
        assert int((out[:, 4:] == MASK_ID).sum()) == 0

    def test_generate_batched_with_padding_mask(self):
        """Real arbitrary-mask path: a left/right-padded batch must run without corruption-by-shape."""
        _init_distributed()
        model = _build_toy_model()
        # Two prompts, different lengths -> padding present; pad id within vocab.
        pad_id = 0
        prompt = torch.tensor([[1, 2, 3, 4], [5, 6, pad_id, pad_id]], device="cuda")
        out = generate_block_diffusion(
            model,
            prompt,
            gen_length=4,
            block_length=4,
            steps=4,
            mask_token_id=MASK_ID,
            eos_token_id=None,
            pad_token_id=pad_id,
        )
        assert out.shape == (2, 4 + 4)
        # Generated region fully unmasked for both rows.
        assert int((out[:, 4:] == MASK_ID).sum()) == 0
