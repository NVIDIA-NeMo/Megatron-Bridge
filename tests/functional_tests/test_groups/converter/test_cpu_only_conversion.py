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

from contextlib import suppress

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def _destroy_distributed_state() -> None:
    with suppress(Exception):
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test_tiny_llama_cpu_only_auto_bridge_conversion(tmp_path, monkeypatch):
    """Convert a tiny local HF Llama model to Megatron with CUDA hidden."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert not torch.cuda.is_available()

    model_dir = tmp_path / "tiny_llama"
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(str(model_dir), torch_dtype=torch.float32)
    try:
        megatron_model = bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)
        assert len(megatron_model) == 1
        assert next(megatron_model[0].parameters()).device.type == "cpu"
    finally:
        _destroy_distributed_state()
