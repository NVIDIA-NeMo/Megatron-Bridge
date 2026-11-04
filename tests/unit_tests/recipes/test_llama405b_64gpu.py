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

from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.llama import llama31_405b_pretrain_64gpu_gb200_fp8mx_config
from megatron.bridge.recipes.llama.h100.llama3 import llama31_405b_pretrain_256gpu_h100_bf16_config
from megatron.bridge.training.config import runtime_config_update


@pytest.fixture
def offline_llama405b():
    """Construct the real model configuration without downloading weights or metadata."""
    hf_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=126,
        hidden_size=16384,
        intermediate_size=53248,
        num_attention_heads=128,
        num_key_value_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        tie_word_embeddings=False,
        rope_scaling={
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )
    with patch.object(
        AutoBridge, "from_hf_pretrained", side_effect=lambda *a, **k: AutoBridge.from_hf_config(hf_config)
    ):
        yield


@pytest.mark.unit
def test_llama405b_64gpu_runtime_contract(offline_llama405b, monkeypatch):
    """Finalization must preserve the full model, precision, and measured execution settings."""
    monkeypatch.setenv("WORLD_SIZE", "64")
    monkeypatch.setenv("RANK", "0")
    cfg = llama31_405b_pretrain_64gpu_gb200_fp8mx_config()
    runtime_config_update(cfg)
    model = cfg.model
    assert (model.num_layers, model.hidden_size, model.ffn_hidden_size) == (126, 16384, 53248)
    assert (model.num_attention_heads, model.num_query_groups, model.vocab_size) == (128, 8, 128256)
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (8, 8)
    assert (model.virtual_pipeline_model_parallel_size, model.context_parallel_size, cfg.data_parallel_size) == (
        8,
        1,
        1,
    )
    assert (model.seq_length, cfg.dataset.seq_length) == (8192, 8192)
    assert (cfg.train.global_batch_size, cfg.train.micro_batch_size) == (1536, 1)
    assert model.sequence_parallel
    assert model.account_for_embedding_in_pipeline_split and model.account_for_loss_in_pipeline_split
    assert model.recompute_granularity is None
    assert model.recompute_modules == []
    assert model.recompute_method is None and model.recompute_num_layers is None
    assert not model.cpu_offloading and not model.fine_grained_activation_offloading
    assert not cfg.optimizer.optimizer_cpu_offload
    assert model.fp8_recipe == "mxfp8" and model.fp8 == "e4m3" and model.fp8_param
    assert not model.first_last_layers_bf16
    assert cfg.ddp.fp8_param_gather
    assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag
    assert model.params_dtype == torch.bfloat16
    assert model.cuda_graph_impl == "none" and model.cuda_graph_modules == []
    assert not model.enable_cuda_graph and not model.external_cuda_graph
    assert not cfg.optimizer.optimizer_cuda_graph
    assert cfg.rng.te_rng_tracker and model.use_te_rng_tracker
    assert not cfg.ddp.grad_reduce_in_fp32
    assert model.tp_comm_overlap
    assert cfg.comm_overlap.tp_comm_overlap_cfg is None


@pytest.mark.unit
def test_llama405b_64gpu_retains_training_defaults(offline_llama405b):
    """A capacity recipe must retain checkpointing, checks, and the training schedule."""
    baseline = llama31_405b_pretrain_256gpu_h100_bf16_config()
    cfg = llama31_405b_pretrain_64gpu_gb200_fp8mx_config()
    assert cfg.train.train_iters == baseline.train.train_iters
    assert cfg.scheduler == baseline.scheduler
    assert cfg.checkpoint == baseline.checkpoint
    assert cfg.validation == baseline.validation
    assert cfg.optimizer == baseline.optimizer
    assert cfg.ddp.check_for_nan_in_grad
    assert cfg.rerun_state_machine.check_for_nan_in_loss
    assert cfg.checkpoint.save is not None and cfg.checkpoint.load is not None
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.model.rope_scaling == baseline.model.rope_scaling
    assert cfg.model.rope_scaling_factor == baseline.model.rope_scaling_factor
    assert cfg.model.rotary_base == baseline.model.rotary_base
    assert not cfg.tokenizer.use_tokenizer_vocab_size
