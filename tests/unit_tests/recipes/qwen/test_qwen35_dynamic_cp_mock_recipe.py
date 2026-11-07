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

"""Tests for the four-GPU Qwen3.5-35B-A3B dynamic-CP recipe on mock variable-length data."""

import pytest
import torch
from transformers import Qwen3_5MoeConfig

from megatron.bridge.data.builders import MockGPTSFTDatasetConfig
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.qwen.gb200 import qwen35
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

# Text configuration of Qwen/Qwen3.5-35B-A3B at the recipe's pinned revision.
_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "moe_intermediate_size": 512,
    "mtp_num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 512,
    "vocab_size": 248320,
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
}


@pytest.fixture
def recipe(monkeypatch: pytest.MonkeyPatch):
    """Build the recipe through the real Qwen3.5 bridge on the published 35B-A3B config, offline."""
    patch_recipe_construction_dependencies(monkeypatch)
    monkeypatch.setattr(qwen35, "AutoBridge", AutoBridge)
    requested = {}

    def from_pretrained(path, **kwargs):
        requested.update(path=path, **kwargs)
        return Qwen3_5MoeConfig(text_config=_TEXT_CONFIG, tie_word_embeddings=False)

    monkeypatch.setattr(qwen35.AutoConfig, "from_pretrained", from_pretrained)
    cfg = qwen35.qwen35_text_35b_a3b_4gpu_gb200_fp8mx_dynamic_cp_config()
    return cfg, requested


def test_model_comes_from_the_pinned_hf_config_through_the_qwen35_bridge(recipe):
    cfg, requested = recipe

    assert requested == {"path": "Qwen/Qwen3.5-35B-A3B", "revision": qwen35._QWEN35_35B_A3B_INSTRUCT_REVISION}
    model = cfg.model
    expected = {
        "num_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_query_groups": 2,
        "kv_channels": 256,
        "experimental_attention_variant": "gdn",
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "normalization": "RMSNorm",
        "layernorm_epsilon": 1e-6,
        "layernorm_zero_centered_gamma": True,
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "qk_layernorm": True,
        "attention_output_gate": True,
        "num_moe_experts": 256,
        "moe_ffn_hidden_size": 512,
        "moe_router_topk": 8,
        "moe_shared_expert_intermediate_size": 512,
        "moe_shared_expert_gate": True,
        "moe_router_load_balancing_type": "global_aux_loss",
        "moe_aux_loss_coeff": 0.001,
        "moe_router_dtype": "fp32",
        "vocab_size": 248320,
        "position_embedding_type": "rope",
        "rotary_base": 10000000,
        "rotary_percent": 0.25,
        "share_embeddings_and_output_weights": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "mtp_num_layers": 1,
        "mtp_loss_scaling_factor": 0.1,
    }
    assert {name: getattr(model, name) for name in expected} == expected
    assert model.linear_attention_freq == [1, 1, 1, 0] * 10


def test_parallelism_and_dynamic_context_parallelism(recipe):
    cfg, _ = recipe
    model = cfg.model

    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (1, 1)
    assert (model.context_parallel_size, model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (
        4,
        4,
        1,
    )
    assert model.seq_length == 65536
    assert model.max_seqlen_per_dp_cp_rank == 16384
    assert model.dynamic_context_parallel is True
    assert model.min_dynamic_context_parallel_size == 1
    # The scheduler is selected at validation from the dataset switch.
    assert getattr(model, "sequence_packing_scheduler", None) is None
    assert model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert (cfg.train.micro_batch_size, cfg.train.global_batch_size) == (1, 32)


def test_data_is_mock_variable_length_sft_without_a_tokenizer(recipe):
    cfg, _ = recipe

    assert isinstance(cfg.dataset, MockGPTSFTDatasetConfig)
    assert cfg.dataset.seq_length == 65536
    assert cfg.dataset.vocab_size == cfg.model.vocab_size
    assert cfg.dataset.length_bucket_fractions == (0.5, 0.3, 0.2)
    assert cfg.dataset.yields_unpacked_samples is True
    assert cfg.dataset.fold_alignment_padding is True
    assert cfg.dataset.dataloader_type == "single"
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.tokenizer.vocab_size == cfg.model.vocab_size
    assert cfg.validation.eval_iters == 0


def test_training_numerics_match_the_original_recipe(recipe):
    cfg, _ = recipe

    assert (cfg.optimizer.lr, cfg.optimizer.min_lr, cfg.optimizer.weight_decay) == (2e-5, 2e-6, 0.033)
    assert (cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2, cfg.optimizer.adam_eps) == (0.9, 0.95, 1e-8)
    assert cfg.optimizer.clip_grad == 1.0
    assert (cfg.scheduler.lr_warmup_iters, cfg.scheduler.lr_decay_iters, cfg.scheduler.lr_decay_style) == (
        10,
        100,
        "cosine",
    )
    for name in ("main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert getattr(cfg.optimizer, name) == torch.float32
    assert cfg.ddp.grad_reduce_in_fp32 is True


def test_performance_settings(recipe):
    cfg, _ = recipe
    model = cfg.model

    assert cfg.optimizer.optimizer_cpu_offload is True and cfg.optimizer.optimizer_offload_fraction == 1.0
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    # BF16 parameter storage and gather while the optimizer lives on the host.
    assert cfg.mixed_precision.fp8_param_gather is False
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is False
    assert model.moe_token_dispatcher_type == "flex" and model.moe_flex_dispatcher_backend == "hybridep"
    assert model.moe_flex_dispatcher_num_sms == 32 and model.moe_hybridep_pad_uneven_dispatch_inputs is True
    assert (model.recompute_granularity, model.recompute_method, model.recompute_num_layers) == ("full", "uniform", 1)
    assert cfg.ddp.overlap_grad_reduce is True and cfg.ddp.overlap_param_gather is True
    assert cfg.train.cuda_memory_fraction == 0.9
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 4
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1


def test_nothing_is_saved_or_loaded_by_default(recipe):
    cfg, _ = recipe

    assert cfg.checkpoint.save is None and cfg.checkpoint.load is None
    assert cfg.checkpoint.pretrained_checkpoint is None
