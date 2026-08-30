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

import importlib

import pytest
import torch

from megatron.bridge.models.bailing.bailing_moe3_provider import BailingMoe3HybridProvider
from megatron.bridge.models.bailing.bailing_moe3_spec import bailing_moe3_hybrid_stack_spec
from megatron.bridge.recipes.bailing.h100.ling_v3_tiny import (
    LING_V3_TINY_MTP_PATTERN,
    ling_v3_tiny_pretrain_8gpu_h100_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_TINY_MAIN_PATTERN = "K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E"
_AUTOBRIDGE_ARCHITECTURE = {
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "kv_channels": 64,
    "ffn_hidden_size": 2048,
    "vocab_size": 4096,
    "normalization": "RMSNorm",
    "layernorm_epsilon": 1.0e-5,
    "gated_linear_unit": True,
    "add_bias_linear": False,
    "add_qkv_bias": False,
    "share_embeddings_and_output_weights": False,
    "position_embedding_type": "rope",
    "rotary_percent": 1.0,
    "rotary_base": 123_456,
    "num_moe_experts": 32,
    "moe_ffn_hidden_size": 256,
    "moe_shared_expert_intermediate_size": 256,
    "moe_router_topk": 4,
    "moe_router_score_function": "sigmoid",
    "moe_router_dtype": "fp32",
    "moe_router_topk_scaling_factor": 1.5,
    "moe_router_num_groups": 4,
    "moe_router_group_topk": 2,
    "moe_router_enable_expert_bias": True,
    "moe_router_load_balancing_type": "none",
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 8,
    "kda_safe_gate": False,
    "kda_lower_bound": None,
    "multi_latent_attention": True,
    "q_lora_rank": 128,
    "kv_lora_rank": 256,
    "qk_head_dim": 64,
    "qk_pos_emb_head_dim": 32,
    "v_head_dim": 64,
    "qk_layernorm": True,
    "attention_output_gate": True,
    "gated_attention_proj_granularity": "headwise",
}


class _FakeAutoBridge:
    """Fake AutoBridge that avoids network access during recipe unit tests."""

    @classmethod
    def from_hf_pretrained(cls, *args: object, **kwargs: object) -> "_FakeAutoBridge":
        del args, kwargs
        return cls()

    def to_megatron_provider(self, *args: object, **kwargs: object) -> BailingMoe3HybridProvider:
        del args, kwargs
        return BailingMoe3HybridProvider(
            hybrid_layer_pattern=_TINY_MAIN_PATTERN,
            num_layers=len(_TINY_MAIN_PATTERN),
            hybrid_stack_spec=bailing_moe3_hybrid_stack_spec,
            **_AUTOBRIDGE_ARCHITECTURE,
        )


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the canonical recipe module's AutoBridge dependency."""
    module = importlib.import_module("megatron.bridge.recipes.bailing.h100.ling_v3_tiny")
    patch_recipe_module_global(monkeypatch, module, "AutoBridge", _FakeAutoBridge)


def test_ling_v3_tiny_recipe_returns_complete_config() -> None:
    """The recipe exposes all containers needed by the pretraining runner."""
    config = ling_v3_tiny_pretrain_8gpu_h100_bf16_config()

    assert isinstance(config, ConfigContainer)
    assert config.model is not None
    assert config.train is not None
    assert config.optimizer is not None
    assert config.scheduler is not None
    assert config.dataset is not None
    assert config.tokenizer is not None
    assert config.checkpoint is not None
    assert set(vars(config.model)) <= set(config.model.__dataclass_fields__)


def test_ling_v3_tiny_recipe_preserves_autobridge_topology_and_adds_mtp() -> None:
    """The recipe preserves AutoBridge topology and adds only its training MTP depth."""
    config = ling_v3_tiny_pretrain_8gpu_h100_bf16_config()
    model = config.model

    assert model.hybrid_layer_pattern == _TINY_MAIN_PATTERN
    assert model.mtp_hybrid_override_pattern == LING_V3_TINY_MTP_PATTERN
    assert model.num_layers == len(_TINY_MAIN_PATTERN)
    assert model.mtp_num_layers == 1
    assert model.hybrid_stack_spec is bailing_moe3_hybrid_stack_spec
    assert model.params_dtype == torch.bfloat16
    assert model.bf16 is True
    assert model.fp16 is False


@pytest.mark.parametrize(("name", "expected"), _AUTOBRIDGE_ARCHITECTURE.items())
def test_ling_v3_tiny_recipe_does_not_override_autobridge_architecture(name: str, expected: object) -> None:
    """Architecture fields remain owned by the provider returned from AutoBridge."""
    model = ling_v3_tiny_pretrain_8gpu_h100_bf16_config().model

    assert getattr(model, name) == expected


def test_ling_v3_tiny_recipe_matches_parallelism_and_runtime_policy() -> None:
    """The recipe carries the eight-GPU smoke parallelism and runtime policy."""
    config = ling_v3_tiny_pretrain_8gpu_h100_bf16_config()
    model = config.model

    assert model.tensor_model_parallel_size == 1
    assert model.pipeline_model_parallel_size == 1
    assert model.pipeline_dtype is None
    assert model.virtual_pipeline_model_parallel_size is None
    assert model.expert_model_parallel_size == 8
    assert model.context_parallel_size == 2
    assert model.sequence_parallel is False
    assert model.cp_comm_type == "p2p"
    assert model.linear_cp_mode == "headwise"
    assert model.moe_router_bias_update_rate == 0.0
    assert model.moe_z_loss_coeff == 2.9e-6
    assert model.moe_token_dispatcher_type == "alltoall"
    assert model.moe_grouped_gemm is True
    assert model.transformer_impl == "transformer_engine"
    assert model.attention_backend is None


def test_ling_v3_tiny_recipe_matches_training_smoke_settings() -> None:
    """The recipe keeps the intentional MTP and short deterministic smoke settings."""
    config = ling_v3_tiny_pretrain_8gpu_h100_bf16_config()
    model = config.model

    assert model.mtp_loss_scaling_factor == 0.1
    assert model.recompute_granularity == "full"
    assert model.recompute_method == "uniform"
    assert model.recompute_num_layers == 1
    assert config.dataset.blend is None
    assert config.dataset.seq_length == 128
    assert config.tokenizer.vocab_size == model.vocab_size
    assert config.train.train_iters == 2
    assert config.train.global_batch_size == 8
    assert config.train.micro_batch_size == 1
    assert config.validation.eval_iters == 0
    assert config.checkpoint.save is None
    assert config.checkpoint.load is None
    assert config.optimizer is not None
    assert config.scheduler is not None
    assert config.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1
