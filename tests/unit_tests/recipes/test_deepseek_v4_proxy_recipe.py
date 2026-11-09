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

"""Shape and training-contract guardrails for the DeepSeek-V4-Flash six-layer proxy recipe."""

import pytest

import megatron.bridge.recipes as recipes
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_library_config as flash_library_config,
)
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_proxy_pretrain_8gpu_gb200_bf16_config as flash_proxy_config,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import (
    DeepSeekV4BuilderOnlyBridge,
    deepseek_v4_model_config,
    patch_recipe_module_global,
)


pytestmark = pytest.mark.unit

_PROXY_GPUS = 8
# The proxy checkpoint's Hugging Face config: DeepSeek-V4-Flash cut to its first six layers and the MTP layer.
_PROXY_HF_OVERRIDES = {"num_hidden_layers": 6, "compress_ratios": [0, 0, 4, 128, 4, 128, 0]}


def _build(recipe, monkeypatch: pytest.MonkeyPatch, *, fused_mhc_supported: bool = True):
    patch_recipe_module_global(monkeypatch, recipe, "AutoBridge", DeepSeekV4BuilderOnlyBridge)
    patch_recipe_module_global(
        monkeypatch, recipe, "deepseek_v4_supports_blackwell_fused_kernels", lambda: fused_mhc_supported
    )
    return recipe()


def test_proxy_matches_the_cut_down_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build(flash_proxy_config, monkeypatch)
    cut = deepseek_v4_model_config("deepseek-ai/DeepSeek-V4-Flash", **_PROXY_HF_OVERRIDES)
    full = deepseek_v4_model_config("deepseek-ai/DeepSeek-V4-Flash")

    # The layers the bridge builds from the cut-down checkpoint, so the import loads without remapping.
    for field in (
        "hybrid_layer_pattern",
        "num_layers",
        "csa_compress_ratios",
        "mtp_num_layers",
        "mtp_hybrid_override_pattern",
        "moe_num_hash_layers",
    ):
        assert getattr(cfg.model, field) == getattr(cut, field), field
    assert cfg.model.hybrid_layer_pattern == "WEWECEHECEHE"
    assert cfg.model.csa_compress_ratios[: cfg.model.num_layers] == full.csa_compress_ratios[:12]

    # Every width, routing and vocabulary setting stays DeepSeek-V4-Flash's.
    for field in (
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "v_head_dim",
        "qk_pos_emb_head_dim",
        "output_projection_groups",
        "output_projection_lora_rank",
        "csa_window_size",
        "dsa_indexer_n_heads",
        "dsa_indexer_head_dim",
        "dsa_indexer_topk",
        "mhc_num_residual_streams",
        "mhc_sinkhorn_iterations",
        "num_moe_experts",
        "moe_ffn_hidden_size",
        "moe_shared_expert_intermediate_size",
        "moe_router_topk",
        "moe_router_topk_scaling_factor",
        "moe_router_score_function",
        "moe_router_load_balancing_type",
        "moe_router_enable_expert_bias",
        "hash_moe_vocab_size",
        "activation_func_clamp_value",
        "vocab_size",
    ):
        assert getattr(cfg.model, field) == getattr(full, field), field


@pytest.mark.parametrize("fused_mhc_supported", [True, False])
def test_proxy_validates_on_one_pipeline_stage(fused_mhc_supported: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build(flash_proxy_config, monkeypatch, fused_mhc_supported=fused_mhc_supported)

    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.pipeline_model_parallel_layout is None
    assert "|" not in cfg.model.hybrid_layer_pattern
    # Fused mHC rejects mhc_norm_eps_inside_sqrt, which only finalize() checks.
    assert cfg.model.use_fused_mhc is fused_mhc_supported
    assert cfg.model.mhc_norm_eps_inside_sqrt is not fused_mhc_supported
    cfg.model.validate()
    # The cuDNN DSA kernels need a Blackwell device at finalize(); the shape and mHC checks do not.
    cfg.model.dsa_kernel_backend = "none"
    cfg.model.finalize()


def test_proxy_fits_its_eight_gpu_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build(flash_proxy_config, monkeypatch)
    model = cfg.model

    dense_ranks = model.tensor_model_parallel_size * model.pipeline_model_parallel_size * model.context_parallel_size
    expert_ranks = (
        model.pipeline_model_parallel_size * model.expert_model_parallel_size * model.expert_tensor_parallel_size
    )
    assert (model.tensor_model_parallel_size, model.expert_model_parallel_size) == (1, 8)
    assert _PROXY_GPUS % dense_ranks == 0 and _PROXY_GPUS % expert_ranks == 0
    assert model.num_moe_experts % model.expert_model_parallel_size == 0
    data_parallel_size = _PROXY_GPUS // dense_ranks
    assert cfg.train.global_batch_size % (cfg.train.micro_batch_size * data_parallel_size) == 0
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == model.expert_model_parallel_size


def test_proxy_keeps_the_library_training_contract_in_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _build(flash_proxy_config, monkeypatch)
    library = _build(flash_library_config, monkeypatch)

    # Optimizer, schedule and batch: the library recipe's.
    for field in (
        "optimizer",
        "lr",
        "min_lr",
        "weight_decay",
        "clip_grad",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "use_precision_aware_optimizer",
        "main_grads_dtype",
        "main_params_dtype",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
    ):
        assert getattr(cfg.optimizer, field) == getattr(library.optimizer, field), field
    assert cfg.scheduler == library.scheduler
    assert (cfg.train.global_batch_size, cfg.train.micro_batch_size) == (256, 1)
    assert cfg.train.train_iters == library.train.train_iters
    assert cfg.model.seq_length == cfg.dataset.seq_length == 4096

    # Natural routing with the DeepSeek-V4 objectives: MTP, the DSA indexer loss, no MoE auxiliary loss.
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.moe_aux_loss_coeff == 0.0
    assert getattr(cfg.model, "moe_expert_rank_capacity_factor", None) is None
    assert getattr(cfg.model, "moe_paged_stash", False) is False
    assert cfg.model.dsa_indexer_loss_coeff == 0.01
    assert cfg.model.dsa_indexer_use_sparse_loss is True
    assert cfg.model.mtp_num_layers == 1

    # BF16 instead of the library recipe's MXFP8.
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp8 is None
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert getattr(cfg.model, "quant_recipe", None) is None
    assert cfg.model.moe_router_padding_for_fp8 is False
    assert cfg.model.use_transformer_engine_op_fuser is False
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in cfg.env_vars

    # The library recipe's execution path on one pipeline stage.
    for field in (
        "dsa_kernel_backend",
        "moe_token_dispatcher_type",
        "moe_flex_dispatcher_backend",
        "moe_flex_dispatcher_num_sms",
        "moe_hybridep_num_sms_preprocessing",
        "moe_shared_expert_overlap",
        "moe_grouped_gemm",
        "moe_permute_fusion",
        "moe_router_fusion",
        "cross_entropy_fusion_impl",
        "recompute_granularity",
        "recompute_modules",
        "fine_grained_activation_offloading",
        "cuda_graph_impl",
    ):
        assert getattr(cfg.model, field) == getattr(library.model, field), field
    assert cuda_graph_module_names(cfg.model) == []
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True

    # DeepSeek-V4 tokens over the checkpoint's own vocabulary, which the hash-routing tables index.
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == "deepseek-ai/DeepSeek-V4-Flash"
    assert cfg.tokenizer.use_tokenizer_vocab_size is False
    assert cfg.model.vocab_size == cfg.model.hash_moe_vocab_size == 129280


def test_proxy_recipe_is_exported() -> None:
    assert recipes.deepseek_v4_flash_proxy_pretrain_8gpu_gb200_bf16_config is flash_proxy_config
    assert recipes.deepseek.deepseek_v4_flash_proxy_pretrain_8gpu_gb200_bf16_config is flash_proxy_config
