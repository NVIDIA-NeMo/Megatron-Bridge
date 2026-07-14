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
"""Standalone model and performance defaults for NeMoDiag V0."""

from functools import partial

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.perf_recipes._common import (
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
)
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


NEMODIAG_V0_PIPELINE_LAYOUT = "Et*4|(t*4|)*6t*3mL"


def set_nemodiag_v0_pipeline_model_parallel_layout(
    model_cfg: MLAModelProvider, layout: str | list[list[str]] | None = None
) -> None:
    """Set the supported NeMoDiag V0 pipeline layout.

    An explicit layout always wins. The default test topology uses PP=2 and
    VP=4; other topologies leave automatic layer placement enabled.
    """
    if layout is not None:
        model_cfg.pipeline_model_parallel_layout = layout
        return

    pp_size = model_cfg.pipeline_model_parallel_size or 1
    vp_size = model_cfg.virtual_pipeline_model_parallel_size or 1
    model_cfg.pipeline_model_parallel_layout = NEMODIAG_V0_PIPELINE_LAYOUT if (pp_size, vp_size) == (2, 4) else None


def nemodiag_v0_pretrain_config() -> ConfigContainer:
    """Build the standalone synthetic model used by NeMoDiag V0 tests."""
    cfg = _pretrain_common()

    cfg.model = MLAModelProvider(
        # Synthetic test architecture.
        num_layers=31,
        hidden_size=7168,
        ffn_hidden_size=18432,
        num_attention_heads=128,
        num_query_groups=128,
        vocab_size=129280,
        layernorm_epsilon=1e-6,
        init_method_std=0.006,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation_func=F.silu,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        share_embeddings_and_output_weights=False,
        qk_layernorm=True,
        multi_latent_attention=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        position_embedding_type="rope",
        rotary_base=10000.0,
        rotary_scaling_factor=40,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
        # Synthetic MoE shape and routing.
        num_moe_experts=144,
        moe_ffn_hidden_size=2048,
        moe_shared_expert_intermediate_size=2048,
        moe_layer_freq=[0, 0, 0] + [1] * 28,
        moe_router_topk=8,
        moe_router_num_groups=8,
        moe_router_group_topk=4,
        moe_router_topk_scaling_factor=2.5,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0001,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_num_sms=16,
        moe_shared_expert_overlap=False,
        # Model implementation and fusions.
        transformer_layer_spec=partial(get_gpt_decoder_block_spec, use_transformer_engine=True),
        transformer_impl="transformer_engine",
        apply_rope_fusion=False,
        gradient_accumulation_fusion=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        cross_entropy_fusion_impl="te",
        cross_entropy_loss_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=True,
        attention_softmax_in_fp32=False,
        make_vocab_size_divisible_by=1280,
        # NeMoDiag V0 topology.
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=4,
        context_parallel_size=1,
        expert_model_parallel_size=36,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        seq_length=4096,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        recompute_granularity=None,
        recompute_modules=None,
        recompute_method=None,
        recompute_num_layers=None,
        fine_grained_activation_offloading=False,
        offload_modules=[],
        cuda_graph_impl="none",
        cuda_graph_scope=[],
        cuda_graph_warmup_steps=3,
        attention_backend=None,
        moe_router_padding_for_fp8=False,
    )
    set_nemodiag_v0_pipeline_model_parallel_layout(cfg.model)

    # Runtime identity is intentionally self-contained and does not download a tokenizer.
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = cfg.model.seq_length

    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 1152
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 2000
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.comm_overlap.overlap_grad_reduce = True

    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def _nemodiag_v0_common(cfg: ConfigContainer) -> None:
    """Apply precision-independent NeMoDiag benchmark settings."""
    cfg.dataset.seq_length = cfg.model.seq_length
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.model.moe_router_force_load_balancing = True
    cfg.dist.enable_megatron_core_experimental = True


def _enable_nemodiag_full_iteration_mxfp8(
    cfg: ConfigContainer,
    *,
    fp8_dot_product_attention: bool = False,
    fp8_output_proj: bool = False,
) -> None:
    """Apply full-iteration MXFP8 and HybridEP settings."""
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.fp8_output_proj = fp8_output_proj
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.mixed_precision.fp8_dot_product_attention = fp8_dot_product_attention
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True


__all__ = [
    "NEMODIAG_V0_PIPELINE_LAYOUT",
    "_benchmark_common",
    "_enable_nemodiag_full_iteration_mxfp8",
    "_enable_overlap_param_gather_with_optimizer_step",
    "_nemodiag_v0_common",
    "_perf_precision",
    "nemodiag_v0_pretrain_config",
    "set_nemodiag_v0_pipeline_model_parallel_layout",
]
