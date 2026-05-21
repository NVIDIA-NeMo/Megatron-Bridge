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

import torch
import torch.nn.functional as F

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


DEEPSEEK_V4_FLASH_HF_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _deepseek_v4_provider(hf_path: str):
    return AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)


def _set_deepseek_v4_smoke_geometry(
    cfg: ConfigContainer,
    *,
    num_layers: int,
    csa_compress_ratios: list[int],
    mtp_num_layers: int | None,
) -> None:
    """Apply small DSv4 geometry for architecture smoke tests."""
    cfg.model.experimental_attention_variant = "dsv4_hybrid"
    cfg.model.multi_latent_attention = True
    cfg.model.qk_layernorm = True
    cfg.model.normalization = "RMSNorm"
    cfg.model.add_bias_linear = False
    cfg.model.gated_linear_unit = True
    cfg.model.activation_func = F.silu
    cfg.model.use_te_activation_func = False
    cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.num_layers = num_layers
    cfg.model.hidden_size = 512
    cfg.model.ffn_hidden_size = 2048
    cfg.model.num_attention_heads = 8
    cfg.model.num_query_groups = 1
    cfg.model.q_lora_rank = 192
    cfg.model.qk_pos_emb_head_dim = 8
    cfg.model.v_head_dim = 16
    cfg.model.o_groups = 8
    cfg.model.o_lora_rank = 192
    cfg.model.seq_length = 1024
    cfg.model.max_position_embeddings = 1024
    cfg.model.position_embedding_type = "rope"
    cfg.model.rotary_base = 10000.0
    cfg.model.rotary_scaling_factor = 1
    cfg.model.original_max_position_embeddings = 1024
    cfg.model.csa_compress_rotary_base = 40000.0
    cfg.model.csa_window_size = 128
    cfg.model.csa_compress_ratios = csa_compress_ratios
    cfg.model.csa_dense_mode = False
    cfg.model.dsa_indexer_n_heads = 64
    cfg.model.dsa_indexer_head_dim = 128
    cfg.model.dsa_indexer_topk = 512
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.mtp_num_layers = mtp_num_layers
    cfg.model.mtp_loss_scaling_factor = 0.1


def _set_deepseek_v4_training_defaults(cfg: ConfigContainer) -> None:
    """Apply conservative defaults shared by DSv4 smoke recipes."""
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8
    cfg.dataset.seq_length = cfg.model.seq_length

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "unfused"
    cfg.model.apply_rope_fusion = False
    cfg.model.gradient_accumulation_fusion = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.masked_softmax_fusion = True
    cfg.model.attention_softmax_in_fp32 = True
    cfg.model.use_fused_mhc = False

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.validation.eval_iters = 10
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"


def _set_deepseek_v4_proxy_mhc(cfg: ConfigContainer, *, include_mtp: bool) -> None:
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = [
        ["embedding", "decoder", "decoder"],
        ["decoder", "decoder"] + (["mtp"] if include_mtp else []) + ["loss"],
    ]
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.enable_hyper_connections = True
    cfg.model.num_residual_streams = 4
    cfg.model.mhc_sinkhorn_iterations = 20
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mhc"]


def _set_deepseek_v4_hash_moe_proxy(cfg: ConfigContainer) -> None:
    cfg.model.num_moe_experts = 8
    cfg.model.moe_ffn_hidden_size = 512
    cfg.model.moe_shared_expert_intermediate_size = 512
    cfg.model.moe_layer_freq = [1] * 4
    cfg.model.moe_router_topk = 1
    cfg.model.moe_router_score_function = "sqrtsoftplus"
    cfg.model.moe_router_enable_expert_bias = True
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_router_pre_softmax = False
    cfg.model.moe_router_load_balancing_type = "noaux_tc"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_n_hash_layers = 1
    cfg.model.actual_vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
    cfg.model.activation_func_clamp_value = 10.0


def _disable_deepseek_v4_moe(cfg: ConfigContainer) -> None:
    cfg.model.num_moe_experts = None
    cfg.model.moe_ffn_hidden_size = None
    cfg.model.moe_shared_expert_intermediate_size = None
    cfg.model.moe_layer_freq = [0] * cfg.model.num_layers
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_n_hash_layers = 0
    cfg.model.actual_vocab_size = None
    cfg.model.activation_func_clamp_value = None


def deepseek_v4_tiny_pretrain_config(hf_path: str = DEEPSEEK_V4_FLASH_HF_PATH) -> ConfigContainer:
    """Return a tiny DeepSeek-V4 pre-training config for DSv4 attention smoke tests.

    This recipe intentionally avoids mHC, MTP, MoE, THD, CP, Muon, and TP>1 so
    that failures isolate to the DSv4 hybrid attention wiring first.
    """
    cfg = _pretrain_common()
    cfg.model = _deepseek_v4_provider(hf_path)

    _set_deepseek_v4_smoke_geometry(
        cfg,
        num_layers=1,
        csa_compress_ratios=[0],
        mtp_num_layers=None,
    )
    _set_deepseek_v4_training_defaults(cfg)

    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.enable_hyper_connections = False
    cfg.model.num_residual_streams = 1
    cfg.model.mhc_sinkhorn_iterations = 0
    cfg.model.num_moe_experts = None
    cfg.model.moe_layer_freq = [0]
    cfg.model.moe_shared_expert_intermediate_size = None
    cfg.model.moe_n_hash_layers = 0
    cfg.model.actual_vocab_size = None
    cfg.model.activation_func_clamp_value = None
    cfg.model.moe_shared_expert_overlap = False

    return cfg


def deepseek_v4_flash_proxy_pretrain_config(hf_path: str = DEEPSEEK_V4_FLASH_HF_PATH) -> ConfigContainer:
    """Return a small DeepSeek-V4 Flash-style proxy pre-training config.

    The proxy keeps TP=1 and CP=1, but exercises the DSv4 mixed CSA/HCA
    schedule, mHC, a small MoE, hash routing, and short-run training.
    MTP is intentionally disabled here to keep the mHC/hash-MoE path isolated;
    use ``deepseek_v4_flash_mtp_proxy_pretrain_config`` for mHC+MTP coverage.
    """
    cfg = _pretrain_common()
    cfg.model = _deepseek_v4_provider(hf_path)

    _set_deepseek_v4_smoke_geometry(
        cfg,
        num_layers=4,
        csa_compress_ratios=[0, 4, 128, 4],
        mtp_num_layers=None,
    )
    _set_deepseek_v4_training_defaults(cfg)

    _set_deepseek_v4_proxy_mhc(cfg, include_mtp=False)
    _set_deepseek_v4_hash_moe_proxy(cfg)

    return cfg


def deepseek_v4_flash_mtp_proxy_pretrain_config(hf_path: str = DEEPSEEK_V4_FLASH_HF_PATH) -> ConfigContainer:
    """Return a small DeepSeek-V4 Flash-style mHC+MTP proxy config.

    This isolates the mHC+MTP contract on the latest MCore dev pin while keeping
    TP=1, CP=1, and MoE disabled. The CSA/HCA schedule includes one extra entry
    for the MTP layer.
    """
    cfg = _pretrain_common()
    cfg.model = _deepseek_v4_provider(hf_path)

    _set_deepseek_v4_smoke_geometry(
        cfg,
        num_layers=4,
        csa_compress_ratios=[0, 4, 128, 4, 0],
        mtp_num_layers=1,
    )
    _set_deepseek_v4_training_defaults(cfg)

    _set_deepseek_v4_proxy_mhc(cfg, include_mtp=True)
    _disable_deepseek_v4_moe(cfg)

    return cfg
