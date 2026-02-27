# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from megatron.bridge import AutoBridge
from megatron.bridge.models.deepseek import MoonlightModelProvider16B
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def _get_moonlight_pipeline_layout(pp_size: int, vp_size: int):
    """Get pipeline layout for Moonlight-16B based on PP and VP size."""
    map_pp_vp_to_layout = {
        (1, 1): None,
        (2, 1): [["embedding"] + ["decoder"] * 14, ["decoder"] * 13 + ["loss"]],
        (4, 1): [["embedding"] + ["decoder"] * 7] + [["decoder"] * 7] * 2 + [["decoder"] * 6 + ["loss"]],
        (8, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 6 + [["decoder"] * 3 + ["loss"]],
        (2, 2): [["embedding"] + ["decoder"] * 7] + [["decoder"] * 7] * 2 + [["decoder"] * 6 + ["loss"]],
        (4, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 6 + [["decoder"] * 3 + ["loss"]],
    }
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for Moonlight-16B. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )
    layout = map_pp_vp_to_layout[(pp_size, vp_size)]
    if layout is not None:
        layout = list([list(x) for x in layout])
    return layout


def moonlight_16b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Moonlight-16B.

    Recommended parallelism: TP=2, PP=1, EP=8
    Uses precision-aware optimizer with bf16 gradients/moments.
    """
    cfg = _pretrain_common()

    # Model config via AutoBridge (dispatches to DeepSeekV3Bridge)
    cfg.model = AutoBridge.from_hf_pretrained("moonshotai/Moonlight-16B-A3B").to_megatron_provider(load_weights=False)
    # TEMPFIX(yuya): Moonlight has no Q LoRA compression (HF q_lora_rank=null),
    # but CONFIG_MAPPING skips None so MLATransformerConfig default (512) would be used
    cfg.model.q_lora_rank = None

    # Tokenizer - uses NullTokenizer with model vocab_size
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.split = "99990,8,2"

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # Set pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(1, 1)

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.train_iters = 500_000
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 2000
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters

    # Optimizer settings - precision-aware optimizer with bf16 moments
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - custom MixedPrecisionConfig (NOT "bf16_mixed" string)
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = True

    # Checkpoint config
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (DIFFERENT: grad_reduce_in_fp32=False)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # for mla rope fusion

    return cfg


# =============================================================================
# SFT Config
# =============================================================================


def moonlight_16b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Moonlight-16B.

    Default parallelism: TP=2, PP=1, EP=8, SP=True

    Returns:
        ConfigContainer with all settings pre-configured for Moonlight-16B SFT.
    """
    cfg = _sft_common()

    # Model config - uses MoonlightModelProvider16B
    cfg.model = MoonlightModelProvider16B(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        sequence_parallel=True,
        expert_tensor_parallel_size=1,
        recompute_granularity="selective",
        recompute_modules=None,
        recompute_method=None,
        recompute_num_layers=None,
    )

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Set pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(1, 1)

    # Sequence length
    seq_length = 4096
    cfg.model.seq_length = seq_length
    # Dataset config - packed_sequence=True by default
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length

    # MoE performance optimizations
    cfg.model.moe_permute_fusion = True

    # DeePEP settings - set to True to enable DeePEP
    enable_deepep = False
    if enable_deepep:
        cfg.model.moe_token_dispatcher_type = "flex"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = False
    else:
        # Default MoE Token Dispatcher settings (without DeePEP)
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = True

    cfg.model.moe_hybridep_num_sms = 16

    # RoPE fusion - set to True to enable RoPE fusion
    apply_rope_fusion = False
    if apply_rope_fusion:
        cfg.model.apply_rope_fusion = True
        cfg.dist.enable_megatron_core_experimental = True

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    # recompute_granularity already set in MoonlightModelProvider16B
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: Moonlight uses MixedPrecisionConfig with bf16=True
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # Training config overrides
    cfg.validation.eval_interval = 50
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides - Moonlight uses specific optimizer settings
    cfg.optimizer.adam_eps = 1e-5
    cfg.optimizer.weight_decay = 0.1

    # Precision-aware optimizer settings
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Scheduler overrides
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.lr_decay_iters = 1000
    cfg.scheduler.min_lr = 0.0

    # Mixed precision config - Moonlight uses MixedPrecisionConfig (not "bf16_mixed" string)
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )

    # Tokenizer - HuggingFace tokenizer with trust_remote_code
    cfg.tokenizer.tokenizer_model = "moonshotai/Moonlight-16B-A3B"
    cfg.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.async_save = False
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # DDP config - Moonlight uses grad_reduce_in_fp32=False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.use_distributed_optimizer = True

    # Communication overlap settings
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    return cfg


# =============================================================================
# PEFT Config
# =============================================================================


def moonlight_16b_peft_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Moonlight-16B.

    Default parallelism: TP=1, PP=1, EP=2, SP=False

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Moonlight-16B PEFT.
    """
    cfg = _peft_common()

    # Model config - PEFT uses TP=1, EP=2
    cfg.model = MoonlightModelProvider16B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=2,
        sequence_parallel=False,
        expert_tensor_parallel_size=1,
        recompute_granularity="selective",
        recompute_modules=None,
        recompute_method=None,
        recompute_num_layers=None,
    )

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Set pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(1, 1)

    # Sequence length
    seq_length = 4096
    cfg.model.seq_length = seq_length
    # Dataset config - packed_sequence=True by default
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length

    # MoE performance optimizations
    cfg.model.moe_permute_fusion = True

    # DeePEP settings - set to True to enable DeePEP
    enable_deepep = False
    if enable_deepep:
        cfg.model.moe_token_dispatcher_type = "flex"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = False
    else:
        # Default MoE Token Dispatcher settings (without DeePEP)
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = True

    cfg.model.moe_hybridep_num_sms = 16

    # RoPE fusion - set to True to enable RoPE fusion
    apply_rope_fusion = False
    if apply_rope_fusion:
        cfg.model.apply_rope_fusion = True
        cfg.dist.enable_megatron_core_experimental = True

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 settings
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # PEFT config - override default LoRA with user-specified scheme
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme

    # Training config overrides
    cfg.validation.eval_interval = 50
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_eps = 1e-5
    cfg.optimizer.weight_decay = 0.1

    # Precision-aware optimizer settings
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Scheduler overrides
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.lr_decay_iters = 1000
    cfg.scheduler.min_lr = 0.0

    # Mixed precision config
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )

    # Tokenizer - HuggingFace tokenizer with trust_remote_code
    cfg.tokenizer.tokenizer_model = "moonshotai/Moonlight-16B-A3B"
    cfg.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.async_save = False
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # DDP config - Moonlight uses grad_reduce_in_fp32=False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.use_distributed_optimizer = True

    # Communication overlap settings
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    return cfg


__all__ = [
    # Pretrain config
    "moonlight_16b_pretrain_config",
    # SFT config
    "moonlight_16b_sft_config",
    # PEFT config
    "moonlight_16b_peft_config",
]
