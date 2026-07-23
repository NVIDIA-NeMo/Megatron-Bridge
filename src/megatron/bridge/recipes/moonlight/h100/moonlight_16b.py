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

from megatron.bridge import AutoBridge
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


_MOONLIGHT_16B_MODEL_ID = "moonshotai/Moonlight-16B-A3B"
_MOONLIGHT_16B_MODEL_REVISION = "476b36a473d4467f94469414bef6cee75c9c8172"  # pragma: allowlist secret
_MOONLIGHT_16B_FINETUNING_UNPADDED_VOCAB_SIZE = 163842


def _moonlight_16b_model_provider() -> MLAModelProvider:
    """Build the Moonlight architecture from its Hugging Face configuration."""
    return AutoBridge.from_hf_pretrained(
        _MOONLIGHT_16B_MODEL_ID, revision=_MOONLIGHT_16B_MODEL_REVISION
    ).to_megatron_provider(load_weights=False)


def _moonlight_16b_finetuning_model_provider(
    *,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    expert_parallel_size: int,
    sequence_parallel: bool,
) -> MLAModelProvider:
    """Build a checkpoint-compatible Moonlight provider for SFT or PEFT."""
    model = _moonlight_16b_model_provider()

    # Preserve all 163842 finetuning-token rows, then explicitly pad to the TP
    # multiple because setup treats a preset model vocab as already padded.
    model.vocab_size = (
        (_MOONLIGHT_16B_FINETUNING_UNPADDED_VOCAB_SIZE + tensor_parallel_size - 1) // tensor_parallel_size
    ) * tensor_parallel_size

    model.tensor_model_parallel_size = tensor_parallel_size
    model.pipeline_model_parallel_size = pipeline_parallel_size
    model.pipeline_dtype = torch.bfloat16
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = context_parallel_size
    model.expert_model_parallel_size = expert_parallel_size
    model.expert_tensor_parallel_size = 1
    model.sequence_parallel = sequence_parallel

    model.recompute_granularity = "selective"
    model.recompute_modules = None
    model.recompute_method = None
    model.recompute_num_layers = None

    # Preserve the finetuning recipe's source-model auxiliary-loss value.
    model.moe_aux_loss_coeff = 0.001
    return model


def _apply_moonlight_16b_finetuning_convergence_contract(cfg: ConfigContainer) -> None:
    """Apply the shared bounded SFT/PEFT convergence contract."""
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.dataset.seed = 1234
    cfg.rng.seed = 5678
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 10

    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.start_weight_decay = 0.033
    cfg.scheduler.end_weight_decay = 0.033
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.scheduler.lr_decay_style = "cosine"

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )

    cfg.checkpoint.save_interval = cfg.train.train_iters
    cfg.checkpoint.load = None
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.async_save = False

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.use_distributed_optimizer = True


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


def moonlight_16b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for Moonlight-16B.

    Recommended parallelism: TP=2, PP=1, EP=8
    Uses precision-aware optimizer with bf16 gradients/moments.
    """
    cfg = _pretrain_common()

    # Model config via AutoBridge (dispatches to DeepSeekV3Bridge)
    cfg.model = _moonlight_16b_model_provider()

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

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def moonlight_16b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the bounded-convergence pre-training config for Moonlight-16B.

    Recommended parallelism: TP=1, PP=1, CP=1, EP=8 on 16 H100 GPUs.
    The 100-step schedule uses GBS/MBS 1024/2 with 32-way gradient
    accumulation, two expert-data replicas, precision-aware bf16 optimizer
    state, and Moonlight's model-native routing configuration.

    Returns:
        ConfigContainer with the Moonlight-16B pre-training contract.
    """
    cfg = moonlight_16b_pretrain_8gpu_h100_bf16_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(1, 1)
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    # Keep each expert-parallel group inside one 8-GPU NVLink domain. The
    # resulting two expert-data replicas avoid cross-node token dispatch.
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    # The 16-GPU topology fits without activation recompute at MBS2. Matching
    # the DeepSeek V3 precision-aware optimizer state reduces accumulation and
    # optimizer overhead while keeping fp32 main parameters.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_deepep_num_sms = None
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.high_priority_a2a_comm_stream = True

    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = True

    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 2
    cfg.dataset.random_seed = 1234
    cfg.rng.seed = 1234
    cfg.scheduler.lr_warmup_iters = 40
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None

    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.lr = 3e-4
    cfg.optimizer.min_lr = 3e-5
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.scheduler.start_weight_decay = 0.033
    cfg.scheduler.end_weight_decay = 0.033
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.scheduler.lr_decay_style = "cosine"

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    cfg.ddp.grad_reduce_in_fp32 = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "USE_MNNVL": 0,
    }
    return cfg


# =============================================================================
# SFT Config
# =============================================================================


def moonlight_16b_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the legacy TP4/PP2 full SFT config for Moonlight-16B.

    Recommended parallelism: TP=4, PP=2, CP=1, EP=4 on 8 H100 GPUs.
    Sequence parallelism resolves offline packing to ``pad_seq_to_mult=4``, so
    this remains a support recipe rather than the shared pad-1 convergence
    cohort.

    Returns:
        ConfigContainer with all settings pre-configured for Moonlight-16B SFT.
    """
    cfg = _sft_common()

    cfg.model = _moonlight_16b_finetuning_model_provider(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        context_parallel_size=1,
        expert_parallel_size=4,
        sequence_parallel=True,
    )

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Set pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(2, 1)

    # Sequence length
    seq_length = 2048
    cfg.model.seq_length = seq_length
    # Dataset config - enable_offline_packing=True by default
    cfg.dataset.seq_length = seq_length
    cfg.dataset.offline_packing_specs.packed_sequence_size = seq_length
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 4

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
    # recompute_granularity already set in model provider
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

    _apply_moonlight_16b_finetuning_convergence_contract(cfg)

    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.offline_packing_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Tokenizer - HuggingFace tokenizer with trust_remote_code
    cfg.tokenizer.tokenizer_model = _MOONLIGHT_16B_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {
        "revision": _MOONLIGHT_16B_MODEL_REVISION,
        "trust_remote_code": True,
    }
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # Communication overlap settings
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def moonlight_16b_sft_8gpu_h100_bf16_tp1_config() -> ConfigContainer:
    """Return the bounded-convergence full SFT config for Moonlight-16B.

    Recommended parallelism is TP=1, PP=1, CP=1, EP=8 on 8 H100 GPUs. This
    topology keeps sequence parallelism disabled so the shared SFT packing
    contract remains ``pad_seq_to_mult=1``. With GBS=32 and MBS=1 it uses
    dense DP=8 and four gradient-accumulation steps.

    Returns:
        ConfigContainer with the Moonlight-16B full-SFT convergence contract.
    """
    cfg = moonlight_16b_sft_8gpu_h100_bf16_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(1, 1)
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.vocab_size = _MOONLIGHT_16B_FINETUNING_UNPADDED_VOCAB_SIZE

    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 1

    # The model fits without activation recompute at this topology. Router
    # fusion and bf16 optimizer state reduce otherwise exposed execution and
    # communication overhead while retaining fp32 main parameters.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # Reuse the tuned single-node HybridEP transport and expert-communication
    # overlap from the bounded pretraining topology.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_deepep_num_sms = None
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_a2a_overlap = False
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "USE_MNNVL": 0,
    }
    return cfg


def moonlight_16b_sft_8gpu_h100_bf16_8k_config() -> ConfigContainer:
    """Return the separate 8K-context SFT config for Moonlight-16B.

    This recipe preserves the previously verified long-context execution and
    precision cohort instead of inheriting the 2K bounded-convergence SFT
    contract. Recommended parallelism is TP=2, PP=1, CP=2, EP=8 on 8 H100
    GPUs.

    Returns:
        ConfigContainer with the Moonlight-16B 8K-context SFT contract.
    """
    cfg = moonlight_16b_sft_8gpu_h100_bf16_config()

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(1, 1)
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.vocab_size = _MOONLIGHT_16B_FINETUNING_UNPADDED_VOCAB_SIZE

    seq_length = 8192
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.offline_packing_specs.packed_sequence_size = seq_length
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    cfg.train.train_iters = 20
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    cfg.optimizer.lr = 1e-6
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.adam_eps = 1e-5
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.scheduler.lr_warmup_iters = 2
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None

    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# =============================================================================
# PEFT Config
# =============================================================================


def moonlight_16b_peft_2gpu_h100_bf16_config(
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

    cfg.model = _moonlight_16b_finetuning_model_provider(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=2,
        sequence_parallel=False,
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
    # Dataset config - enable_offline_packing=True by default
    cfg.dataset.seq_length = seq_length
    cfg.dataset.offline_packing_specs.packed_sequence_size = seq_length

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
        cfg.dataset.offline_packing_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

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
    cfg.tokenizer.tokenizer_model = _MOONLIGHT_16B_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {
        "revision": _MOONLIGHT_16B_MODEL_REVISION,
        "trust_remote_code": True,
    }

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

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def moonlight_16b_peft_4gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return the bounded-convergence PEFT config for Moonlight-16B.

    Recommended parallelism is TP=1, PP=1, CP=1, EP=4 on 4 H100 GPUs. This
    keeps sequence parallelism disabled and uses dense DP=4 with eight gradient
    accumulation steps. The default LoRA keeps the base model frozen and
    targets only the attention QKV and output projections with rank 8, alpha
    16, and zero dropout.

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with the Moonlight-16B PEFT convergence contract.
    """
    cfg = moonlight_16b_peft_2gpu_h100_bf16_config(peft_scheme=peft_scheme)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = _get_moonlight_pipeline_layout(1, 1)
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.vocab_size = _MOONLIGHT_16B_FINETUNING_UNPADDED_VOCAB_SIZE

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True

    seq_length = 2048
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.offline_packing_specs.packed_sequence_size = seq_length
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 4

    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(
            peft_scheme,
            dim=8,
            alpha=16,
            dropout=0.0,
            target_modules=["linear_q_proj", "linear_kv_down_proj", "linear_kv_up_proj", "linear_proj"],
        )

    _apply_moonlight_16b_finetuning_convergence_contract(cfg)

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "moonlight_16b_peft_2gpu_h100_bf16_config",
    "moonlight_16b_peft_4gpu_h100_bf16_config",
    "moonlight_16b_pretrain_16gpu_h100_bf16_config",
    "moonlight_16b_pretrain_8gpu_h100_bf16_config",
    "moonlight_16b_sft_8gpu_h100_bf16_8k_config",
    "moonlight_16b_sft_8gpu_h100_bf16_config",
    "moonlight_16b_sft_8gpu_h100_bf16_tp1_config",
]
