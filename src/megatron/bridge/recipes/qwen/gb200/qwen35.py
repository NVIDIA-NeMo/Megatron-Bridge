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

"""GB200 text-only pretraining and SFT recipes for Qwen3.5 models."""

from __future__ import annotations

from copy import deepcopy

import torch
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_coderforge_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed


_QWEN35_9B_BASE = "Qwen/Qwen3.5-9B-Base"
_QWEN35_35B_A3B_BASE = "Qwen/Qwen3.5-35B-A3B-Base"
_QWEN35_35B_A3B_INSTRUCT_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"  # pragma: allowlist secret
_CODERFORGE_REVISION = "060fca96cf723b2ebab3181e9e59fafd273df3cb"  # pragma: allowlist secret


def qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config(
    seq_length: int = 131072,
    *,
    hf_path: str = "Qwen/Qwen3.5-35B-A3B",
    hf_revision: str | None = _QWEN35_35B_A3B_INSTRUCT_REVISION,
) -> ConfigContainer:
    """MXFP8 text SFT with packed chat data, TP1/CP8/EP16 and one MTP layer.

    Set a pretrained text checkpoint before training. The default CoderForge
    source is normalized and packed by the dataset builder. Model config and
    tokenizer share a pinned revision; use ``hf_revision=None`` for an extracted
    local causal-LM checkpoint. No vision model is constructed.

    Args:
        seq_length: Packed token budget, a positive multiple of 16 for CP8.
        hf_path: HF model ID or extracted local text checkpoint.
        hf_revision: Matching model/tokenizer revision, or None for local inputs.

    Returns:
        Training configuration with MXFP8 parameter gather and CuTeDSL grouped MLP.
    """
    if seq_length <= 0 or seq_length % 16:
        raise ValueError("seq_length must be positive and divisible by 16 for CP8.")
    cfg = _sft_common()
    hf_config = AutoConfig.from_pretrained(hf_path, revision=hf_revision)
    text_config = deepcopy(getattr(hf_config, "text_config", hf_config))
    if text_config.model_type != "qwen3_5_moe_text":
        raise ValueError("Expected a Qwen3.5 MoE text configuration.")
    if seq_length > text_config.max_position_embeddings:
        raise ValueError("seq_length exceeds the text model's max_position_embeddings.")
    if hasattr(hf_config, "text_config"):
        text_config.tie_word_embeddings = hf_config.tie_word_embeddings
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": hf_revision}

    cfg.model.seq_length = seq_length
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 8
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.mtp_num_layers = 1
    cfg.model.bias_activation_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.calculate_per_token_loss = True
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["gdn_norm_out", "moe"]
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_use_grouped_tensor = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_single_grouped_weight = True

    # Pad each runtime segment to 2*CP, not each conversation to seq_length.
    # Physical and logical cu_seqlens remain distinct for attention and MTP.
    cfg.dataset = default_coderforge_config(
        seq_length=seq_length,
        enable_offline_packing=True,
        pad_seq_to_mult=16,
    )
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _CODERFORGE_REVISION}
    cfg.dataset.do_validation = True
    cfg.dataset.hf_validation_proportion = 0.05
    # Preserve the original long-context recipe's sampling and worker lifecycle.
    cfg.dataset.seed = 1234
    cfg.dataset.persistent_workers = True
    cfg.train.train_iters = 500
    cfg.train.global_batch_size = 32
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.rng.seed = 1234
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 8
    # Preserve the VL baseline's LR schedule rather than silently retuning SFT.
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200, lr_decay_iters=300000, max_lr=2e-5, min_lr=2e-6
    )
    cfg.optimizer.overlap_param_gather = True
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.checkpoint.load_main_params_from_ckpt = True
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL": 1,
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM": 1,
    }
    cfg.checkpoint.load = None
    return cfg


def qwen35_text_9b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a text-only Qwen3.5-9B pretraining config for eight GB200 GPUs."""
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_9B_BASE).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_9B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Follow the Llama 3 8B GB200 topology: keep model parallelism at one and
    # use all eight GPUs for data parallelism.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Capture the dense attention and MLP modules. Keep cross entropy on the
    # native fused path validated by the 64-GPU GB200 performance run.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn", "mlp"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.rerun_state_machine.check_for_nan_in_loss = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    return cfg


def qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a text-only Qwen3.5-35B-A3B pretraining config for eight GB200 GPUs."""
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_35B_A3B_BASE).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_35B_A3B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Match the Qwen3.5-VL GB200 topology while training only the text model.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 512
    # MBS4 is suitable for force-balanced throughput benchmarking, but OOMs
    # with learned routing. MBS1 was validated with real RP2 data on GB200.
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    # Keep the library-safe native implementation instead of the performance
    # harness's TE cross-entropy path, which currently warns about stability.
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Fixed-length text batches can use the scopes that the VLM recipe must
    # disable for variable-length multimodal inputs.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    return cfg
