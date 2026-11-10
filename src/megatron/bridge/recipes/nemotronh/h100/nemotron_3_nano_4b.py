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

"""Recipes for the dense NVIDIA Nemotron 3 Nano 4B model."""

from __future__ import annotations

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
_HF_MODEL_REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"  # pragma: allowlist secret


def _model_config(*, seq_length: int, context_parallel_size: int = 1) -> BridgeHybridModelConfig:
    """Build the dense Nemotron 3 Nano 4B architecture from its pinned HF config."""
    model = AutoBridge.from_hf_pretrained(_HF_MODEL_ID, revision=_HF_MODEL_REVISION).get_model_config()
    # The model is dense, but its Transformers config declares default MoE values. Keep every
    # MoE field at its Megatron Core default.
    model.num_moe_experts = None
    model.moe_ffn_hidden_size = None
    model.moe_shared_expert_intermediate_size = None
    model.moe_router_topk_scaling_factor = None
    model.moe_router_num_groups = None
    model.moe_router_group_topk = None
    model.moe_router_score_function = "softmax"
    model.moe_router_enable_expert_bias = False
    model.moe_router_load_balancing_type = "aux_loss"
    model.moe_aux_loss_coeff = 0.0
    model.moe_router_dtype = None
    model.moe_grouped_gemm = False
    model.moe_permute_fusion = False
    model.moe_shared_expert_overlap = False
    model.moe_token_dispatcher_type = "allgather"
    model.seq_length = seq_length
    model.should_pad_vocab = False
    model.use_mamba_mem_eff_path = True
    model.transformer_impl = "transformer_engine"
    model.attention_backend = None
    model.cross_entropy_loss_fusion = True
    model.cross_entropy_fusion_impl = "native"
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.pipeline_dtype = torch.bfloat16
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = context_parallel_size
    model.sequence_parallel = False
    return model


def _configure_tokenizer(cfg: ConfigContainer) -> None:
    """Pin the model-native tokenizer revision."""
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = _HF_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _HF_MODEL_REVISION, "use_fast": True}


def _configure_kernels(cfg: ConfigContainer) -> None:
    """Set the common BF16 execution and memory policy."""
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True


def _configure_optimizer(cfg: ConfigContainer, *, lr: float, min_lr: float, grad_reduce_in_fp32: bool) -> None:
    """Apply the bounded model-card convergence cohort optimizer contract."""
    cfg.optimizer.lr = lr
    cfg.optimizer.min_lr = min_lr
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1.0e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.start_weight_decay = 0.033
    cfg.scheduler.end_weight_decay = 0.033
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.scheduler.lr_decay_style = "cosine"

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = grad_reduce_in_fp32
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = grad_reduce_in_fp32
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True


def nemotron_3_nano_4b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the bounded random-init pretraining config for eight H100 GPUs.

    Recommended parallelism is TP=1, PP=1, CP=1, DP=8. The recipe is random
    initialization by default; loading the released HF checkpoint is not part of
    this recipe's verification contract.
    """
    cfg = _pretrain_common()
    cfg.model = _model_config(seq_length=4096)
    _configure_tokenizer(cfg)
    _configure_kernels(cfg)

    cfg.dataset.seq_length = 4096
    cfg.dataset.random_seed = 1234
    cfg.dataset.num_dataset_builder_threads = 1
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.rng.seed = 1234
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    _configure_optimizer(cfg, lr=3.0e-4, min_lr=3.0e-5, grad_reduce_in_fp32=False)
    cfg.scheduler.lr_warmup_iters = 40
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_nano_4b_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the 2K packed full-SFT config for eight H100 GPUs.

    Recommended parallelism is TP=1, PP=1, CP=1, DP=8.
    """
    cfg = _sft_common()
    cfg.model = _model_config(seq_length=2048)
    _configure_tokenizer(cfg)
    _configure_kernels(cfg)

    cfg.dataset.seq_length = 2048
    cfg.dataset.seed = 1234
    cfg.dataset.offline_packing_specs.packed_sequence_size = 2048
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 1
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.rng.seed = 5678
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    _configure_optimizer(cfg, lr=5.0e-6, min_lr=0.0, grad_reduce_in_fp32=True)
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_nano_4b_sft_8gpu_h100_bf16_32k_config() -> ConfigContainer:
    """Return the packed 32K full-SFT config with CP=2 for eight H100 GPUs.

    Recommended parallelism is TP=1, PP=1, CP=2, DP=4.
    """
    cfg = nemotron_3_nano_4b_sft_8gpu_h100_bf16_config()
    cfg.model.context_parallel_size = 2
    cfg.model.cp_comm_type = "a2a"
    cfg.model.seq_length = 32768
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.calculate_per_token_loss = True
    cfg.dataset.seq_length = 32768
    cfg.dataset.offline_packing_specs.packed_sequence_size = 32768
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 4
    cfg.train.global_batch_size = 8
    cfg.ddp.average_in_collective = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_nano_4b_peft_8gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return the packed attention-LoRA config for eight H100 GPUs.

    Args:
        peft_scheme: PEFT scheme (``"lora"``, ``"dora"``), or a custom PEFT
            instance.

    Recommended parallelism is TP=1, PP=1, CP=1, DP=8.
    """
    cfg = _peft_common()
    cfg.model = _model_config(seq_length=2048)
    _configure_tokenizer(cfg)
    _configure_kernels(cfg)

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in {"lora", "dora"}:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
    cfg.peft = peft_cfg

    cfg.dataset.seq_length = 2048
    cfg.dataset.seed = 1234
    cfg.dataset.offline_packing_specs.packed_sequence_size = 2048
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 4
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.rng.seed = 5678
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    _configure_optimizer(cfg, lr=1.0e-4, min_lr=0.0, grad_reduce_in_fp32=True)
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "nemotron_3_nano_4b_peft_8gpu_h100_bf16_config",
    "nemotron_3_nano_4b_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_nano_4b_sft_8gpu_h100_bf16_32k_config",
    "nemotron_3_nano_4b_sft_8gpu_h100_bf16_config",
]
