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
from megatron.core.activations import squared_relu

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
_HF_MODEL_REVISION = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"  # pragma: allowlist secret


def _model_config(*, seq_length: int, context_parallel_size: int = 1) -> HybridModelProvider:
    """Build the exact dense Nemotron 3 Nano 4B architecture."""
    return HybridModelProvider(
        hybrid_layer_pattern="M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
        num_layers=42,
        hidden_size=3136,
        ffn_hidden_size=12544,
        num_attention_heads=40,
        num_query_groups=8,
        kv_channels=128,
        mamba_num_heads=96,
        mamba_head_dim=80,
        mamba_state_dim=128,
        mamba_num_groups=8,
        mamba_chunk_size=256,
        seq_length=seq_length,
        vocab_size=131072,
        should_pad_vocab=False,
        make_vocab_size_divisible_by=128,
        share_embeddings_and_output_weights=False,
        position_embedding_type="none",
        activation_func=squared_relu,
        gated_linear_unit=False,
        add_bias_linear=False,
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        init_method_std=0.02,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        masked_softmax_fusion=True,
        apply_query_key_layer_scaling=False,
        persist_layer_norm=True,
        attention_softmax_in_fp32=False,
        first_last_layers_bf16=True,
        is_hybrid_model=True,
        use_mamba_mem_eff_path=True,
        transformer_impl="transformer_engine",
        attention_backend=None,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=context_parallel_size,
        sequence_parallel=False,
    )


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
    cfg.optimizer.use_precision_aware_optimizer = True
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
