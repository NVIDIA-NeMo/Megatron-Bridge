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

"""Bounded H100 training recipes for Muse Glimmer 30B."""

from __future__ import annotations

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_MODEL_ID = "meta-models/Muse-Glimmer-30B"
_MODEL_REVISION = "f84ecc3a0ea984a4c04542a84269e3d065350a6e"  # pragma: allowlist secret
_CORD_V2_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"  # pragma: allowlist secret


def muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config() -> ConfigContainer:
    """Return a 100-step random-init multimodal pretraining config on 32 H100 GPUs."""
    cfg = _sft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID, revision=_MODEL_REVISION).get_model_config()
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = f"{'*' * 20}|{'*' * 32}"
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "p2p"
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.recompute_vision_layers = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_kwargs = {
        "revision": _MODEL_REVISION,
        "max_image_tokens": 256,
    }
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.enable_in_batch_packing = False
    # Worker-local RNG and prefetch queues are not part of the checkpoint.
    # Main-process loading keeps the step-50 resume data stream reproducible.
    cfg.dataset.num_workers = 0
    cfg.dataset.dataloader_type = "cyclic"

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 1234
    cfg.train.train_iters = 100
    # TP8/PP2 leaves two data-parallel replicas on 32 GPUs. One sample per
    # replica keeps this bounded verification recipe to one microbatch per
    # optimizer step while exercising the complete multimodal model.
    cfg.train.global_batch_size = 2
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=40,
        max_lr=3e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_glimmer_30b_sft_32gpu_h100_bf16_config() -> ConfigContainer:
    """Return a 100-step full multimodal SFT config on 32 H100 GPUs."""
    cfg = _sft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID, revision=_MODEL_REVISION).get_model_config()
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = f"{'*' * 20}|{'*' * 32}"
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "p2p"
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.recompute_vision_layers = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_kwargs = {
        "revision": _MODEL_REVISION,
        "max_image_tokens": 256,
    }
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.num_workers = 4
    cfg.dataset.dataloader_type = "cyclic"
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.enable_in_batch_packing = False

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config() -> ConfigContainer:
    """Return a 100-step packed 8K full SFT config with CP2 on 32 H100 GPUs."""
    cfg = _sft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID, revision=_MODEL_REVISION).get_model_config()
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = "|".join("*" * layers for layers in (9, 15, 15, 13))
    cfg.model.context_parallel_size = 2
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.recompute_vision_layers = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]

    cfg.dataset.seq_length = 8192
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_kwargs = {
        "revision": _MODEL_REVISION,
        "max_image_tokens": 256,
    }
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.num_workers = 4
    cfg.dataset.dataloader_type = "cyclic"
    cfg.dataset.pad_to_max_length = False
    cfg.dataset.enable_in_batch_packing = True

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_glimmer_30b_peft_8gpu_h100_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a 100-step attention-projection LoRA config on 8 H100 GPUs."""
    cfg = _peft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID, revision=_MODEL_REVISION).get_model_config()
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = "*" * 52
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "p2p"
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.recompute_vision_layers = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]

    cfg.dataset.seq_length = 8192
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_kwargs = {
        "revision": _MODEL_REVISION,
        "max_image_tokens": 256,
    }
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.num_workers = 4
    cfg.dataset.dataloader_type = "cyclic"
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.enable_in_batch_packing = False

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in {"lora", "dora"}:
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
    cfg.peft = peft_cfg

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=1e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "muse_glimmer_30b_peft_8gpu_h100_bf16_config",
    "muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config",
    "muse_glimmer_30b_sft_32gpu_h100_bf16_config",
    "muse_glimmer_30b_sft_32gpu_h100_bf16_long_context_config",
]
