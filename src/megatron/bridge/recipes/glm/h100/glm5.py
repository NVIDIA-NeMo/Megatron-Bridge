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
"""Functional H100 recipes for GLM-5.2 verification."""

from __future__ import annotations

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config, default_tulu3_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_GLM52_MODEL_ID = "zai-org/GLM-5.2"
_GLM52_MODEL_REVISION = "4d67f66cc64d3219133b767c253b2ad1425c6c88"  # pragma: allowlist secret
_TULU3_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"  # pragma: allowlist secret

_GLM52_VPP2_LAYOUT = (
    "Et|t||tttt|tttt|tttt|tttt||tttt|tttt|tttt|tttt||tttt|tttt|tttt|tttt||tttt|tttt|tttt|tttt||tttt|tttt|ttttmL"
)
_GLM52_PP13_LAYOUT = "Etttttt|tttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttttttt|tttt|tttt|tttt|tttt|ttttmL"
_GLM52_PP19_LONG_CONTEXT_LAYOUT = (
    "Etttttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|tttt|ttttmL"
)


def _configure_model(
    cfg: ConfigContainer,
    *,
    seq_length: int,
    context_parallel_size: int,
    expert_parallel_size: int,
    virtual_pipeline_size: int | None,
    microbatch_group_size: int | None,
) -> None:
    cfg.model = AutoBridge.from_hf_pretrained(_GLM52_MODEL_ID, revision=_GLM52_MODEL_REVISION).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_model = _GLM52_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _GLM52_MODEL_REVISION}

    cfg.model.seq_length = seq_length
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 13
    cfg.model.virtual_pipeline_model_parallel_size = virtual_pipeline_size
    cfg.model.pipeline_model_parallel_layout = _GLM52_VPP2_LAYOUT if virtual_pipeline_size == 2 else _GLM52_PP13_LAYOUT
    cfg.model.context_parallel_size = context_parallel_size
    cfg.model.expert_model_parallel_size = expert_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.microbatch_group_size_per_vp_stage = microbatch_group_size

    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.mtp_num_layers = 1
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1
    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}


def _configure_optimizer(
    cfg: ConfigContainer,
    *,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    train_iters: int,
) -> None:
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=min_lr,
    )
    cfg.scheduler.lr_decay_iters = train_iters


def _configure_tulu3_dataset(cfg: ConfigContainer, *, pad_seq_to_mult: int, output_name: str) -> None:
    cfg.dataset = default_tulu3_config(
        seq_length=2048,
        enable_offline_packing=True,
        pad_seq_to_mult=pad_seq_to_mult,
    )
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = f"work/data/glm5-2/{output_name}"
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False


def glm52_pretrain_416gpu_h100_bf16_config() -> ConfigContainer:
    """GLM-5.2 bounded pretraining on 416 H100 GPUs."""
    cfg = _pretrain_common()
    _configure_model(
        cfg,
        seq_length=4096,
        context_parallel_size=1,
        expert_parallel_size=32,
        virtual_pipeline_size=2,
        microbatch_group_size=16,
    )
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.random_seed = 1234
    cfg.rng.seed = 1234
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    _configure_optimizer(cfg, max_lr=3e-4, min_lr=3e-5, warmup_iters=40, train_iters=100)
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    return cfg


def glm52_sft_functional_416gpu_h100_bf16_config() -> ConfigContainer:
    """GLM-5.2 bounded full SFT on 416 H100 GPUs."""
    cfg = _sft_common()
    _configure_model(
        cfg,
        seq_length=2048,
        context_parallel_size=16,
        expert_parallel_size=32,
        virtual_pipeline_size=2,
        microbatch_group_size=16,
    )
    _configure_tulu3_dataset(cfg, pad_seq_to_mult=32, output_name="tulu3-full-sft-pad32")
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    _configure_optimizer(cfg, max_lr=5e-6, min_lr=0.0, warmup_iters=10, train_iters=100)
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    return cfg


def glm52_sft_long_context_608gpu_h100_bf16_config() -> ConfigContainer:
    """GLM-5.2 200K packed SFT with context parallelism on 608 H100 GPUs."""
    cfg = _sft_common()
    _configure_model(
        cfg,
        seq_length=200000,
        context_parallel_size=32,
        expert_parallel_size=32,
        virtual_pipeline_size=None,
        microbatch_group_size=None,
    )
    cfg.model.pipeline_model_parallel_size = 19
    cfg.model.pipeline_model_parallel_layout = _GLM52_PP19_LONG_CONTEXT_LAYOUT
    cfg.dataset.seq_length = 200000
    cfg.dataset.hf_dataset = None
    cfg.dataset.dataset_root = "work/data/glm5-2/synthetic-200k"
    cfg.dataset.hf_output_root = None
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.seed = 1234
    cfg.dataset.preprocessing = ChatSFTPreprocessingConfig()
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.packed_sequence_size = 200000
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 64
    cfg.rng.seed = 5678
    cfg.train.train_iters = 20
    cfg.train.global_batch_size = 13
    _configure_optimizer(cfg, max_lr=1e-6, min_lr=0.0, warmup_iters=2, train_iters=20)
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    return cfg


def glm52_peft_208gpu_h100_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """GLM-5.2 bounded LoRA SFT on 208 H100 GPUs."""
    cfg = _peft_common()
    _configure_model(
        cfg,
        seq_length=2048,
        context_parallel_size=1,
        expert_parallel_size=16,
        virtual_pipeline_size=None,
        microbatch_group_size=None,
    )
    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in {"lora", "dora"}:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
        peft_cfg.target_modules = [
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
            "linear_proj",
        ]
    cfg.peft = peft_cfg
    _configure_tulu3_dataset(cfg, pad_seq_to_mult=4, output_name="tulu3-peft-pad4")
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    _configure_optimizer(cfg, max_lr=1e-4, min_lr=0.0, warmup_iters=10, train_iters=100)
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    return cfg


__all__ = [
    "glm52_peft_208gpu_h100_bf16_config",
    "glm52_pretrain_416gpu_h100_bf16_config",
    "glm52_sft_functional_416gpu_h100_bf16_config",
    "glm52_sft_long_context_608gpu_h100_bf16_config",
]
