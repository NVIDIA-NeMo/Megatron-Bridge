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

"""Beam recipes for H200 nodes (8 GPUs, 141 GB each).

Beam is a 501B-parameter MoE (52 layers, 256 experts, top-8) built on ``HybridModel``. About 98% of its
parameters are experts, so every recipe shards them with expert parallelism; ``cp_comm_type="a2a"`` is set
because Beam's sliding-window and NoPE layers need all-to-all context parallelism whenever CP > 1.

The SFT and PEFT recipes own the ``qwen3_30b_a3b_convergence_v2`` contract (8K packed Tulu 3, GBS 8,
FP32 optimizer state). The pretrain recipe keeps Beam's own bounded settings.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.common import ModelConfig
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config, default_tulu3_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


BEAM_HF_MODEL_ID = "reflection/Beam-1.0-501B-A23B"
BEAM_HF_MODEL_REVISION = "06641076793dad51b8907f7eb2a14f4e08fd3ec8"  # pragma: allowlist secret

# qwen3_30b_a3b_convergence_v2 data selection: Tulu 3 train[:10000] at a pinned dataset revision.
TULU3_SPLIT = "train[:10000]"
TULU3_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"  # pragma: allowlist secret
CONVERGENCE_V2_SEQ_LENGTH = 8192
CONVERGENCE_V2_GLOBAL_BATCH_SIZE = 8


def _beam_model_config(hf_path: str | None) -> ModelConfig:
    """Build the Beam model config from the pinned HF checkpoint, or from ``hf_path`` when given."""
    if hf_path is None:
        bridge = AutoBridge.from_hf_pretrained(BEAM_HF_MODEL_ID, revision=BEAM_HF_MODEL_REVISION)
    else:
        bridge = AutoBridge.from_hf_pretrained(hf_path)
    return bridge.get_model_config()


def _set_beam_tokenizer(cfg: ConfigContainer, hf_path: str | None) -> None:
    """Point the tokenizer at the same checkpoint as the model."""
    if hf_path is None:
        cfg.tokenizer.tokenizer_model = BEAM_HF_MODEL_ID
        cfg.tokenizer.hf_tokenizer_kwargs = {"revision": BEAM_HF_MODEL_REVISION}
    else:
        cfg.tokenizer.tokenizer_model = hf_path


def _set_beam_parallelism(
    cfg: ConfigContainer,
    *,
    tensor_parallel: int,
    pipeline_parallel: int,
    expert_parallel: int,
    seq_length: int,
) -> None:
    """Apply the parallel layout and the Beam execution settings shared by every recipe."""
    model = cfg.model
    model.tensor_model_parallel_size = tensor_parallel
    model.pipeline_model_parallel_size = pipeline_parallel
    model.pipeline_dtype = torch.bfloat16 if pipeline_parallel > 1 else None
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.cp_comm_type = "a2a"
    model.expert_model_parallel_size = expert_parallel
    model.expert_tensor_parallel_size = 1
    model.sequence_parallel = tensor_parallel > 1
    model.seq_length = seq_length

    model.transformer_impl = "transformer_engine"
    model.moe_token_dispatcher_type = "alltoall"
    model.moe_router_force_load_balancing = False
    model.recompute_granularity = "full"
    model.recompute_method = "uniform"
    model.recompute_num_layers = 1


def _set_convergence_v2_optimizer(cfg: ConfigContainer, *, lr: float) -> None:
    """Apply the qwen3_30b_a3b_convergence_v2 optimizer, schedule and precision fingerprint."""
    cfg.mixed_precision = bf16_mixed()

    cfg.train.train_iters = 100
    cfg.train.global_batch_size = CONVERGENCE_V2_GLOBAL_BATCH_SIZE
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.rng.seed = 5678

    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.lr_decay_style = "cosine"
    # Effective weight decay is the constant 0.033 applied by the scheduler, as in the Qwen anchor recipe.
    cfg.scheduler.start_weight_decay = 0.033
    cfg.scheduler.end_weight_decay = 0.033
    cfg.scheduler.weight_decay_incr_style = "constant"

    cfg.optimizer.lr = lr
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True

    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.logger.log_interval = 1


def _set_convergence_v2_dataset(cfg: ConfigContainer, *, pad_seq_to_mult: int) -> None:
    """Use the pinned 8K offline-packed Tulu 3 selection of qwen3_30b_a3b_convergence_v2."""
    cfg.dataset = default_tulu3_config(
        seq_length=CONVERGENCE_V2_SEQ_LENGTH,
        enable_offline_packing=True,
        pad_seq_to_mult=pad_seq_to_mult,
    )
    cfg.dataset.hf_dataset.split = TULU3_SPLIT
    cfg.dataset.hf_dataset.load_kwargs = {"revision": TULU3_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.seed = 1234


def beam_pretrain_128gpu_h200_bf16_config(hf_path: str | None = None) -> ConfigContainer:
    """Return a pre-training config for Beam on 16 H200 nodes.

    Parallelism: TP=2, PP=1, EP=128, SP (DP=64). GBS 256 at 4K tokens is four micro-batches per data-parallel
    rank. The precision-aware optimizer keeps BF16 gradients and Adam moments, which is what fits 501B
    parameters on 128 H200 GPUs; the recipe never selects a pretrained checkpoint.

    Args:
        hf_path: Optional local HF checkpoint used instead of the pinned Hub revision.

    Returns:
        ConfigContainer for Beam pre-training.
    """
    cfg = _pretrain_common()
    cfg.model = _beam_model_config(hf_path)
    _set_beam_tokenizer(cfg, hf_path)
    _set_beam_parallelism(cfg, tensor_parallel=2, pipeline_parallel=1, expert_parallel=128, seq_length=4096)

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8

    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 1e-5
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.mixed_precision = bf16_mixed()
    # The mixed-precision setup copies this onto the DDP config at runtime.
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}
    return cfg


def beam_sft_128gpu_h200_bf16_config(hf_path: str | None = None) -> ConfigContainer:
    """Return a full SFT config for Beam on 16 H200 nodes under ``qwen3_30b_a3b_convergence_v2``.

    Parallelism: TP=8, PP=2, EP=64, SP (DP=8), so each data-parallel rank takes one 8K pack of the
    eight per step. The contract's GBS 8 caps DP at 8; reaching the 16 model-parallel ranks with TP and PP
    rather than CP keeps the contract's loss normalization (CP > 1 forces per-token loss for finetuning).

    Args:
        hf_path: Optional local HF checkpoint used instead of the pinned Hub revision.

    Returns:
        ConfigContainer for Beam full-parameter SFT.
    """
    cfg = _sft_common()
    cfg.model = _beam_model_config(hf_path)
    _set_beam_tokenizer(cfg, hf_path)
    _set_beam_parallelism(
        cfg, tensor_parallel=8, pipeline_parallel=2, expert_parallel=64, seq_length=CONVERGENCE_V2_SEQ_LENGTH
    )
    # lcm(CP multiple 1, sequence-parallel multiple CP*TP = 8)
    _set_convergence_v2_dataset(cfg, pad_seq_to_mult=8)
    _set_convergence_v2_optimizer(cfg, lr=5e-6)

    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}
    return cfg


def beam_peft_16gpu_h200_bf16_config(
    peft_scheme: str | PEFT = "lora",
    hf_path: str | None = None,
) -> ConfigContainer:
    """Return a PEFT config for Beam on 2 H200 nodes under ``qwen3_30b_a3b_convergence_v2``.

    Parallelism: TP=2, PP=1, EP=16, SP (DP=8). LoRA/DoRA use rank 8, alpha 16 and no dropout on the
    attention projections ``linear_qkv`` (whose output also carries Beam's attention output gate) and
    ``linear_proj``; the base model stays frozen.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'.
        hf_path: Optional local HF checkpoint used instead of the pinned Hub revision.

    Returns:
        ConfigContainer for Beam parameter-efficient fine-tuning.
    """
    cfg = _peft_common()
    cfg.model = _beam_model_config(hf_path)
    _set_beam_tokenizer(cfg, hf_path)
    _set_beam_parallelism(
        cfg, tensor_parallel=2, pipeline_parallel=1, expert_parallel=16, seq_length=CONVERGENCE_V2_SEQ_LENGTH
    )

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
    cfg.peft = peft_cfg

    # lcm(CP multiple 1, sequence-parallel multiple CP*TP = 2)
    _set_convergence_v2_dataset(cfg, pad_seq_to_mult=2)
    _set_convergence_v2_optimizer(cfg, lr=1e-4)

    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}
    return cfg
