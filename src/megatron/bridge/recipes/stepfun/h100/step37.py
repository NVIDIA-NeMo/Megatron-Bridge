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

"""Step3.7 (`stepfun-ai/Step-3.7-Flash`) recipe.

Only the Flickr8k SFT path is supported. ``Step37Model.forward`` takes
``list[ImageForInsert]`` directly, and the data path is the
self-contained ``Step37Flickr8kSFTDataProvider`` (HF datasets / processor
not involved).
"""

from __future__ import annotations

import os

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.stepfun.data.flickr8k import Step37Flickr8kSFTDataProvider
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import (
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
)
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend


_STEP37_HF_PATH = "stepfun-ai/Step-3.7-Flash"
_STEP37_FLICKR8K_SAMPLE_COUNT = 8
_STEP37_FLICKR8K_MAX_PACKING_SEQLEN = 2048
_STEP37_FLICKR8K_SEQLEN_DIVISIBLE_BY = 64
_STEP37_FLICKR8K_OVERSIZE_POLICY = "drop"
_STEP37_FLICKR8K_DATASET_SAMPLING = "random"
_STEP37_FLICKR8K_CACHE_DIR = ".cache/step37_flickr8k"
_STEP37_FLICKR8K_PROMPT = "Describe this image in one sentence."
_STEP37_FLICKR8K_SMOKE_CACHE_DIR = ".cache/step37_flickr8k_smoke"
_STEP37_FLICKR8K_SMOKE_FIXED_PACK_IDX = 0
_STEP37_FLICKR8K_SMOKE_TRAIN_ITERS = 100
_STEP37_FLICKR8K_SMOKE_MAX_LR = 5e-3


# =============================================================================
# Step3.7 SFT Configuration — Flickr8k packed pipeline
# =============================================================================


def step37_sft_64gpu_h100_bf16_flickr8k_config() -> ConfigContainer:
    """Step3.7 SFT recipe — the only supported Step3.7 path.

    Uses the Flickr8k packed pipeline:

    - ``cfg.dataset`` is :class:`Step37Flickr8kSFTDataProvider` (sync
      packing, no async wrapper, no ``DirectHFSFTDatasetConfig``).
    - ``--step_func step37_flickr8k_step`` consumes the packed dict and
      passes ``list[ImageForInsert]`` straight to ``Step37Model.forward``.
    - ``micro_batch_size`` is pinned at ``1`` — each pack already aggregates
      multiple sub-seqs via ``cu_seqlens``.
    - Tokenizer loaded with ``trust_remote_code=False``; **no HF custom
      Python code** runs in the data path.

    The default train split is limited to 8 samples for smoke coverage. Use CLI
    overrides such as ``dataset.sample_count=null`` for a full Flickr8k run.
    """
    # Start from the generic SFT baseline (gives us cfg.train / cfg.optimizer
    # / cfg.scheduler / cfg.ddp / cfg.checkpoint / cfg.logger placeholders),
    # then override every VLM/Step3.7-specific field. We don't use
    # ``_sft_common_vlm`` because it forces ``DirectHFSFTDatasetConfig``
    # which is exactly the layer we're replacing.
    cfg = _sft_common()

    # ── Model: AutoBridge load from HF id or local snapshot ──────────────
    cfg.model = AutoBridge.from_hf_pretrained(_STEP37_HF_PATH).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = _STEP37_FLICKR8K_MAX_PACKING_SEQLEN
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend=None)

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # ── Training (smoke defaults; bump train_iters via CLI for real runs) ─
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1  # ← pinned for the packed flickr8k path
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 10

    # ── Optimizer (full SFT — low LR) ─────────────────────────────────────
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=5e-6,
        min_lr=5e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # ── DDP (no overlap for VLMs) ─────────────────────────────────────────
    cfg.ddp = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        average_in_collective=True,
        data_parallel_sharding_strategy="optim_grads_params",
        use_distributed_optimizer=True,
    )

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"

    # ── Tokenizer (NullTokenizer placeholder; CLI must set padded_vocab_size) ─
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="NullTokenizer",
        vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
    )

    # ── Logger ────────────────────────────────────────────────────────────
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    cfg.logger = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    cfg.checkpoint.save_interval = 500
    cfg.checkpoint.save = checkpoint_dir
    cfg.checkpoint.load = checkpoint_dir
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    cfg.rng = RNGConfig(seed=1234)

    # ── Dataset: Flickr8k packed provider ─────────────────────────────────
    cfg.dataset = Step37Flickr8kSFTDataProvider(
        tokenizer_path=_STEP37_HF_PATH,
        repo_id="intro/flickr8k",
        split="train",
        sample_count=_STEP37_FLICKR8K_SAMPLE_COUNT,
        cache_dir=_STEP37_FLICKR8K_CACHE_DIR,
        prompt=_STEP37_FLICKR8K_PROMPT,
        max_packing_seqlen=_STEP37_FLICKR8K_MAX_PACKING_SEQLEN,
        seqlen_divisible_by=_STEP37_FLICKR8K_SEQLEN_DIVISIBLE_BY,
        oversize_policy=_STEP37_FLICKR8K_OVERSIZE_POLICY,  # type: ignore[arg-type]
        dataset_sampling=_STEP37_FLICKR8K_DATASET_SAMPLING,  # type: ignore[arg-type]
        seq_length=_STEP37_FLICKR8K_MAX_PACKING_SEQLEN,
        num_workers=0,  # sync — no async prefetch (per user requirement)
        persistent_workers=False,
        pin_memory=True,
        data_sharding=False,
    )

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# =============================================================================
# Step3.7 SFT — smoke config (single fixed pack, fast loss-drop)
# =============================================================================


def step37_sft_4gpu_h100_bf16_flickr8k_smoke_config() -> ConfigContainer:
    """Smoke variant of :func:`step37_flickr8k_sft_config` — the same packed
    sample on every DP rank, every step. Deterministic and tiny: it repeats
    pack[``fixed_pack_idx``] indefinitely so the loss curve visibly drops as
    the model overfits a single batch.

    Differences vs. the regular SFT config:

    - ``dataset.fixed_pack_idx`` pins ``__getitem__`` → identical input
      across every DP rank and every iteration.
    - ``dataset.dataset_sampling = "sequential"`` for reproducibility.
    - ``max_lr`` bumped 5e-6 → 5e-3 so the overfit happens within
      ``train_iters`` steps.
    - Language model unfrozen (the regular config freezes it); vision tower
      stays frozen (overfitting on the projector + LM is enough and avoids
      the PE-G/14 backward cost).
    - ``log_interval=1``, eval disabled, no mid-run checkpoint save.

    The smoke recipe repeats pack 0 for 100 iterations with a high learning
    rate so loss can drop quickly.
    """
    cfg = step37_sft_64gpu_h100_bf16_flickr8k_config()
    cfg.dataset.sample_count = _STEP37_FLICKR8K_SAMPLE_COUNT
    cfg.dataset.cache_dir = _STEP37_FLICKR8K_SMOKE_CACHE_DIR
    cfg.dataset.dataset_sampling = "sequential"
    cfg.dataset.fixed_pack_idx = _STEP37_FLICKR8K_SMOKE_FIXED_PACK_IDX
    cfg.train.train_iters = _STEP37_FLICKR8K_SMOKE_TRAIN_ITERS

    cfg.model.mtp_num_layers = 0
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.num_layers_in_first_pipeline_stage = 12
    cfg.model.num_layers_in_last_pipeline_stage = 9
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.variable_seq_lengths = True
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_permute_fusion = True
    cfg.model.attention_dropout = 0.0
    cfg.model.rotary_percent = 0.5
    cfg.model.window_size = [512, 0]
    cfg.model.window_attn_skip_freq = [
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
    ]
    cfg.model.moe_layer_freq = [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    cfg.model.rotary_base_per_layer = [
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
        10000,
        10000,
        10000,
        5000000,
    ]
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 16

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=max(1, min(10, _STEP37_FLICKR8K_SMOKE_TRAIN_ITERS // 5)),
        lr_decay_iters=_STEP37_FLICKR8K_SMOKE_TRAIN_ITERS,
        max_lr=_STEP37_FLICKR8K_SMOKE_MAX_LR,
        min_lr=_STEP37_FLICKR8K_SMOKE_MAX_LR / 10.0,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.logger.log_interval = 1
    cfg.validation.eval_interval = _STEP37_FLICKR8K_SMOKE_TRAIN_ITERS + 1
    cfg.validation.eval_iters = 0
    cfg.checkpoint.save_interval = _STEP37_FLICKR8K_SMOKE_TRAIN_ITERS + 1

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "step37_sft_4gpu_h100_bf16_flickr8k_smoke_config",
    "step37_sft_64gpu_h100_bf16_flickr8k_config",
]
