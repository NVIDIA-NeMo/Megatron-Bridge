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

"""EARLY, UNOPTIMIZED text-only training with frozen Qwen3.8 n-gram tables."""

from megatron.bridge.models import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_muon_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


def qwen38_flash_next_text_frozen_ngram_pretrain_32gpu_gb200_bf16_muon_config(
    *, hf_path: str = "Qwen/Qwen3.8-Flash-Next"
) -> ConfigContainer:
    """Return the EARLY, UNOPTIMIZED 4K frozen-PLE Muon recipe for 32 GB200 GPUs.

    Uses TP2/SP, EP32/ETP1, HybridEP and fused MoE operations with natural
    routing. Only the hashed n-gram tables are frozen and sharded across EP;
    PLE projections, norms and convolution remain trainable. The language
    implementation is shared with the vision-language model.

    Muon uses layer-wise distribution and full-matrix, duplicated TP
    orthogonalization; non-matrix and embedding weights use its Adam fallback.
    The 30-update schedule (five-step LR warmup) is for performance bring-up,
    not convergence. Use updates 21-30 for the final ten-step timing window. The
    verification card records the measured result and its evidence limits.

    Supply indexed text through ``dataset.data_path`` and an imported text-only
    Megatron checkpoint through ``checkpoint.pretrained_checkpoint`` to the
    standard pretrain runner. Pinned HF initialization is also supported.
    Start a fresh optimizer from these model weights. CUDA
    graphs, recomputation, checkpoint saving and evaluation remain disabled.
    Set ``OMP_NUM_THREADS=8`` to match the measured host-thread configuration.

    Args:
        hf_path: Public model ID or local snapshot containing config/tokenizer
            assets. Supply the pretrained weights separately at launch.

    Returns:
        Text-only training configuration with frozen EP-sharded n-gram tables.
    """
    revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"  # pragma: allowlist secret
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, revision=revision, text_only=True).get_model_config()
    model = cfg.model
    model.ple_freeze_ngram_embeddings = True
    model.ple_embedding_sharding = "ep"
    model.tensor_model_parallel_size = 2
    model.sequence_parallel = True
    model.expert_model_parallel_size = 32
    model.expert_tensor_parallel_size = 1
    model.pipeline_model_parallel_size = model.context_parallel_size = 1
    model.seq_length = 4096
    model.gradient_accumulation_fusion = True
    model.bias_activation_fusion = True
    model.moe_token_dispatcher_type = "flex"
    model.moe_flex_dispatcher_backend = "hybridep"
    model.moe_permute_fusion = True
    model.moe_permute_fusion_into_hybridep = True
    model.moe_router_fusion = True
    model.moe_shared_expert_overlap = False
    model.moe_grouped_gemm = True
    model.moe_aux_loss_coeff = 0.0
    model.moe_router_force_load_balancing = False
    model.hidden_dropout = model.attention_dropout = 0.0
    model.calculate_per_token_loss = True
    model.recompute_granularity = None
    model.cuda_graph_impl = "none"

    cfg.train.train_iters = 30
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 32
    cfg.dataset.seq_length = 4096
    cfg.dataset.split = "100,0,0"
    cfg.dataset.num_workers = 2
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = cfg.train.train_iters + 1
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": revision}

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.checkpoint.load = cfg.checkpoint.save = None
    cfg.checkpoint.save_interval = 0
    cfg.checkpoint.finetune = True
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = None
    cfg.logger.tensorboard_log_interval = 1
    cfg.logger.log_throughput_to_tensorboard = True
    cfg.optimizer, cfg.scheduler = distributed_muon_with_cosine_annealing(
        muon_momentum=0.95,
        muon_use_nesterov=True,
        muon_scale_mode="unit_rms_norm",
        muon_fp32_matmul_prec="medium",
        muon_num_ns_steps=5,
        muon_tp_mode="duplicated",
        muon_extra_scale_factor=0.2,
        muon_scalar_optimizer="adam",
        lr_warmup_iters=5,
        lr_decay_iters=cfg.train.train_iters,
        weight_decay=0.0,
        max_lr=1e-4,
        min_lr=1e-4,
        clip_grad=1.0,
    )
    cfg.optimizer.muon_split_qkv = False
    cfg.optimizer.use_distributed_optimizer = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.scheduler.lr_decay_style = "constant"
    cfg.scheduler.start_weight_decay = cfg.scheduler.end_weight_decay = 0.0
    # Muon's layer-wise distribution is separate from the flat distributed Adam
    # buffers selected by this DDP flag. Parameter-gather overlap stays off.
    cfg.ddp.use_distributed_optimizer = False
    return cfg
