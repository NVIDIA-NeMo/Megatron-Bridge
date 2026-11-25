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


"""EARLY, UNOPTIMIZED fixed-4K image SFT with frozen EP-sharded PLE tables."""

from megatron.bridge.data.builders.energon import EnergonDatasetConfig, QwenVLEnergonTaskEncoderConfig
from megatron.bridge.models import AutoBridge
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import bf16_mixed


def qwen38_flash_next_sft_64gpu_gb200_bf16_config(
    *,
    hf_path: str = "Qwen/Qwen3.8-Flash-Next",
) -> ConfigContainer:
    """Return EARLY, UNOPTIMIZED image SFT on 64 GB200 GPUs with TP4/SP and EP64.

    Use the shared Qwen VL forward step with ``--step-func qwen3_vl_step``.
    Supply ``dataset.path`` and ``checkpoint.pretrained_checkpoint`` at launch;
    the latter accepts the full local Hugging Face checkpoint. Standard
    finetuning loads weights and refreshes optimizer master parameters.

    The language model, vision encoder and projector train together. PLE n-gram
    lookup tables are frozen and EP-sharded; PLE projections, normalization and
    convolution remain trainable. QSA indexer weights stay frozen because hard
    selection is detached from the training objective.

    Supply prepared Energon image conversations, such as DataComp image-caption
    samples. Batches are padded to a fixed 4096 tokens with no packing.
    Synchronized manual garbage collection avoids rank-dependent Python pauses.
    Checkpoint saving, evaluation, recomputation and CUDA graphs are disabled.
    Standard logs record loss, gradient norm, iteration time and model TFLOP/s.

    Args:
        hf_path: Public model ID or local directory containing its HF config
            and processor assets. Weights are supplied separately at launch.

    Returns:
        Declarative training configuration for the full vision-language model.
    """
    revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"  # pragma: allowlist secret
    model = AutoBridge.from_hf_pretrained(hf_path, revision=revision, text_only=False).get_model_config()
    model.perform_initialization = False
    model.ple_freeze_ngram_embeddings = True
    model.ple_embedding_sharding = "ep"
    model.tensor_model_parallel_size = 4
    model.expert_model_parallel_size = 64
    model.expert_tensor_parallel_size = 1
    model.pipeline_model_parallel_size = model.context_parallel_size = 1
    model.sequence_parallel = True
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
    model.hidden_dropout = model.attention_dropout = 0.0
    model.calculate_per_token_loss = True
    model.recompute_granularity = None
    model.cuda_graph_impl = "none"
    vision = model.vision_config
    encoder = QwenVLEnergonTaskEncoderConfig(
        hf_processor_path=hf_path,
        hf_processor_revision=revision,
        patch_size=vision["patch_size"],
        temporal_patch_size=vision["temporal_patch_size"],
        spatial_merge_size=vision["spatial_merge_size"],
        min_pixels=65536,
        max_pixels=262144,
        max_num_images=1,
        max_num_frames=1,
        max_visual_tokens=4096 - 1,
    )
    dataset = EnergonDatasetConfig(
        path=None,
        seq_length=4096,
        micro_batch_size=1,
        task_encoder=encoder,
        num_workers=2,
        persistent_workers=True,
        do_validation=False,
        enable_in_batch_packing=False,
        packing_buffer_size=None,
        pad_to_max_length=True,
        pad_to_multiple_of=8,
        shuffle_buffer_size=100,
    )
    precision = bf16_mixed()
    precision.grad_reduce_in_fp32 = True
    cfg = ConfigContainer(
        model=model,
        train=TrainingConfig(
            train_iters=1000000,
            micro_batch_size=1,
            global_batch_size=16,
            manual_gc=True,
            manual_gc_interval=100,
        ),
        validation=ValidationConfig(eval_iters=0, eval_interval=1000000 + 1),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=1e-6,
            min_lr=1e-6,
            weight_decay=0.0,
            clip_grad=1.0,
            bf16=True,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=False,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="constant",
            lr_decay_iters=1000000,
            lr_warmup_iters=0,
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            weight_decay_incr_style="constant",
        ),
        ddp=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=precision.grad_reduce_in_fp32,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            check_for_nan_in_grad=True,
            average_in_collective=False,
        ),
        dataset=dataset,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_path,
            use_tokenizer_vocab_size=False,
            hf_tokenizer_kwargs={"revision": revision},
        ),
        checkpoint=CheckpointConfig(
            save=None, load=None, pretrained_checkpoint=None, save_interval=0, also_save_hf_checkpoint=False
        ),
        logger=LoggerConfig(
            log_interval=1,
            log_throughput=True,
            tensorboard_dir=None,
            tensorboard_log_interval=1,
            log_throughput_to_tensorboard=True,
            log_timers_to_tensorboard=True,
        ),
        rng=RNGConfig(seed=1234),
        mixed_precision=precision,
    )
    return cfg
