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


"""Early Qwen3.8-Flash-Next image SFT recipe; performance is unoptimized."""

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
    """Return full-model image SFT on 64 GB200 GPUs with TP8/SP and EP64.

    Early recipe: performance is unoptimized. Its optimizer/data settings come
    from the 5,349-step Nemotron-Image-Training-v3 ECD subset run. Use the shared
    Qwen VL forward step through ``scripts/training/run_recipe.py``. Supply
    ``dataset.path`` and ``checkpoint.pretrained_checkpoint`` at launch; the
    latter accepts the full local Hugging Face checkpoint. Standard finetuning
    loads the weights and refreshes optimizer master parameters before training.

    Prepare complete sequence-2048 samples with
    ``scripts/data/prepare_nemotron_image_v3_subset.py``. Packing, checkpoint
    saving, evaluation, and performance optimizations are disabled. Standard
    training logs record loss, gradient norm, iteration time, and estimated
    model TFLOP/s/GPU. This recipe is not a numerical-parity test.

    Args:
        hf_path: Public model ID or local directory containing its HF config
            and processor assets. Weights are supplied separately at launch.

    Returns:
        Declarative training configuration for the full vision-language model.
    """
    revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"  # pragma: allowlist secret
    model = AutoBridge.from_hf_pretrained(hf_path, revision=revision, text_only=False).get_model_config()
    model.perform_initialization = False
    model.tensor_model_parallel_size = 8
    model.expert_model_parallel_size = 64
    model.expert_tensor_parallel_size = 1
    model.pipeline_model_parallel_size = model.context_parallel_size = 1
    model.sequence_parallel = True
    model.seq_length = 2048
    model.gradient_accumulation_fusion = False
    model.moe_token_dispatcher_type = "allgather"
    model.moe_permute_fusion = model.moe_shared_expert_overlap = False
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
        max_visual_tokens=2048 - 1,
    )
    dataset = EnergonDatasetConfig(
        path=None,
        seq_length=2048,
        micro_batch_size=1,
        task_encoder=encoder,
        num_workers=2,
        persistent_workers=True,
        do_validation=False,
        enable_in_batch_packing=False,
        packing_buffer_size=None,
        pad_to_max_length=False,
        pad_to_multiple_of=8,
        shuffle_buffer_size=100,
    )
    precision = bf16_mixed()
    precision.grad_reduce_in_fp32 = True
    cfg = ConfigContainer(
        model=model,
        train=TrainingConfig(train_iters=1000000, micro_batch_size=1, global_batch_size=8),
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
