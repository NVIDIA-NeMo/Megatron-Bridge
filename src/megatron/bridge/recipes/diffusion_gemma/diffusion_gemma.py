# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Training recipe for the public DiffusionGemma 26B-A4B checkpoint."""

from megatron.bridge import AutoBridge
from megatron.bridge.models.diffusion_gemma.data import DiffusionGemmaDatasetConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig


_HF_PATH = "google/diffusiongemma-26B-A4B-it"


def diffusion_gemma_26b_sft_config(
    *,
    train_path: str,
    validation_path: str | None = None,
    test_path: str | None = None,
    hf_model_path: str = _HF_PATH,
) -> ConfigContainer:
    """Return the 8-GPU EP=8 block-diffusion SFT configuration."""
    bridge = AutoBridge.from_hf_pretrained(hf_model_path)
    provider = bridge.to_megatron_provider(load_weights=True)
    cfg = ConfigContainer(
        model=provider,
        train=TrainingConfig(train_iters=1000, global_batch_size=8, micro_batch_size=1),
        optimizer=OptimizerConfig(lr=5e-5, min_lr=5e-6, weight_decay=0.0, use_distributed_optimizer=True),
        scheduler=SchedulerConfig(
            lr_warmup_iters=20,
            lr_decay_iters=1000,
            lr_decay_style="cosine",
            start_weight_decay=0.0,
            end_weight_decay=0.0,
        ),
        dataset=DiffusionGemmaDatasetConfig(
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            hf_processor_path=hf_model_path,
        ),
        logger=LoggerConfig(log_interval=1),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=provider.text_config.vocab_size),
        checkpoint=CheckpointConfig(
            save="diffusiongemma-checkpoints",
            load="diffusiongemma-checkpoints",
            save_interval=100,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        ddp=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            check_for_nan_in_grad=True,
        ),
        mixed_precision="bf16_mixed",
    )
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.diffusion_loss_variant = "base-sft"
    cfg.model.diffusion_self_conditioning_prob = 0.5
    cfg.model.diffusion_encoder_loss_weight = 1.0
    cfg.model.diffusion_decoder_loss_weight = 1.0
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    # Gemma 4 global attention uses an unfused path for long multimodal
    # contexts. Full per-layer recompute keeps large-image prompts with long
    # prior target blocks within 8×H200 memory; see the DiffusionGemma docs.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=cfg.model.text_config.vocab_size)
    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    if validation_path:
        cfg.validation.eval_interval = 100
        cfg.validation.eval_iters = 16
    return cfg
