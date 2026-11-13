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

"""Exercise Bridge's real training, DDP, optimizer, and checkpoint pipeline.

Launch in fresh processes: six uninterrupted steps, three steps plus a save,
then a resume through step six. The train horizon/scheduler are identical in
all launches; ``--stop-after`` only controls early exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from megatron.core.transformer.enums import AttnBackend
from toy_parity import toy_config

from megatron.bridge import AutoBridge
from megatron.bridge.models.diffusion_gemma.data import MockDiffusionGemmaDatasetConfig
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.diffusion_gemma_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.common_utils import print_rank_0


class TrainingReport(Callback):
    """Write per-step evidence from the real training loop, not a fake state."""

    def __init__(self, output: Path, *, record_performance: bool = False) -> None:
        self.output = output
        self.rows: list[dict] = []
        self.start_step = -1
        self.record_performance = record_performance
        self.step_started_at = 0.0
        self.trainable_parameters = 0

    def on_train_start(self, context: CallbackContext) -> None:
        self.start_step = context.state.train_state.step
        self.trainable_parameters = sum(
            parameter.numel() for chunk in context.model for parameter in chunk.parameters() if parameter.requires_grad
        )
        print_rank_0(f"SMOKE_START_STEP={self.start_step}")

    def on_train_step_start(self, context: CallbackContext) -> None:
        if self.record_performance:
            torch.cuda.synchronize()
            self.step_started_at = time.perf_counter()

    def on_train_step_end(self, context: CallbackContext) -> None:
        metrics = {key: float(value.detach()) for key, value in (context.loss_dict or {}).items()}
        row = {
            "step": context.state.train_state.step + 1,
            "metrics": metrics,
            "grad_norm": float(context.grad_norm) if context.grad_norm is not None else None,
            "skipped": bool(context.skipped_iter),
            "consumed_samples_before_step": context.state.train_state.consumed_train_samples,
        }
        if self.record_performance:
            torch.cuda.synchronize()
            row["seconds"] = time.perf_counter() - self.step_started_at
            row["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            row["max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        if not metrics or not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise AssertionError(f"Non-finite or missing training metrics: {row}")
        if row["skipped"]:
            raise AssertionError(f"Optimizer skipped the training step: {row}")
        self.rows.append(row)
        print_rank_0("SMOKE_STEP=" + json.dumps(row))

    def on_train_end(self, context: CallbackContext) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        report = {
            "start_step": self.start_step,
            "end_step": context.state.train_state.step,
            "consumed_train_samples": context.state.train_state.consumed_train_samples,
            "steps": self.rows,
            "rank0_trainable_parameters": self.trainable_parameters,
            "tp": context.state.cfg.model.tensor_model_parallel_size,
            "ep": context.state.cfg.model.expert_model_parallel_size,
            "distributed_optimizer": context.state.cfg.optimizer.use_distributed_optimizer,
            "frozen_base": False,
            "passed": bool(self.rows) and context.state.train_state.step == context.state.cfg.train.train_iters,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2))
        print_rank_0("SMOKE_REPORT=" + json.dumps(report))


def main() -> None:
    """Build a tiny or full-checkpoint smoke config and execute pretrain()."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-steps", type=int, default=6)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--distributed-optimizer", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.model_path is None:
        hf_config = toy_config()
        provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
    else:
        bridge = AutoBridge.from_hf_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        hf_config = bridge.hf_pretrained.config
        provider = bridge.to_megatron_provider(load_weights=not args.resume)
    cfg = ConfigContainer(
        model=provider,
        train=TrainingConfig(train_iters=args.train_steps, global_batch_size=4, micro_batch_size=1),
        optimizer=OptimizerConfig(lr=1e-4, min_lr=1e-4),
        scheduler=SchedulerConfig(lr_decay_iters=args.train_steps, lr_warmup_iters=0),
        dataset=MockDiffusionGemmaDatasetConfig(),
        logger=LoggerConfig(log_interval=1),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=hf_config.text_config.vocab_size),
        checkpoint=CheckpointConfig(ckpt_format="torch_dist"),
    )
    cfg.model.tensor_model_parallel_size = args.tp
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = args.ep
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = args.tp > 1
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.diffusion_self_conditioning_prob = 0.5
    cfg.model.diffusion_loss_variant = "base-sft"
    cfg.model.diffusion_encoder_loss_weight = 1.0 if args.model_path is not None else 0.0
    cfg.model.attention_backend = AttnBackend.unfused
    cfg.model.seq_length = 32
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cuda_graph_impl = "none"
    cfg.model.recompute_granularity = None
    cfg.train.train_iters = args.train_steps
    cfg.train.exit_interval = args.stop_after
    cfg.train.global_batch_size = 2 * int(os.environ.get("WORLD_SIZE", "1")) // args.tp
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_iters = args.eval_iters
    cfg.validation.eval_interval = args.eval_interval
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 1e-4
    cfg.optimizer.weight_decay = 0.0
    cfg.optimizer.use_distributed_optimizer = args.distributed_optimizer or args.model_path is not None
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = args.train_steps
    cfg.scheduler.lr_decay_style = "constant"
    cfg.scheduler.start_weight_decay = 0.0
    cfg.scheduler.end_weight_decay = 0.0
    cfg.ddp = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=cfg.optimizer.use_distributed_optimizer,
        check_for_nan_in_grad=True,
    )
    cfg.mixed_precision = "bf16_mixed"
    text = hf_config.text_config
    vision = hf_config.vision_config
    cfg.dataset = MockDiffusionGemmaDatasetConfig(
        canvas_length=16 if args.model_path is None else 64,
        prompt_length=8,
        completion_length=12 if args.model_path is None else 32,
        token_range=(10, min(text.vocab_size - 20, 1900)),
        pad_token_id=text.pad_token_id,
        eos_token_id=1,
        bos_token_id=2,
        image_token_id=hf_config.image_token_id,
        boi_token_id=hf_config.boi_token_id,
        eoi_token_id=hf_config.eoi_token_id,
        image_patches_per_side=4 if args.model_path is None else 6,
        patch_size=vision.patch_size,
        pooling_kernel_size=vision.pooling_kernel_size,
        pad_to_multiple_of=args.tp,
        seed=1234,
    )
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.vocab_size = text.vocab_size
    cfg.tokenizer.tokenizer_model = None
    cfg.logger.tensorboard_dir = None
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = False
    cfg.logger.save_config_filepath = str(args.work_dir / "config.yaml")
    cfg.checkpoint.save = None if args.no_save else str(args.work_dir / "checkpoints")
    cfg.checkpoint.load = str(args.work_dir / "checkpoints") if args.resume else None
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.save_interval = 3
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.load_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.load_rng = True
    cfg.checkpoint.finetune = False
    cfg.checkpoint.async_save = False
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.rng.seed = 1234
    args.work_dir.mkdir(parents=True, exist_ok=True)
    pretrain(
        cfg, forward_step, callbacks=[TrainingReport(args.report, record_performance=args.model_path is not None)]
    )


if __name__ == "__main__":
    main()
