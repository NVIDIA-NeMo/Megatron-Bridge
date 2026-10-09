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

"""Verify loaded Qwen3.8-Flash-Next language weights with short real-data pretraining.

Launch with the normal distributed launcher in the card's container environment.
The checkpoint must be a local complete HF snapshot, and data-prefix must identify
Megatron indexed text (.bin/.idx) tokenized with that snapshot's tokenizer. The
default topology is 32 GPUs: TP8, EP32, expert TP1, PP1, CP1, dense DP4.
This script writes small evidence files, never model checkpoints. Successful
training verifies a forward/backward/update path, not convergence or HF parity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.bridge.data.loaders import build_train_valid_test_datasets
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.models import AutoBridge
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
    runtime_config_update,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.training.optim import _get_scheduler
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import _get_num_moe_layers


logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _global_max(value: float) -> float:
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


class VerificationEvidence(Callback):
    """Record finite losses, nonzero gradients, and sampled FP32 master updates."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.loaded = False
        self.rows: list[dict[str, Any]] = []
        self.samples: list[torch.Tensor] = []
        self.before: list[torch.Tensor] = []
        self.started = 0.0
        self.step_started = 0.0

    def on_train_start(self, context: CallbackContext) -> None:
        """Capture runtime provenance and select small optimizer master samples."""
        import megatron.core

        import megatron.bridge

        if not self.loaded or context.optimizer is None:
            raise RuntimeError("HF loading and optimizer construction must complete before training.")
        # Ordinary distributed Adam exposes its FP32 master shards here. Small
        # dense parameters (norms/gates) reliably receive gradients; BF16 model
        # weights can round away a single update at this deliberately small LR.
        params = sorted(
            (p for p in context.optimizer.get_parameters() if p.numel() and p.dtype == torch.float32),
            key=lambda p: p.numel(),
        )
        self.samples = [p.detach().view(-1)[:256] for p in params[:32]]
        if _global_max(float(bool(self.samples))) == 0:
            raise RuntimeError("No FP32 optimizer master shards available for update verification.")
        grad_dtypes = {
            str(parameter.main_grad.dtype)
            for model in context.model
            for parameter in model.parameters()
            if getattr(parameter, "main_grad", None) is not None
        }
        if _global_max(float(grad_dtypes != {"torch.bfloat16"})):
            raise RuntimeError(f"Expected BF16 gradient buffers on every rank, found {grad_dtypes} locally.")
        self.started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        if dist.get_rank() == 0:
            self.args.output.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents accidental mixing with an earlier run.
            (self.args.output / "steps.jsonl").open("x").close()
            provenance = {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()
                },
                "world_size": dist.get_world_size(),
                "dense_data_parallel_size": dist.get_world_size() // self.args.tensor_parallel_size,
                "gpu": torch.cuda.get_device_name(),
                "gpu_memory_bytes": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                "packages": {
                    p: _version(p)
                    for p in ("torch", "transformers", "tokenizers", "megatron-core", "transformer-engine")
                },
                "bridge_commit": _git_revision(Path(megatron.bridge.__file__).parent)
                or os.environ.get("BRIDGE_SOURCE_COMMIT"),
                "core_commit": _git_revision(Path(megatron.core.__file__).parent),
                "runner_sha256": _sha256(Path(__file__)),
                "checkpoint_config_sha256": _sha256(self.args.checkpoint / "config.json"),
                "checkpoint_index_sha256": _sha256(self.args.checkpoint / "model.safetensors.index.json"),
                "data_index_sha256": _sha256(Path(str(self.args.data_prefix) + ".idx")),
                "data_bin_sha256": _sha256(Path(str(self.args.data_prefix) + ".bin")),
                "data_bin_bytes": Path(str(self.args.data_prefix) + ".bin").stat().st_size,
                "hf_model_id": "Qwen/Qwen3.8-Flash-Next",
                "hf_revision_reported": self.args.hf_revision,
                "container_image": os.environ.get("NEMO_CONTAINER_IMAGE"),
                "squashfs_sha256": os.environ.get("NEMO_SQUASHFS_SHA256"),
                "hf_weights_loaded_before_optimizer": True,
                "text_only": True,
                "observed_gradient_buffer_dtypes": sorted(grad_dtypes),
                "moe_configuration": {
                    "token_dispatcher_type": context.state.cfg.model.moe_token_dispatcher_type,
                    "permute_fusion": context.state.cfg.model.moe_permute_fusion,
                    "shared_expert_overlap": context.state.cfg.model.moe_shared_expert_overlap,
                },
                "optimizer": "distributed Adam with FP32 master weights and moments",
                "parameter_update_evidence": "global maximum change in sampled FP32 optimizer masters",
                "limitations": [
                    "No HF numerical parity",
                    "No convergence claim",
                    "No persisted checkpoint reload",
                    "No vision or MTP",
                    "Short sequences do not exercise QSA selection beyond its budget",
                ],
            }
            _write_json(self.args.output / "provenance.json", provenance)
        dist.barrier()

    def on_train_step_start(self, context: CallbackContext) -> None:
        """Snapshot bounded master-weight samples before the optimizer step."""
        self.before = [sample.clone() for sample in self.samples]
        self.step_started = time.monotonic()

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Fail the run on bad losses, absent gradients, skipped or absent updates."""
        losses = {key: float(value.item()) for key, value in (context.loss_dict or {}).items()}
        grad_norm = float(context.grad_norm) if context.grad_norm is not None else float("nan")
        deltas = [(sample - before).abs().max() for sample, before in zip(self.samples, self.before)]
        local_delta = torch.stack(deltas).max().item() if deltas else 0.0
        bad = (
            not losses
            or any(not math.isfinite(value) for value in losses.values())
            or not math.isfinite(grad_norm)
            or grad_norm <= 0
            or context.skipped_iter is None
            or bool(context.skipped_iter)
            or not math.isfinite(local_delta)
        )
        failed = bool(_global_max(float(bad)))
        delta = _global_max(local_delta if math.isfinite(local_delta) else 0.0)
        failed = failed or delta <= 0
        row = {
            "step": len(self.rows) + 1,
            "losses": {key: value if math.isfinite(value) else None for key, value in losses.items()},
            "grad_norm_before_clipping": grad_norm if math.isfinite(grad_norm) else None,
            "skipped_iter": None if context.skipped_iter is None else bool(context.skipped_iter),
            "sampled_master_update_max_abs_global": delta,
            "elapsed_seconds": time.monotonic() - self.step_started,
            "passed": not failed,
        }
        self.rows.append(row)
        if dist.get_rank() == 0:
            with (self.args.output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            logger.info("Verification step: %s", row)
        if failed:
            raise RuntimeError(
                "Verification failed: nonfinite loss/gradient, absent gradients, skipped or absent update."
            )

    def on_train_end(self, context: CallbackContext) -> None:
        """Write aggregate results only after every requested step succeeded."""
        if len(self.rows) != self.args.steps:
            raise RuntimeError(f"Expected {self.args.steps} training steps, observed {len(self.rows)}.")
        peak_allocated = _global_max(float(torch.cuda.max_memory_allocated()))
        peak_reserved = _global_max(float(torch.cuda.max_memory_reserved()))
        if dist.get_rank() == 0:
            _write_json(
                self.args.output / "summary.json",
                {
                    "passed": True,
                    "steps": len(self.rows),
                    "skipped_steps": 0,
                    "elapsed_seconds": time.monotonic() - self.started,
                    "first_loss": self.rows[0]["losses"],
                    "last_loss": self.rows[-1]["losses"],
                    "grad_norm_min": min(row["grad_norm_before_clipping"] for row in self.rows),
                    "grad_norm_max": max(row["grad_norm_before_clipping"] for row in self.rows),
                    "sampled_master_update_min_global": min(
                        row["sampled_master_update_max_abs_global"] for row in self.rows
                    ),
                    "peak_allocated_bytes_global": int(peak_allocated),
                    "peak_reserved_bytes_global": int(peak_reserved),
                    "forward_verified_by": "finite real-data training loss after HF weight import",
                    "backward_verified_by": "finite positive pre-clipping gradient norm",
                    "update_verified_by": "nonzero sampled FP32 master change at every step",
                },
            )


def parse_args() -> argparse.Namespace:
    """Read portable paths and bounded training/topology settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-revision", default="de4b8e4d43b917e7706784d8bb445c9af86a3540")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Finalize configuration without building a model or initializing distributed execution",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--expert-parallel-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    for name in (
        "steps",
        "tensor_parallel_size",
        "expert_parallel_size",
        "sequence_length",
        "micro_batch_size",
        "global_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if world % args.tensor_parallel_size or world % args.expert_parallel_size:
        parser.error("WORLD_SIZE must be divisible by TP and EP (expert TP is fixed at 1)")
    if args.global_batch_size % ((world // args.tensor_parallel_size) * args.micro_batch_size):
        parser.error("global-batch-size must be divisible by dense DP times micro-batch-size")
    if args.sequence_length % args.tensor_parallel_size:
        parser.error("sequence-length must be divisible by TP for sequence parallelism")
    for path in (
        args.checkpoint / "config.json",
        args.checkpoint / "model.safetensors.index.json",
        Path(str(args.data_prefix) + ".idx"),
        Path(str(args.data_prefix) + ".bin"),
    ):
        if not path.is_file():
            parser.error(f"Required local file is missing: {path}")
    if (args.output / "steps.jsonl").exists() or (args.output / "summary.json").exists():
        parser.error("output already contains verification evidence; choose a new directory")
    return args


def build_config(args: argparse.Namespace) -> tuple[ConfigContainer, VerificationEvidence]:
    """Construct a training config and lazy HF loader without building the model."""
    bridge = AutoBridge.from_hf_pretrained(args.checkpoint, text_only=True)
    model = bridge.get_model_config()
    # A caller-provided revision is provenance for a local snapshot; hashes in
    # provenance.json identify its actual metadata. No network resolution occurs.
    model.extra_checkpoint_metadata["hf_model_revision"] = args.hf_revision
    model.perform_initialization = False
    model.tensor_model_parallel_size = args.tensor_parallel_size
    model.expert_model_parallel_size = args.expert_parallel_size
    model.expert_tensor_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.context_parallel_size = 1
    model.sequence_parallel = args.tensor_parallel_size > 1
    model.seq_length = args.sequence_length
    model.gradient_accumulation_fusion = False
    # The RC4 SPCX variable-split all-to-all path corrupts this workload.
    # Use the supported all-gather path, with unfused empty-expert handling.
    model.moe_token_dispatcher_type = "allgather"
    model.moe_permute_fusion = False
    model.moe_shared_expert_overlap = False
    model.moe_grouped_gemm = True
    evidence = VerificationEvidence(args)

    def load_weights(models: list[torch.nn.Module]) -> list[torch.nn.Module]:
        bridge.load_hf_weights(models)
        evidence.loaded = True
        return models

    model.pre_wrap_hooks.append(load_weights)
    # A string preset would reset grad_reduce_in_fp32 to True at runtime.
    precision = bf16_mixed()
    precision.grad_reduce_in_fp32 = False
    cfg = ConfigContainer(
        model=model,
        train=TrainingConfig(
            train_iters=args.steps, micro_batch_size=args.micro_batch_size, global_batch_size=args.global_batch_size
        ),
        validation=ValidationConfig(eval_iters=0, eval_interval=args.steps + 1),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=args.learning_rate,
            min_lr=args.learning_rate,
            weight_decay=0.0,
            clip_grad=1.0,
            bf16=True,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=False,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="constant",
            lr_decay_iters=args.steps,
            lr_warmup_iters=0,
            start_weight_decay=0.0,
            end_weight_decay=0.0,
            weight_decay_incr_style="constant",
        ),
        ddp=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            check_for_nan_in_grad=True,
        ),
        dataset=GPTDatasetConfig(
            seq_length=args.sequence_length,
            random_seed=1234,
            data_path=[str(args.data_prefix)],
            split="100,0,0",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            dataloader_type="single",
            num_workers=0,
            persistent_workers=False,
            num_dataset_builder_threads=1,
            path_to_cache=str(
                Path(tempfile.gettempdir())
                / "qwen3_8_flash_next_dataset_cache"
                / f"rank-{os.environ.get('RANK', '0')}"
            ),
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(args.checkpoint), use_tokenizer_vocab_size=False
        ),
        checkpoint=CheckpointConfig(
            save=None, load=None, pretrained_checkpoint=None, save_interval=0, also_save_hf_checkpoint=False
        ),
        logger=LoggerConfig(log_interval=1, tensorboard_dir=None),
        rng=RNGConfig(seed=1234),
        mixed_precision=precision,
    )
    return cfg, evidence


def main() -> None:
    """Load released language weights before DDP and run standard Bridge pretraining."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg, evidence = build_config(args)
    # Core only creates missing indexes on global rank 0 after distributed
    # initialization. Prepare each process's small local cache before that point.
    cfg.dataset.tokenizer = build_tokenizer(cfg.tokenizer)
    cfg.dataset.finalize()
    build_train_valid_test_datasets(cfg, get_dataset_provider(cfg.dataset))
    if args.check_config:
        runtime_config_update(cfg)
        # Construct the actual Core scheduler on a CPU optimizer as well: config
        # validation alone does not catch omitted weight-decay endpoints.
        probe_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=args.learning_rate)
        probe_scheduler = _get_scheduler(cfg.optimizer, cfg.scheduler, probe_optimizer)
        probe_scheduler.step(args.global_batch_size)
        assert probe_optimizer.param_groups[0]["lr"] == args.learning_rate
        assert probe_optimizer.param_groups[0]["weight_decay"] == 0.0
        moe_layers = _get_num_moe_layers(cfg.model)
        if moe_layers != cfg.model.num_layers:
            raise RuntimeError("Qwen4-Exp metric logging must count the MoE inside every hybrid block.")
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.output / "config-check.json",
            {
                "passed": True,
                "scheduler_constructed_on_cpu": True,
                "dataset_cache_prepared": True,
                "moe_layers": moe_layers,
            },
        )
        logger.info("Configuration and CPU scheduler checks passed; no model or process groups were created.")
        return
    pretrain(config=cfg, forward_step_func=forward_step, callbacks=[evidence])
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
