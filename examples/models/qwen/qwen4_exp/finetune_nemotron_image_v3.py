#!/usr/bin/env python3
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


"""Full-checkpoint Qwen4-Exp VL recipe for prepared Nemotron-Image-Training-v3.

Early recipe: performance is unoptimized. The measured training loss remains
valid; this is not a performance benchmark or a numerical parity test.
Use the companion preparation script and README. The default configuration is
64 GB200, TP8/SP, EP64, PP1/CP1, matching the recorded Qwen3.8-Flash-Next run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from megatron.bridge.data.builders.energon import (
    EnergonDatasetConfig,
    QwenVLEnergonTaskEncoderConfig,
    build_energon_task_encoder,
)
from megatron.bridge.models import AutoBridge
from megatron.bridge.models.qwen_vl.qwen3_vl_step import get_batch
from megatron.bridge.training.callbacks import Callback, CallbackContext
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
    runtime_config_update,
)
from megatron.bridge.training.losses import create_masked_next_token_loss_function
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.padding_utils import pad_or_truncate_2d_to_len


LOG = logging.getLogger(__name__)
SELECTED_RECORDS: dict = {}


def require(condition: bool, message: str) -> None:
    """Reject invalid data or configuration without relying on Python assertions."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash one explicit file in bounded chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write(path: Path, data: dict) -> None:
    """Atomically persist a JSON measurement or completion record."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def parameter_family(name: str) -> str:
    """Classify a model parameter for sampled update measurements."""
    if "vision_model." in name:
        return "projector" if ".merger." in name else "vision"
    return "language"


def global_max(value: float) -> float:
    """Reduce one diagnostic scalar over all training ranks."""
    t = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def admit_dataset(args: argparse.Namespace) -> dict:
    """Bind whole-sample selection and actual processor assets before any model load."""
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    require(
        manifest["completed"] is True and manifest["source_integrity_verified"] is True,
        "Dataset preparation incomplete",
    )
    require(
        manifest["limits"]
        == {
            "max_seq_length": args.sequence_length,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "alignment": 8,
        },
        "Dataset/driver preprocessing mismatch",
    )
    require(
        manifest["counts"]["train"] > 0 and manifest["counts"]["val"] == 0, "Expected nonempty train-only selection"
    )
    processor = args.processor or args.checkpoint
    for name, record in manifest["processor_assets"].items():
        path = processor / name
        require(
            path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
            "Processor asset mismatch: " + name,
        )
    selection = args.dataset / "selection.jsonl"
    require(sha(selection) == manifest["files"]["selection.jsonl"]["sha256"], "Selection metadata differs")
    allowed = {}
    accepted = 0
    for line in selection.read_text().splitlines():
        row = json.loads(line)
        if row["decision"] != "accepted":
            continue
        accepted += 1
        records = row["untruncated_collator_tensors"]
        key = records["input_ids"]["sha256"]
        require(key not in allowed or allowed[key] == records, "Ambiguous selected token/mask record")
        allowed[key] = records
    require(accepted == manifest["counts"]["train"], "Selected row census differs")
    if args.check_config:
        for name, record in manifest["files"].items():
            path = args.dataset / name
            require(
                path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                "Prepared dataset bytes differ: " + name,
            )
    return allowed


def inspect_batch(batch: dict, args: argparse.Namespace, model_cfg: Any) -> dict:
    """Reject empty-image, overlength, malformed or wholly masked examples."""
    tokens = batch.get("tokens")
    if tokens is None:
        tokens = batch.get("input_ids")
    labels, mask = batch["labels"], batch["loss_mask"]
    require(tokens.ndim == 2 and tokens.shape[0] == 1, "Only unpacked microbatch1 supported")
    require(0 < tokens.shape[1] <= args.sequence_length, "Overlength example: no runtime truncation")
    require(labels.shape == tokens.shape == mask.shape, "Token/label/mask shape mismatch")
    require(bool(torch.isfinite(mask).all()) and bool(((mask == 0) | (mask == 1)).all()), "Binary finite loss mask")
    require(int(mask.sum()) > 0, "No supervised assistant tokens")
    require(bool(((tokens >= 0) & (tokens < model_cfg.vocab_size)).all()), "Token outside vocabulary")
    require(
        bool(((labels[mask.bool()] >= 0) & (labels[mask.bool()] < model_cfg.vocab_size)).all()),
        "Supervised label outside vocabulary",
    )
    visual = batch.get("visual_inputs")
    require(visual is not None, "Image-bearing examples required; no text-only fallback")
    media = visual.normalized_for_model()
    pixel, grid = media.get("pixel_values"), media.get("image_grid_thw")
    require(
        pixel is not None and pixel.numel() > 0 and grid is not None and grid.numel() > 0, "Missing image pixels/grid"
    )
    require(media.get("pixel_values_videos") is None, "This run is image-only")
    require(
        grid.ndim == 2 and grid.shape[1] == 3 and bool((grid > 0).all()) and bool((grid[:, 0] == 1).all()),
        "Invalid still-image grid",
    )
    merge = model_cfg.vision_config["spatial_merge_size"]
    require(bool((grid[:, 1:] % merge == 0).all()), "Image grid merge divisibility")
    patches = int(grid.prod(dim=-1).sum())
    image_tokens = int((tokens == model_cfg.image_token_id).sum())
    require(image_tokens == patches // (merge * merge) and image_tokens > 0, "Visual placeholders/grid mismatch")
    require(pixel.shape[0] == patches and bool(torch.isfinite(pixel).all()), "Patch count/nonfinite pixels")
    require(
        not bool((mask.bool() & (labels == model_cfg.image_token_id)).any()), "Image placeholder labels must be masked"
    )
    # Native config finalization requires fixed sequence extent for EP>1.
    # Preparation authenticates the complete sample padded only to a multiple of8.
    attention = batch.get("attention_mask")
    require(
        attention is not None and attention.shape == tokens.shape and tokens.shape[1] == args.sequence_length,
        "Expected native fixed-extent attention mask",
    )
    require(bool(((attention == 0) | (attention == 1)).all()), "Binary attention mask required")
    raw_length = int(attention.sum())
    require(0 < raw_length <= args.sequence_length, "Invalid unpadded sample length")
    require(
        bool((attention[:, :raw_length] == 1).all()) and bool((attention[:, raw_length:] == 0).all()),
        "Attention mask must be right-contiguous",
    )
    selected_extent = ((raw_length + 7) // 8) * 8
    require(raw_length <= selected_extent <= args.sequence_length, "Selected prefix exceeds native extent")
    require(
        bool((tokens[:, selected_extent:] == 0).all())
        and bool((labels[:, selected_extent:] == -100).all())
        and bool((mask[:, selected_extent:] == 0).all()),
        "Noncanonical extra EP padding",
    )
    actual, full = {}, {}
    for name, tensor in (("input_ids", tokens), ("labels", labels), ("loss_mask", mask)):
        prefix = tensor[:, :selected_extent].contiguous()
        actual[name] = {
            "sha256": hashlib.sha256(prefix.numpy().tobytes()).hexdigest(),
            "shape": list(prefix.shape),
            "dtype": str(prefix.dtype),
        }
        full[name] = {
            "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    require(
        actual["input_ids"]["sha256"] in SELECTED_RECORDS
        and SELECTED_RECORDS[actual["input_ids"]["sha256"]] == actual,
        "Native collator prefix differs from complete selected input/labels/mask",
    )
    return {
        "sequence_tokens": tokens.shape[1],
        "raw_sample_tokens": raw_length,
        "selected_extent": selected_extent,
        "supervised_tokens": int(mask.sum()),
        "images": grid.shape[0],
        "image_tokens": image_tokens,
        "patches": patches,
        "input_ids_sha256": full["input_ids"]["sha256"],
        "selected_prefix_records": actual,
        "native_padded_records": full,
    }


class Evidence(Callback):
    """Record loss and sampled updates, and stop before the requested deadline.

    These per-step checks add overhead; performance is unoptimized. Sampled
    changes do not establish full-gradient or numerical parity.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.loaded = False
        self.rows = 0
        self.batches: list[dict] = []
        self.params = []
        self.before = []
        self.started = 0.0
        self.family_inventory: dict = {}

    def on_train_start(self, context: CallbackContext) -> None:
        require(self.loaded and context.optimizer is not None, "Full HF load must precede optimizer")
        self.args.output.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank()
        (self.args.output / f"rank-{rank:02d}-steps.jsonl").open("x").close()
        # Pair actual native owned masters with their original model parameter.
        # Both BF16->FP32 and model-FP32 branches use native aligned groups.
        names = {id(p): name for module in context.model for name, p in module.named_parameters()}
        leaves = list(getattr(context.optimizer, "chained_optimizers", [context.optimizer]))
        candidates = {family: [] for family in ("language", "vision", "projector")}
        for leaf in leaves:
            for originals, masters in (
                ("model_float16_groups", "shard_fp32_from_float16_groups"),
                ("model_fp32_groups", "shard_fp32_groups"),
            ):
                require(
                    hasattr(leaf, originals) and hasattr(leaf, masters), "Native distributed optimizer groups required"
                )
                for models, shards in zip(getattr(leaf, originals), getattr(leaf, masters), strict=True):
                    for model_param, master in zip(models, shards, strict=True):
                        require(
                            id(model_param) in names and master.dtype == torch.float32, "Unbound native owned master"
                        )
                        name = names[id(model_param)]
                        if master.numel() and "indexer" not in name:
                            candidates[parameter_family(name)].append((name, master))
        self.family_samples = {}
        for family, values in candidates.items():
            self.family_samples[family] = sorted(values, key=lambda item: (item[1].numel(), item[0]))[:32]
            require(global_max(float(bool(values))) == 1, "No owned master witnesses for " + family)
        self.named_masters = [
            (family, name, p) for family, values in self.family_samples.items() for name, p in values
        ]
        self.params = [p.detach().reshape(-1)[:128] for _, _, p in self.named_masters]
        expected = "torch.float32" if self.args.grad_buffer_dtype == "fp32" else "torch.bfloat16"
        dtypes = {
            str(p.main_grad.dtype)
            for m in context.model
            for p in m.parameters()
            if getattr(p, "main_grad", None) is not None
        }
        require(global_max(float(dtypes != {expected})) == 0, "Unexpected gradient buffer dtype")
        self.started = time.monotonic()
        remaining = self.args.stop_at_epoch - time.time() - self.args.exit_reserve_seconds
        require(remaining > 60, "Insufficient remaining allocation time")
        # Native duration exit is checked collectively after each completed step.
        context.state.cfg.train.exit_duration_in_mins = (
            self.args.stop_at_epoch - self.args.exit_reserve_seconds - context.state.start_time
        ) / 60
        write(
            self.args.output / f"rank-{rank:02d}-provenance.json",
            {
                "args": vars(self.args),
                "world_size": dist.get_world_size(),
                "runner_sha256": sha(Path(__file__)),
                "checkpoint_config_sha256": sha(self.args.checkpoint / "config.json"),
                "checkpoint_index_sha256": sha(self.args.checkpoint / "model.safetensors.index.json"),
                "dataset_manifest_sha256": sha(self.args.dataset / "manifest.json"),
                "container_image": os.environ.get("VL_CONTAINER_IMAGE"),
                "bridge_source_revision": os.environ.get("VL_BRIDGE_REVISION"),
                "family_witnesses": {
                    family: [
                        {"name": name, "owned_elements": p.numel(), "sample_elements": min(128, p.numel())}
                        for name, p in values
                    ]
                    for family, values in self.family_samples.items()
                },
                "full_vl_weights_loaded": self.loaded,
                "all_model_parameters_trainable": True,
                "local_module_inventory": self.family_inventory,
                "gradient_buffer_dtypes": sorted(dtypes),
                "calculate_per_token_loss": context.state.cfg.model.calculate_per_token_loss,
                "recompute": None,
                "checkpoint_saving": False,
                "gpu": torch.cuda.get_device_name(),
                "limitations": [
                    "No HF parity or convergence claim",
                    "Master update evidence is sampled",
                    "Hard QSA indexer has no LM gradient",
                ],
            },
        )

    def on_train_step_start(self, context: CallbackContext) -> None:
        self.before = [p.clone() for p in self.params]
        self.batches = []
        self.step_started = time.monotonic()

    def on_train_step_end(self, context: CallbackContext) -> None:
        losses = {k: float(v.item()) for k, v in (context.loss_dict or {}).items()}
        norm = float(context.grad_norm) if context.grad_norm is not None else float("nan")
        family_rows = {}
        local_bad = False
        for family in self.family_samples:
            deltas, grads = [], []
            for (kind, _, master), sample, before in zip(self.named_masters, self.params, self.before, strict=True):
                if kind != family:
                    continue
                deltas.append((sample - before).abs().max().item())
                if master.grad is None:
                    local_bad = True
                    continue
                grad = master.grad.detach().reshape(-1)[:128]
                local_bad |= not bool(torch.isfinite(grad).all())
                grads.append(grad.abs().max().item())
            local_delta = max(deltas or [0.0])
            local_grad = max(grads or [0.0])
            local_bad |= not math.isfinite(local_delta) or not math.isfinite(local_grad)
            family_rows[family] = {
                "sampled_owned_master_delta_max": global_max(local_delta if math.isfinite(local_delta) else 0),
                "sampled_postclip_owned_grad_max": global_max(local_grad if math.isfinite(local_grad) else 0),
            }
        delta = max(v["sampled_owned_master_delta_max"] for v in family_rows.values())
        local_bad |= any(
            v["sampled_owned_master_delta_max"] <= 0 or v["sampled_postclip_owned_grad_max"] <= 0
            for v in family_rows.values()
        )
        bad = (
            local_bad
            or not losses
            or any(not math.isfinite(v) for v in losses.values())
            or not math.isfinite(norm)
            or norm <= 0
            or context.skipped_iter is None
            or bool(context.skipped_iter)
            or not math.isfinite(delta)
            or not self.batches
        )
        failed = bool(global_max(float(bad)))
        update = global_max(delta if math.isfinite(delta) else 0)
        failed = failed or update <= 0
        self.rows += 1
        row = {
            "step": self.rows,
            "losses": {k: v if math.isfinite(v) else None for k, v in losses.items()},
            "grad_norm": norm if math.isfinite(norm) else None,
            "skipped": context.skipped_iter,
            "sampled_master_update_global_max": update,
            "families": family_rows,
            "batches": self.batches,
            "elapsed_seconds": time.monotonic() - self.step_started,
            "passed": not failed,
        }
        with (self.args.output / f"rank-{dist.get_rank():02d}-steps.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        require(not failed, "Nonfinite/skipped/missing-image or absent sampled update")
        # Reserve at least two measured steps beyond the explicit exit reserve.
        estimated = 2 * global_max(row["elapsed_seconds"])
        if time.time() + estimated >= self.args.stop_at_epoch - self.args.exit_reserve_seconds:
            context.state.cfg.train.exit_duration_in_mins = 1e-9

    def on_train_end(self, context: CallbackContext) -> None:
        require(self.rows > 0, "No completed optimizer updates")
        write(
            self.args.output / f"rank-{dist.get_rank():02d}-summary.json",
            {
                "execution_passed": True,
                "steps": self.rows,
                "elapsed_seconds": time.monotonic() - self.started,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "full_vl_training": True,
                "numerical_parity_passed": False,
                "checkpoint_saved": False,
            },
        )


def build_config(args: argparse.Namespace) -> tuple[ConfigContainer, Evidence]:
    """Build the early recipe with unoptimized performance and full HF weight loading."""
    LOG.warning("Early recipe: PERFORMANCE IS UNOPTIMIZED. Loss measurements remain valid.")
    bridge = AutoBridge.from_hf_pretrained(args.checkpoint)
    model = bridge.get_model_config()
    require(
        hasattr(model, "vision_config") and not model.hf_model_text_only,
        "Full conditional-generation checkpoint required",
    )
    model.perform_initialization = False
    model.tensor_model_parallel_size = 8
    model.expert_model_parallel_size = args.expert_parallel_size
    model.expert_tensor_parallel_size = 1
    model.pipeline_model_parallel_size = model.context_parallel_size = 1
    model.sequence_parallel = True
    model.seq_length = args.sequence_length
    model.gradient_accumulation_fusion = False
    model.moe_token_dispatcher_type = "allgather"
    model.moe_permute_fusion = model.moe_shared_expert_overlap = False
    model.moe_grouped_gemm = True
    model.moe_aux_loss_coeff = 0.0
    model.hidden_dropout = model.attention_dropout = 0.0
    model.calculate_per_token_loss = True
    model.recompute_granularity = None
    model.cuda_graph_impl = "none"
    evidence = Evidence(args)

    def load_weights(models: list[torch.nn.Module]) -> list[torch.nn.Module]:
        bridge.load_hf_weights(models)
        inventory = {
            "language": {"tensors": 0, "elements": 0},
            "vision": {"tensors": 0, "elements": 0},
            "projector": {"tensors": 0, "elements": 0},
        }
        for module in models:
            for name, p in module.named_parameters():
                require(p.requires_grad, f"Unexpected frozen parameter: {name}")
                family = parameter_family(name)
                inventory[family]["tensors"] += 1
                inventory[family]["elements"] += p.numel()
        require(all(v["elements"] > 0 for v in inventory.values()), "Missing full VL module family")
        evidence.family_inventory = inventory
        evidence.loaded = True
        return models

    model.pre_wrap_hooks.append(load_weights)
    vision = model.vision_config
    encoder = QwenVLEnergonTaskEncoderConfig(
        hf_processor_path=str(args.processor or args.checkpoint),
        patch_size=vision["patch_size"],
        temporal_patch_size=vision["temporal_patch_size"],
        spatial_merge_size=vision["spatial_merge_size"],
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_num_images=1,
        max_num_frames=1,
        max_visual_tokens=args.sequence_length - 1,
    )
    dataset = EnergonDatasetConfig(
        path=str(args.dataset),
        seq_length=args.sequence_length,
        micro_batch_size=1,
        task_encoder=encoder,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        do_validation=False,
        enable_in_batch_packing=False,
        packing_buffer_size=None,
        pad_to_max_length=False,
        pad_to_multiple_of=8,
        shuffle_buffer_size=100,
    )
    precision = bf16_mixed()
    precision.grad_reduce_in_fp32 = args.grad_buffer_dtype == "fp32"
    cfg = ConfigContainer(
        model=model,
        train=TrainingConfig(train_iters=args.steps, micro_batch_size=1, global_batch_size=args.global_batch_size),
        validation=ValidationConfig(eval_iters=0, eval_interval=args.steps + 1),
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
            lr_decay_iters=args.steps,
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


def cpu_check(cfg: ConfigContainer, evidence: Evidence, args: argparse.Namespace) -> None:
    """Use real installed processor and real dataset rows without CUDA or PG."""
    runtime_config_update(cfg)
    from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule

    encoder = build_energon_task_encoder(cfg.dataset)
    dm = EnergonMultiModalDataModule(
        path=str(args.dataset),
        tokenizer=None,
        seq_length=args.sequence_length,
        task_encoder=encoder,
        micro_batch_size=1,
        num_workers=0,
        num_val_workers=0,
        shuffle_buffer_size=1,
    )
    iterator = iter(dm.train_dataloader())
    rows = [inspect_batch(next(iterator), args, cfg.model) for _ in range(args.check_batches)]
    from megatron.bridge.training.optim import _get_scheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=1e-6)
    scheduler = _get_scheduler(cfg.optimizer, cfg.scheduler, optimizer)
    scheduler.step(args.global_batch_size)
    require(
        optimizer.param_groups[0]["lr"] == 1e-6 and optimizer.param_groups[0]["weight_decay"] == 0.0,
        "Scheduler differs",
    )
    require(not dist.is_initialized() and not torch.cuda.is_initialized(), "CPU check initialized CUDA/PG")
    args.output.mkdir(parents=True, exist_ok=True)
    write(
        args.output / "config-check.json",
        {
            "passed": True,
            "rows": rows,
            "vision_config": cfg.model.vision_config,
            "dataset_manifest_sha256": sha(args.dataset / "manifest.json"),
            "selected_input_identities": len(SELECTED_RECORDS),
            "dataset": str(args.dataset),
            "processor": str(args.processor or args.checkpoint),
            "grad_buffer_dtype": args.grad_buffer_dtype,
            "calculate_per_token_loss": True,
            "full_vl": True,
            "model_constructed": False,
            "runner_sha256": sha(Path(__file__)),
        },
    )


def main() -> None:
    """Train on the prepared ECD subset or check data/configuration on CPU."""
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--processor", type=Path)
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--check-batches", type=int, default=4)
    p.add_argument("--world-size", type=int, default=64, help="This recipe is scoped to 64 ranks.")
    p.add_argument("--expert-parallel-size", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--min-pixels", type=int, default=65536)
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--grad-buffer-dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--stop-at-epoch", type=float, help="Absolute UNIX deadline; overrides --max-runtime-minutes.")
    p.add_argument(
        "--max-runtime-minutes", type=float, default=240.0, help="Budget from driver startup, including model load."
    )
    p.add_argument("--exit-reserve-seconds", type=int, default=180)
    args = p.parse_args()
    if args.stop_at_epoch is None:
        require(
            math.isfinite(args.max_runtime_minutes) and args.max_runtime_minutes > 0,
            "Positive finite runtime required",
        )
        args.stop_at_epoch = time.time() + 60 * args.max_runtime_minutes
    require(not __import__("sys").flags.optimize, "Run Python without -O so framework checks remain enabled")
    require(
        args.world_size == 64 and args.expert_parallel_size == 64, "This early recipe requires world size64 and EP64"
    )
    require(
        args.global_batch_size > 0 and args.global_batch_size % (args.world_size // 8) == 0, "GBS must divide dense DP"
    )
    require(
        args.sequence_length == 2048 and args.global_batch_size == 8,
        "This early recipe requires sequence2048 and global batch8",
    )
    require(
        0 < args.min_pixels <= args.max_pixels and args.steps > 0 and args.check_batches > 0,
        "Positive bounded settings",
    )
    require(args.exit_reserve_seconds >= 60 and math.isfinite(args.stop_at_epoch), "Explicit finite deadline/reserve")
    require(
        (args.checkpoint / "model.safetensors.index.json").is_file() and (args.dataset / "manifest.json").is_file(),
        "Local checkpoint/prepared dataset manifest required",
    )
    if not args.check_config:
        require(int(os.environ["WORLD_SIZE"]) == args.world_size, "Actual world differs")
    global SELECTED_RECORDS
    SELECTED_RECORDS = admit_dataset(args)
    cfg, evidence = build_config(args)
    if args.check_config:
        cpu_check(cfg, evidence, args)
        return

    def forward_step(state: Any, iterator: Any, model: torch.nn.Module, return_schedule_plan: bool = False) -> Any:
        require(not return_schedule_plan, "EP overlap schedule unsupported")
        batch = next(iterator)
        evidence.batches.append(inspect_batch(batch, args, state.cfg.model))
        tokens, labels, mask, _, _, media = get_batch(
            iter([batch]), state.cfg, is_first_pp_stage=True, is_last_pp_stage=True
        )
        # Preserve actual image placeholders; pad only to fixed EP-compatible extent.
        n = args.sequence_length
        tokens = pad_or_truncate_2d_to_len(tokens, target_len=n, max_cap=n, pad_value=0)
        labels = pad_or_truncate_2d_to_len(labels, target_len=n, max_cap=n, pad_value=-100)
        mask = pad_or_truncate_2d_to_len(mask, target_len=n, max_cap=n, pad_value=0)
        output = model(
            input_ids=tokens,
            labels=labels,
            loss_mask=mask,
            position_ids=None,
            attention_mask=None,
            pixel_values=media["pixel_values"],
            image_grid_thw=media["image_grid_thw"],
            packed_seq_params=None,
        )
        return output, create_masked_next_token_loss_function(
            mask,
            state.cfg.rerun_state_machine.check_for_nan_in_loss,
            state.cfg.rerun_state_machine.check_for_spiky_loss,
        )

    pretrain(config=cfg, forward_step_func=forward_step, callbacks=[evidence])


if __name__ == "__main__":
    main()
