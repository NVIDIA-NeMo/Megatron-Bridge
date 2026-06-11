#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Qwen3.5-VL MegatronMIMO SFT runner on HF VLM conversation data.

This is the MegatronMIMO counterpart to the standard Qwen3.5-VL SFT recipe.
It runs on the same HF CORD-v2-style VLM conversation data, but routes the
batch through the MegatronMIMO heterogeneous-parallelism training path:
language and image encoder modules run on disjoint rank groups, each with its
own TP/PP/DP configuration.

Conversation examples are built with the standard HF VLM provider, then the
resulting Qwen batch is adapted into the MIMO forward shape:

  - language inputs: ``input_ids``, MRoPE ``position_ids``, labels, loss mask
  - image inputs: ``modality_inputs["images"]["qwen_visual"]``

Example 2-GPU smoke:

  FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 \\
  uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \\
    examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py \\
      --hf-model Qwen/Qwen3.5-0.8B \\
      --component language=tp=1,dp=1,rank_offset=0 \\
      --component images=tp=1,dp=1,rank_offset=1 \\
      --train-iters 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.data.hf_datasets.provider import HFConversationDatasetProvider
from megatron.bridge.data.megatron_mimo.dp_utils import get_megatron_mimo_sampling_info
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import reorganize_inputs
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.checkpointing import load_checkpoint
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetBuildContext,
    LoggerConfig,
    OptimizerConfig,
    ProfilingConfig,
    TrainingConfig,
)
from megatron.bridge.training.megatron_mimo_parallel_utils import get_active_module_pg
from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo
from megatron.bridge.training.state import GlobalState, TrainState
from megatron.bridge.training.tokenizers.config import TokenizerConfig


G_COMPONENT_KEY_TO_FIELD = {
    "tp": "tensor_model_parallel_size",
    "pp": "pipeline_model_parallel_size",
    "dp": "data_parallel_size",
    "cp": "context_parallel_size",
    "etp": "expert_tensor_parallel_size",
    "rank_offset": "rank_offset",
}
G_DEFAULT_COMPONENTS = [
    "language=tp=1,dp=1,rank_offset=0",
    "images=tp=1,dp=1,rank_offset=1",
]
G_EXAMPLE_ROOT = "/workspace/qwen35_vl_mimo"

G_RANK_LOG_FILE = None


@dataclass(frozen=True)
class Qwen35MIMOHFSpec:
    """Qwen3.5-VL constants needed by the HF-data MIMO adapter."""

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    pad_token_id: int = 0
    spatial_merge_size: int = 2
    image_modality_name: str = "images"
    image_encoder_key: str = "qwen_visual"

    @property
    def square_merge_size(self) -> int:
        return self.spatial_merge_size**2


def _log(message: str) -> None:
    """Write a rank-prefixed message to stdout and the per-rank log file."""
    rank = dist.get_rank() if dist.is_initialized() else "?"
    line = f"[Rank {rank}] {message}\n"
    if G_RANK_LOG_FILE is not None:
        G_RANK_LOG_FILE.write(line)
        G_RANK_LOG_FILE.flush()
    print(line, end="", flush=True)


def _get_int_attr(config: object | None, name: str, default: int) -> int:
    if config is None:
        return default
    value = getattr(config, name, default)
    return default if value is None else int(value)


def _build_hf_spec(hf_config: object) -> Qwen35MIMOHFSpec:
    text_config = getattr(hf_config, "text_config", hf_config)
    vision_config = getattr(hf_config, "vision_config", None)
    return Qwen35MIMOHFSpec(
        image_token_id=_get_int_attr(hf_config, "image_token_id", 248056),
        video_token_id=_get_int_attr(hf_config, "video_token_id", 248057),
        vision_start_token_id=_get_int_attr(hf_config, "vision_start_token_id", 248053),
        vision_end_token_id=_get_int_attr(hf_config, "vision_end_token_id", 248054),
        pad_token_id=_get_int_attr(text_config, "pad_token_id", 0),
        spatial_merge_size=_get_int_attr(vision_config, "spatial_merge_size", 2),
    )


def _parse_component_spec(raw: str) -> tuple[str, ModuleParallelismConfig]:
    if "=" not in raw:
        raise ValueError(f"Invalid --component {raw!r}; expected name=tp=N[,pp=N,dp=N,rank_offset=N]")

    name, _, payload = raw.partition("=")
    parsed: dict[str, int] = {}
    for item in payload.split(","):
        key, _, raw_value = item.partition("=")
        if key not in G_COMPONENT_KEY_TO_FIELD or not raw_value:
            raise ValueError(f"Invalid component field {item!r} in {raw!r}")
        parsed[G_COMPONENT_KEY_TO_FIELD[key]] = int(raw_value)

    return name, ModuleParallelismConfig(**parsed)


def _parse_profile_ranks(raw: str) -> list[int]:
    value = raw.strip().lower()
    if value in ("", "all"):
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _build_parallelism_config(component_specs: list[str], world_size: int) -> MegatronMIMOParallelismConfig:
    module_parallelisms: dict[str, ModuleParallelismConfig] = {}
    for raw in component_specs:
        name, parallelism = _parse_component_spec(raw)
        if name in module_parallelisms:
            raise ValueError(f"Duplicate --component for {name!r}")
        if parallelism.data_parallel_size is None:
            raise ValueError(f"Component {name!r} must set dp explicitly in {raw!r}.")
        module_parallelisms[name] = parallelism

    if MIMO_LANGUAGE_MODULE_KEY not in module_parallelisms:
        raise ValueError(f"Component layout must include {MIMO_LANGUAGE_MODULE_KEY!r}.")

    used_ranks = max(p.rank_offset + p.total_ranks for p in module_parallelisms.values())
    if used_ranks != world_size:
        raise ValueError(
            f"Component layout uses {used_ranks} ranks, but torch world_size is {world_size}. "
            "Set --component rank_offset/dp/tp/pp to cover every rank exactly once."
        )

    return MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)


def _validate_mimo_batch_sizes(
    parallelism_config: MegatronMIMOParallelismConfig,
    args: argparse.Namespace,
) -> list[str]:
    if args.micro_batch_size <= 0:
        raise ValueError(f"--micro-batch-size must be positive, got {args.micro_batch_size}.")
    if args.global_batch_size <= 0:
        raise ValueError(f"--global-batch-size must be positive, got {args.global_batch_size}.")
    if args.global_batch_size % args.micro_batch_size != 0:
        raise ValueError(
            f"--global-batch-size ({args.global_batch_size}) must be divisible by "
            f"--micro-batch-size ({args.micro_batch_size})."
        )

    summaries = []
    for name, parallelism in parallelism_config.module_parallelisms.items():
        dp = parallelism.data_parallel_size
        if dp is None:
            raise ValueError(f"Component {name!r} must set dp explicitly.")
        if args.micro_batch_size % dp != 0:
            raise ValueError(
                f"--micro-batch-size ({args.micro_batch_size}) must be divisible by component {name!r} dp ({dp})."
            )
        summaries.append(f"{name}: dp={dp}, local_mbs={args.micro_batch_size // dp}")
    return summaries


def _build_mimo_provider(
    hf_config: object,
    parallelism_config: MegatronMIMOParallelismConfig,
    args: argparse.Namespace,
) -> MegatronMIMOProvider:
    bridge = AutoBridge.from_hf_config(hf_config)
    standard_provider = bridge.to_megatron_provider(load_weights=False)
    standard_provider.seq_length = args.seq_length
    if hasattr(standard_provider, "language_max_sequence_length"):
        standard_provider.language_max_sequence_length = args.seq_length
    standard_provider.bf16 = not args.fp32
    standard_provider.fp16 = False
    standard_provider.use_cpu_initialization = True
    if hasattr(standard_provider, "mtp_num_layers"):
        standard_provider.mtp_num_layers = None
    if hasattr(standard_provider, "_pack_sequences_in_batch"):
        standard_provider._pack_sequences_in_batch = False

    provider = MegatronMIMOProvider.from_standard_provider(
        standard_provider=standard_provider,
        megatron_mimo_parallelism_config=parallelism_config,
    )
    provider.use_cpu_initialization = True
    provider.bf16 = not args.fp32
    provider.fp16 = False
    provider.freeze_language_model = args.freeze_llm
    provider.freeze_modality_encoders = {"images": args.freeze_vision}
    provider.freeze_modality_projections = {"images": args.freeze_projector}
    if not hasattr(provider, "num_moe_experts"):
        provider.num_moe_experts = None
    return provider


def _build_data_provider(args: argparse.Namespace) -> HFConversationDatasetProvider:
    maker_name = args.dataset_maker
    if not maker_name.startswith("make_"):
        maker_name = f"make_{maker_name}_dataset"
    provider = HFConversationDatasetProvider(
        seq_length=args.seq_length,
        hf_processor_path=args.processor_path or args.hf_model,
        maker_name=maker_name,
        num_workers=args.num_workers,
        dataloader_type=args.dataloader_type,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        pack_sequences_in_batch=False,
        do_validation=True,
        do_test=False,
        trust_remote_code=args.trust_remote_code,
    )
    provider.drop_last = True
    return provider


def _pad_or_truncate_2d(tensor: torch.Tensor | None, target_len: int, pad_value: int | float) -> torch.Tensor | None:
    if tensor is None:
        return None
    cur_len = tensor.size(1)
    if cur_len == target_len:
        return tensor.contiguous()
    if cur_len > target_len:
        return tensor[:, :target_len].contiguous()
    pad = torch.full(
        (tensor.size(0), target_len - cur_len),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1).contiguous()


def _normalized_visual_kwargs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    visual_inputs = batch.get("visual_inputs")
    if visual_inputs is None:
        return {}
    return visual_inputs.normalized_for_model()


def _adapt_qwen35_hf_batch(
    batch: dict[str, Any],
    spec: Qwen35MIMOHFSpec,
    *,
    seq_length: int,
    pad_to_seq_length: bool,
) -> dict[str, Any]:
    input_ids = batch.get("tokens") if batch.get("tokens") is not None else batch["input_ids"]
    labels = batch.get("labels")
    loss_mask = batch.get("loss_mask")
    attention_mask = batch.get("attention_mask")

    if pad_to_seq_length:
        input_ids = _pad_or_truncate_2d(input_ids, seq_length, spec.pad_token_id)
        labels = _pad_or_truncate_2d(labels, seq_length, -100)
        loss_mask = _pad_or_truncate_2d(loss_mask, seq_length, 0)
        attention_mask = _pad_or_truncate_2d(attention_mask, seq_length, 0)

    if attention_mask is None or attention_mask.dim() != 2:
        rope_attention_mask = (input_ids != spec.pad_token_id).long()
    else:
        rope_attention_mask = attention_mask.long()

    visual_kwargs = _normalized_visual_kwargs(batch)
    pixel_values = visual_kwargs.get("pixel_values")
    image_grid_thw = visual_kwargs.get("image_grid_thw")

    position_ids, _ = get_rope_index(
        spec.spatial_merge_size,
        spec.image_token_id,
        spec.video_token_id,
        spec.vision_start_token_id,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=rope_attention_mask,
    )

    modality_inputs = None
    if pixel_values is not None and image_grid_thw is not None:
        vision_data, vision_grid_thw, _ = reorganize_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_id=spec.image_token_id,
            video_token_id=spec.video_token_id,
            square_merge_size=spec.square_merge_size,
        )
        modality_inputs = {
            spec.image_modality_name: {
                spec.image_encoder_key: {
                    "hidden_states": vision_data,
                    "grid_thw": vision_grid_thw,
                }
            }
        }

    return {
        "input_ids": input_ids.contiguous(),
        "position_ids": position_ids.contiguous(),
        "attention_mask": None,
        "labels": None if labels is None else labels.contiguous(),
        "loss_mask": None if loss_mask is None else loss_mask.contiguous(),
        "modality_inputs": modality_inputs,
    }


def _summarize_batch(batch: dict[str, Any], adapted: dict[str, Any], spec: Qwen35MIMOHFSpec) -> str:
    input_ids = adapted["input_ids"]
    image_token_counts = (input_ids == spec.image_token_id).sum(dim=1)
    image_text = int((image_token_counts > 0).sum().item())
    batch_size = int(input_ids.size(0))
    raw_images = 0
    image_grid_thw = _normalized_visual_kwargs(batch).get("image_grid_thw")
    if image_grid_thw is not None:
        raw_images = int(image_grid_thw.reshape(-1, 3).size(0))
    return (
        f"batch_size={batch_size}, image_text={image_text}, "
        f"llm_image_tokens={int(image_token_counts.sum().item())}, raw_images={raw_images}, "
        f"seq_len={input_ids.size(1)}"
    )


class _Qwen35HFMimoCollateAdapter:
    """Pickleable collate wrapper that runs the MIMO adapt step inside the dataloader workers.

    Runs after the standard HF VLM collate (which produces `input_ids`,
    `visual_inputs`, etc.) and before the batch crosses the worker→main IPC
    boundary. Moves `get_rope_index` and `reorganize_inputs` off the main
    process critical path; the main process now only receives MIMO-shaped
    batches and the `get_batch` host-to-device copy.
    """

    def __init__(
        self,
        base_collate: Callable[[list[Any]], dict[str, Any]],
        spec: Qwen35MIMOHFSpec,
        seq_length: int,
        pad_to_seq_length: bool,
    ) -> None:
        self.base_collate = base_collate
        self.spec = spec
        self.seq_length = seq_length
        self.pad_to_seq_length = pad_to_seq_length

    def __call__(self, items: list[Any]) -> dict[str, Any]:
        batch = self.base_collate(items)
        return _adapt_qwen35_hf_batch(
            batch,
            self.spec,
            seq_length=self.seq_length,
            pad_to_seq_length=self.pad_to_seq_length,
        )


def _summarize_adapted_batch(adapted: dict[str, Any], spec: Qwen35MIMOHFSpec) -> str:
    input_ids = adapted["input_ids"]
    image_token_counts = (input_ids == spec.image_token_id).sum(dim=1)
    image_text = int((image_token_counts > 0).sum().item())
    batch_size = int(input_ids.size(0))
    modality_inputs = adapted.get("modality_inputs") or {}
    image_inputs = modality_inputs.get(spec.image_modality_name) or {}
    encoder_inputs = image_inputs.get(spec.image_encoder_key) or {}
    grid_thw = encoder_inputs.get("grid_thw")
    raw_images = 0 if grid_thw is None else int(grid_thw.reshape(-1, 3).size(0))
    return (
        f"batch_size={batch_size}, image_text={image_text}, "
        f"llm_image_tokens={int(image_token_counts.sum().item())}, raw_images={raw_images}, "
        f"seq_len={input_ids.size(1)}"
    )


def _wrap_iter_logging(
    loader_iter: Iterator[dict[str, Any]],
    spec: Qwen35MIMOHFSpec,
) -> Iterator[dict[str, Any]]:
    for batch_idx, adapted in enumerate(loader_iter):
        _log(f"hf batch {batch_idx}: {_summarize_adapted_batch(adapted, spec)}")
        yield adapted


def _make_build_data_iterators(spec: Qwen35MIMOHFSpec, args: argparse.Namespace):
    def _build_data_iterators(cfg, _megatron_mimo_infra, *, train_state=None):
        if train_state is None:
            train_state = TrainState()

        if cfg.model._grids is None:
            raise ValueError("MegatronMIMOProvider._grids is None. Model must be built before data iterators.")

        sampler_dp_rank, sampler_dp_size, needs_data = get_megatron_mimo_sampling_info(
            cfg.model.megatron_mimo_parallelism_config,
            cfg.model._grids,
        )
        if not needs_data:
            return None, None

        train_samples = max(cfg.train.train_iters * cfg.train.global_batch_size, 10)
        context = DatasetBuildContext(
            train_samples=train_samples,
            valid_samples=0,
            test_samples=0,
            tokenizer=None,
        )
        train_ds, _, _ = cfg.dataset.build_datasets(context)
        if train_ds is None:
            raise ValueError("HF conversation provider did not build a train dataset.")
        base_collate = getattr(train_ds, "collate_fn", None)
        if base_collate is None:
            raise ValueError("HF conversation train dataset does not expose collate_fn.")

        # Wrap the dataset's collate so the MIMO adapt runs in worker processes
        # alongside the HF VLM processor work; the main process used to spend
        # ~1s/iter doing `get_rope_index` here via a generator that ran adapt
        # post-`next(...)`.
        collate_fn = _Qwen35HFMimoCollateAdapter(
            base_collate=base_collate,
            spec=spec,
            seq_length=args.seq_length,
            pad_to_seq_length=args.pad_to_seq_length,
        )

        train_loader = build_pretraining_data_loader(
            dataset=train_ds,
            consumed_samples=train_state.consumed_train_samples,
            dataloader_type=cfg.dataset.dataloader_type,
            micro_batch_size=cfg.train.micro_batch_size,
            num_workers=cfg.dataset.num_workers,
            data_sharding=cfg.dataset.data_sharding,
            collate_fn=collate_fn,
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            data_parallel_rank=sampler_dp_rank,
            data_parallel_size=sampler_dp_size,
            drop_last=cfg.dataset.drop_last,
        )

        # `pretrain_megatron_mimo` calls `next(data_iterator)` per microbatch, so
        # return an iterator (DataLoader is iterable but not itself an iterator).
        if args.log_batches:
            return _wrap_iter_logging(train_loader, spec), None
        return iter(train_loader), None

    return _build_data_iterators


def _build_checkpoint_config(args: argparse.Namespace) -> CheckpointConfig:
    checkpoint_cfg = CheckpointConfig()
    if args.load_checkpoint is not None and args.pretrained_checkpoint is not None:
        raise ValueError(
            "Use either --load-checkpoint for resume or --pretrained-checkpoint for model weights, not both."
        )
    checkpoint_cfg.save = args.checkpoint_dir
    if args.checkpoint_interval is not None:
        checkpoint_cfg.save_interval = args.checkpoint_interval
    if args.load_checkpoint is not None:
        checkpoint_cfg.load = args.load_checkpoint
    if args.pretrained_checkpoint is not None:
        # Converted MegatronMIMO checkpoints are saved from the unwrapped model.
        # The training path wraps submodules in DDP before its normal checkpoint
        # load, which changes expected keys to e.g. language_model.module.*.
        # Load converted weights with a pre-wrap hook instead.
        checkpoint_cfg.load_optim = False
        checkpoint_cfg.load_rng = False
    checkpoint_cfg.ckpt_format = "torch_dist"
    checkpoint_cfg.fully_parallel_save = True
    checkpoint_cfg.dist_ckpt_optim_fully_reshardable = True
    checkpoint_cfg.save_rng = False
    return checkpoint_cfg


def _register_converted_checkpoint_pre_wrap_hook(
    model_provider: MegatronMIMOProvider,
    checkpoint_path: str | None,
) -> None:
    if checkpoint_path is None:
        return

    def _load_converted_checkpoint(model_list):
        if len(model_list) != 1:
            raise ValueError(f"Expected a single MegatronMIMO model, got {len(model_list)} chunks.")

        infra = model_provider.build_infra()
        active_module_name, local_pg_collection = get_active_module_pg(infra)
        load_state = GlobalState()
        load_state.cfg = ConfigContainer(
            model=model_provider,
            train=None,
            optimizer=OptimizerConfig(use_distributed_optimizer=False),
            ddp=None,
            scheduler=None,
            dataset=None,
            logger=LoggerConfig(),
            tokenizer=None,
            checkpoint=CheckpointConfig(
                async_save=False,
                load=checkpoint_path,
                finetune=True,
                load_optim=False,
                load_rng=False,
                ckpt_format="torch_dist",
                fully_parallel_save=False,
            ),
            dist=None,
        )

        _log(
            "loading converted MegatronMIMO checkpoint before DDP wrap: "
            f"{checkpoint_path} (module={active_module_name})"
        )
        load_checkpoint(
            state=load_state,
            model=model_list,
            optimizer=None,
            opt_param_scheduler=None,
            pg_collection=local_pg_collection,
            module_name=active_module_name,
        )
        _log("converted MegatronMIMO checkpoint loaded before DDP wrap")
        return model_list

    model_provider.register_pre_wrap_hook(_load_converted_checkpoint)


def _build_config(
    *,
    model_provider: MegatronMIMOProvider,
    data_provider: HFConversationDatasetProvider,
    args: argparse.Namespace,
) -> ConfigContainer:
    optimizer_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=args.lr_warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        clip_grad=args.clip_grad,
        start_weight_decay=args.start_weight_decay,
        end_weight_decay=args.end_weight_decay,
    )
    optimizer_cfg.bf16 = not args.fp32
    optimizer_cfg.fp16 = False
    optimizer_cfg.use_precision_aware_optimizer = False
    optimizer_cfg.main_grads_dtype = torch.float32
    optimizer_cfg.main_params_dtype = torch.float32
    optimizer_cfg.exp_avg_dtype = torch.float32
    optimizer_cfg.exp_avg_sq_dtype = torch.float32

    logger_cfg = LoggerConfig()
    logger_cfg.log_interval = args.log_interval
    logger_cfg.log_timers_to_tensorboard = True
    logger_cfg.tensorboard_dir = args.tensorboard_dir
    logger_cfg.wandb_project = args.wandb_project
    logger_cfg.wandb_exp_name = args.wandb_exp_name
    logger_cfg.wandb_entity = args.wandb_entity
    logger_cfg.wandb_save_dir = args.wandb_save_dir

    profiling_cfg = ProfilingConfig(
        use_nsys_profiler=args.profile == "nsys",
        use_pytorch_profiler=args.profile == "pytorch",
        profile_step_start=args.profile_step_start,
        profile_step_end=args.profile_step_end,
        profile_ranks=_parse_profile_ranks(args.profile_ranks),
        record_shapes=args.profile_record_shapes,
        pytorch_profiler_collect_shapes=args.profile_record_shapes,
        nvtx_ranges=args.profile_nvtx_ranges,
    )

    cfg = ConfigContainer(
        train=TrainingConfig(
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            train_iters=args.train_iters,
            eval_interval=None,
            eval_iters=None,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        model=model_provider,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        dataset=data_provider,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(),
        checkpoint=_build_checkpoint_config(args),
        profiling=profiling_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
        ),
    )
    cfg.data_parallel_size = 1
    cfg.rng.seed = args.seed
    cfg.mixed_precision = "bf16_mixed" if not args.fp32 else None
    return cfg


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("yes", "true", "t", "1"):
        return True
    if lowered in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _default_model_tag(hf_model: str) -> str:
    return Path(hf_model.rstrip("/")).name


def _resolve_default_paths(args: argparse.Namespace) -> None:
    model_tag = _default_model_tag(args.hf_model)
    if args.pretrained_checkpoint is None and args.load_checkpoint is None and not args.allow_random_init:
        args.pretrained_checkpoint = str(Path(args.experiment_root) / "models" / "mimo" / f"{model_tag}-mimo")
    if args.checkpoint_dir is None:
        run_name = args.run_name or f"{model_tag}_cord_v2_mimo_hf"
        args.checkpoint_dir = str(Path(args.experiment_root) / "results" / "mimo" / run_name)
    if args.log_dir is None:
        args.log_dir = str(Path(args.experiment_root) / "logs" / "mimo_hf")
    if args.tensorboard_dir is None:
        args.tensorboard_dir = str(Path(args.checkpoint_dir) / "tb_logs")
    if args.wandb_save_dir is None:
        args.wandb_save_dir = str(Path(args.checkpoint_dir) / "wandb")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MegatronMIMO Qwen3.5-VL HF CORD-v2 validation training")
    parser.add_argument("--hf-model", type=str, default="Qwen/Qwen3.5-0.8B", help="HF model id or local config path")
    parser.add_argument("--processor-path", type=str, default=None, help="HF processor path; defaults to --hf-model")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--component",
        action="append",
        default=None,
        help="Component layout: name=tp=N[,pp=N,cp=N,dp=N,rank_offset=N]",
    )
    parser.add_argument("--experiment-root", type=str, default=G_EXAMPLE_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset-maker", type=str, default="cord_v2")
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dataloader-type", choices=("single", "cyclic"), default="single")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of bf16")
    parser.add_argument("--freeze-vision", type=_str2bool, default=False)
    parser.add_argument("--freeze-llm", type=_str2bool, default=False)
    parser.add_argument("--freeze-projector", type=_str2bool, default=False)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--min-lr", type=float, default=5.0e-7)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lr-warmup-iters", type=int, default=200)
    parser.add_argument("--lr-decay-iters", type=int, default=300000)
    parser.add_argument("--start-weight-decay", type=float, default=0.033)
    parser.add_argument("--end-weight-decay", type=float, default=0.033)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--log-throughput",
        action="store_true",
        help="Accepted for launcher compatibility; MIMO throughput logging is disabled until heterogeneous FLOPs accounting is wired.",
    )
    parser.add_argument(
        "--log-throughput-to-tensorboard",
        action="store_true",
        help="Accepted for launcher compatibility; MIMO throughput logging is disabled until heterogeneous FLOPs accounting is wired.",
    )
    parser.add_argument("--throughput-window-size", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-exp-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Checkpoint directory for full resume")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Existing MegatronMIMO checkpoint to load as model weights before training",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow training without --pretrained-checkpoint or --load-checkpoint for performance-only smoke runs.",
    )
    parser.add_argument(
        "--pad-to-seq-length",
        type=_str2bool,
        default=True,
        help="Pad/truncate HF conversation batches to --seq-length before MIMO forward.",
    )
    parser.add_argument("--profile", choices=("none", "nsys", "pytorch"), default="none")
    parser.add_argument("--profile-step-start", type=int, default=1)
    parser.add_argument("--profile-step-end", type=int, default=2)
    parser.add_argument(
        "--profile-ranks",
        type=str,
        default="0",
        help="Comma-separated global ranks to profile, or 'all' for every rank.",
    )
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-nvtx-ranges", action="store_true")
    parser.add_argument("--log-batches", action="store_true", help="Log per-batch image/token summary.")
    args = parser.parse_args()
    _resolve_default_paths(args)
    return args


def main() -> None:
    """Entry point for Qwen3.5-VL MegatronMIMO HF-data validation training."""
    global G_RANK_LOG_FILE

    args = _parse_args()
    components = args.component or G_DEFAULT_COMPONENTS
    if args.wandb_project is None:
        os.environ.setdefault("WANDB_MODE", "disabled")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    G_RANK_LOG_FILE = open(Path(args.log_dir) / f"rank_{rank}.log", "w")
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(Path(args.log_dir) / f"rank_{rank}_full.log", mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )

    succeeded = False
    try:
        _log(f"distributed initialized (world_size={dist.get_world_size()})")
        _log(f"loading HF config from {args.hf_model}")
        hf_config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
        hf_spec = _build_hf_spec(hf_config)
        _log(
            f"qwen constants: image_token_id={hf_spec.image_token_id}, "
            f"vision_start_token_id={hf_spec.vision_start_token_id}, "
            f"spatial_merge_size={hf_spec.spatial_merge_size}"
        )

        parallelism_config = _build_parallelism_config(components, dist.get_world_size())
        _log(f"component layout: {components}")
        for summary in _validate_mimo_batch_sizes(parallelism_config, args):
            _log(f"batch contract: global_mbs={args.micro_batch_size}, {summary}")

        _log("building Qwen3.5-VL MegatronMIMO provider")
        model_provider = _build_mimo_provider(hf_config, parallelism_config, args)
        _register_converted_checkpoint_pre_wrap_hook(model_provider, args.pretrained_checkpoint)

        _log(f"building HF conversation data provider: maker={args.dataset_maker}")
        data_provider = _build_data_provider(args)

        _log(f"pretrained checkpoint: {args.pretrained_checkpoint}")
        _log(f"checkpoint dir: {args.checkpoint_dir}")
        _log("building training config")
        cfg = _build_config(model_provider=model_provider, data_provider=data_provider, args=args)

        _log("launching pretrain_megatron_mimo")
        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_make_build_data_iterators(hf_spec, args),
        )
        _log("PASSED")
        succeeded = True
    finally:
        if succeeded:
            dist.destroy_process_group()
        if G_RANK_LOG_FILE is not None:
            G_RANK_LOG_FILE.close()
            G_RANK_LOG_FILE = None


if __name__ == "__main__":
    main()
