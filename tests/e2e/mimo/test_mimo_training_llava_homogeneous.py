# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""End-to-end homogeneous MIMO LLaVA training test.

Exercises the standard pretrain() loop with MimoModelProvider in homogeneous
mode (mimo_parallelism_config=None). All modules (LLM + vision encoder) run
on every rank together. The LLM uses TP=4, PP=1 across all GPUs.

Run:
    torchrun --nproc_per_node=8 tests/e2e/mimo/test_mimo_training_llava_homo.py
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from test_mimo_training_llava import (
    _ENCODER_SEQ_LEN,
    IMAGE_SPECIAL_TOKEN_ID,
    MAX_SEQ_LENGTH,
    VOCAB_SIZE,
    _build_model_specs,
)


# ---------------------------------------------------------------------------
# LLaVA model configs (Vicuna-7B + CLIP ViT-L/14 + MLP projection)
# ---------------------------------------------------------------------------

IMAGE_SPECIAL_TOKEN_ID = 32000
VOCAB_SIZE = 32256
CLIP_OUTPUT_DIM = 1024  # CLIP ViT-L/14 hidden size
MAX_SEQ_LENGTH = 4096
_IMG_SIZE = 336
_PATCH_DIM = 14
# CLIP ViT-L/14 @ 336×336: (336/14)^2 = 576 patches + 1 class token = 577
_ENCODER_SEQ_LEN = 577


def _make_vision_config() -> TransformerConfig:
    """CLIP ViT-L/14 vision encoder config."""
    cfg = TransformerConfig(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    return cfg


def _make_language_config() -> TransformerConfig:
    """Vicuna-7B language model config (same arch as Llama-7B)."""
    cfg = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        use_cpu_initialization=True,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
    )

    cfg.ffn_hidden_size = 11008
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = 1e-5

    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 10000
    cfg.rotary_percent = 1.0

    cfg.seq_length = MAX_SEQ_LENGTH
    cfg.max_position_embeddings = MAX_SEQ_LENGTH

    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    cfg.num_query_groups = 32
    cfg.add_bias_linear = False
    cfg.untie_embeddings_and_output_weights = False

    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    cfg.pipeline_dtype = torch.bfloat16
    cfg.bf16 = True
    cfg.cross_entropy_loss_fusion = True
    cfg.variable_seq_lengths = True

    return cfg


def _make_projection_config(hidden_size: int = 4096) -> TransformerConfig:
    """Vision→language projection MLP config."""
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = 4096
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu
    return cfg


def _build_model_specs():
    """Return (language_model_spec, modality_submodules_spec, special_token_ids)."""
    vision_config = _make_vision_config()
    language_config = _make_language_config()
    projection_config = _make_projection_config(hidden_size=language_config.hidden_size)

    # CLIP ViT-L/14 encoder
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )

    # Vision→language projection MLP
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
            "projector_type": "mlp",
            "input_size": CLIP_OUTPUT_DIM,
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": VOCAB_SIZE,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "position_embedding_type": "rope",
        },
    )

    modality_submodules_spec = {"images": vision_submodule_spec}
    special_token_ids = {"images": IMAGE_SPECIAL_TOKEN_ID}
    return language_model_spec, modality_submodules_spec, special_token_ids


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

from megatron.bridge.data.mimo.hf_provider import HFMimoDatasetProvider
from megatron.bridge.training.config import DatasetBuildContext


def _llava_preprocess(example, dataset_root):
    """Convert LLaVA conversations format to plain text and resolve image paths."""
    conversations = example.get("conversations", [])
    text_parts = [turn.get("value", "") for turn in conversations]
    example["text"] = " ".join(text_parts).replace("<image>", "").strip()
    # Resolve relative image paths to absolute paths
    if "image" in example and example["image"] and not os.path.isabs(example["image"]):
        example["image"] = os.path.join(dataset_root, example["image"])
    return example


def _pad_1d(tensors, pad_value=0):
    """Pad a list of 1-D tensors to the longest length and stack."""
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        if t.size(0) < max_len:
            padding = torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype)
            padded.append(torch.cat([t, padding]))
        else:
            padded.append(t)
    return torch.stack(padded)


def _mimo_collate_with_loss_masking(batch, modality_names, pad_token_id=0):
    """Collate with padding, label shifting, and loss masking.

    Handles:
    - Variable-length sequence padding to the longest in the batch.
    - Label shifting: labels[i] = input_ids[i+1], labels[-1] = -100.
    - Loss mask: 0 for padding, placeholder tokens (IMAGE_SPECIAL_TOKEN_ID),
      and the last position (no valid next-token target).
    """
    import warnings

    if not batch:
        return {}

    lengths = [item["input_ids"].size(0) for item in batch]
    variable_lengths = len(set(lengths)) > 1

    # Pad and stack standard fields
    if variable_lengths:
        input_ids = _pad_1d([item["input_ids"] for item in batch], pad_value=pad_token_id)
        attention_mask = _pad_1d([item["attention_mask"] for item in batch], pad_value=0)
        position_ids = _pad_1d([item["position_ids"] for item in batch], pad_value=0)
    else:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        position_ids = torch.stack([item["position_ids"] for item in batch])

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100

    # Loss mask: start from dataset-provided or all-ones, then enforce masking
    if "loss_mask" in batch[0]:
        if variable_lengths:
            loss_mask = _pad_1d([item["loss_mask"] for item in batch], pad_value=0)
        else:
            loss_mask = torch.stack([item["loss_mask"] for item in batch])
    else:
        loss_mask = torch.ones_like(labels, dtype=torch.float)

    # Zero out loss on padding positions and ignore-index positions
    loss_mask = loss_mask * (labels != -100).float()
    loss_mask = loss_mask * (input_ids != pad_token_id).float()

    # Zero out loss on placeholder token targets
    loss_mask = loss_mask * (labels != IMAGE_SPECIAL_TOKEN_ID).float()

    # Sync labels with loss_mask so CrossEntropyLoss ignores masked positions
    labels = labels.masked_fill(loss_mask == 0, -100)

    # Collate modality inputs
    modality_inputs = {}
    for modality_name in modality_names:
        modality_batch_items = [item.get("modality_inputs", {}).get(modality_name, {}) for item in batch]
        if not any(modality_batch_items):
            continue
        first_non_empty = next((item for item in modality_batch_items if item), {})
        if not first_non_empty:
            continue
        modality_inputs[modality_name] = {}
        for key in first_non_empty.keys():
            values = [item[key] for item in modality_batch_items if key in item]
            if values and isinstance(values[0], torch.Tensor):
                try:
                    modality_inputs[modality_name][key] = torch.stack(values)
                except RuntimeError:
                    warnings.warn(
                        f"Cannot stack tensors for '{modality_name}.{key}' - shapes differ. Keeping as list.",
                        stacklevel=2,
                    )
                    modality_inputs[modality_name][key] = values
            elif values:
                modality_inputs[modality_name][key] = values

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "modality_inputs": modality_inputs,
    }


class HomogeneousHFMimoDatasetProvider(HFMimoDatasetProvider):
    """HFMimoDatasetProvider that sets a custom collate_fn on returned datasets.

    Uses _mimo_collate_with_loss_masking which handles padding, label shifting,
    and loss masking for placeholder / padding tokens.
    """

    def build_datasets(self, context: DatasetBuildContext):
        train_ds, valid_ds, test_ds = super().build_datasets(context)
        collate_fn = partial(_mimo_collate_with_loss_masking, modality_names=["images"])
        for ds in (train_ds, valid_ds, test_ds):
            if ds is not None:
                ds.collate_fn = collate_fn
        return train_ds, valid_ds, test_ds


def _build_hf_data_provider(dataset_root: str) -> HomogeneousHFMimoDatasetProvider:
    """Build an HFMimoDatasetProvider for liuhaotian/LLaVA-Pretrain."""
    provider = HomogeneousHFMimoDatasetProvider(
        seq_length=MAX_SEQ_LENGTH,
        hf_dataset_path=dataset_root,
        hf_data_files="blip_laion_cc_sbu_558k.json",
        hf_tokenizer_path="llava-hf/llava-1.5-7b-hf",
        processor_paths={"images": "openai/clip-vit-large-patch14-336"},
        special_token_ids={"images": IMAGE_SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"images": _ENCODER_SEQ_LEN},
        modality_columns={"images": "image"},
        text_column="text",
        train_split="train",
        preprocess_fn=lambda example: _llava_preprocess(example, dataset_root),
    )
    provider.drop_last = True
    provider.dataloader_type = "cyclic"

    return provider


# ---------------------------------------------------------------------------
# Forward step function (homogeneous path)
# ---------------------------------------------------------------------------

from megatron.bridge.training.mimo_step import loss_func as mimo_loss_func


def forward_step_func(data_iterator, model):
    """Forward step for homogeneous MIMO via pretrain()."""
    batch = next(data_iterator)

    input_ids = batch["input_ids"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    position_ids = batch["position_ids"].cuda(non_blocking=True)

    # Use the loss_mask produced by the collate (accounts for padding, special tokens, label shifting)
    batch_loss_mask = batch.get("loss_mask")
    if batch_loss_mask is not None:
        batch_loss_mask = batch_loss_mask.cuda(non_blocking=True)

    modality_inputs = {}
    if "modality_inputs" in batch:
        for mod_name, mod_tensors in batch["modality_inputs"].items():
            modality_inputs[mod_name] = {
                "clip": {"x": mod_tensors["pixel_values"].cuda(non_blocking=True).to(torch.bfloat16)}
            }

    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        attention_mask=None,
        modality_inputs=modality_inputs,
    )

    output_tensor, model_loss_mask = output

    # Prefer the batch loss_mask (from collate) over the model's
    loss_mask = batch_loss_mask if batch_loss_mask is not None else model_loss_mask
    if loss_mask is None:
        loss_mask = torch.ones_like(output_tensor)

    return output_tensor, partial(mimo_loss_func, loss_mask)


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig


def _build_config(
    mimo_provider: MimoModelProvider,
    data_provider: HomogeneousHFMimoDatasetProvider,
    opt_config: OptimizerConfig,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
    train_iters: int = 2,
    log_interval: int = 1,
    wandb_project: str | None = None,
    wandb_exp_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_save_dir: str | None = None,
    lr_warmup_iters: int = 0,
) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=train_iters,
    )
    train_cfg.log_interval = log_interval

    logger_cfg = LoggerConfig()
    logger_cfg.log_timers_to_tensorboard = True
    logger_cfg.log_interval = log_interval
    logger_cfg.wandb_project = wandb_project
    logger_cfg.wandb_exp_name = wandb_exp_name
    logger_cfg.wandb_entity = wandb_entity
    logger_cfg.wandb_save_dir = wandb_save_dir
    logger_cfg.tensorboard_dir = os.path.join(wandb_save_dir or "/tmp/tb_logs", "tb_logs") if wandb_project else None

    scheduler_cfg = SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=opt_config.min_lr,
        start_weight_decay=opt_config.weight_decay,
        end_weight_decay=opt_config.weight_decay,
    )

    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        check_for_nan_in_grad=False,
    )

    cfg = ConfigContainer(
        train=train_cfg,
        model=mimo_provider,
        optimizer=opt_config,
        scheduler=scheduler_cfg,
        dataset=data_provider,
        ddp=ddp_cfg,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="llava-hf/llava-1.5-7b-hf",
        ),
        checkpoint=CheckpointConfig(),
        validation=ValidationConfig(eval_interval=2, eval_iters=0),
    )
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from megatron.bridge.training.pretrain import pretrain


_rank_log_file = None


def _log(msg):
    """Write with rank prefix to per-rank log file and flush."""
    global _rank_log_file
    rank = dist.get_rank() if dist.is_initialized() else "?"
    line = f"[Rank {rank}] {msg}\n"
    if _rank_log_file:
        _rank_log_file.write(line)
        _rank_log_file.flush()
    print(line, end="", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Homogeneous MIMO LLaVA training")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--global-batch-size", type=int, default=1, help="Global batch size across all GPUs")
    parser.add_argument("--train-iters", type=int, default=2, help="Number of training iterations")
    parser.add_argument("--min-lr", type=float, default=2.0e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Checkpoint save interval (iterations)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--wandb-project", type=str, default="Megatron-Bridge-MIMO", help="W&B project name")
    parser.add_argument("--wandb-exp-name", type=str, default="mimo-llava-e2e-test", help="W&B experiment name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb-save-dir", type=str, default="/tmp/wandb", help="W&B save directory")
    parser.add_argument(
        "--lr-warmup-iters", type=int, default=20, help="Number of iterations to linearly warmup learning rate"
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory of the LLaVA-Pretrain dataset")
    return parser.parse_args()


def main():
    global _rank_log_file

    args = parse_args()

    # 1. Initialize distributed first so we know rank
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Seed all RNGs for reproducible weight initialization
    seed = 40
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Open per-rank log file
    log_dir = "/tmp/claude-0/mimo_rank_logs"
    os.makedirs(log_dir, exist_ok=True)
    _rank_log_file = open(f"{log_dir}/rank_{rank}.log", "w")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/rank_{rank}_full.log", mode="w"), logging.StreamHandler(sys.stderr)],
        force=True,
    )

    _log(f"distributed initialized (world_size={dist.get_world_size()})")

    # 2. Build model provider
    _log("building model specs")
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()

    mimo_provider = MimoModelProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        mimo_parallelism_config=None,  # Homogeneous mode
        topology={"images": ["llm"], "llm": []},
        use_cpu_initialization=True,
        bf16=True,
        vocab_size=VOCAB_SIZE,
        seq_length=MAX_SEQ_LENGTH,
    )

    # 3. Build data provider
    _log("building data provider")
    data_provider = _build_hf_data_provider(args.dataset_root)

    # 4. Build optimizer config (Bridge OptimizerConfig for ConfigContainer)
    _log("building optimizer config")
    print_rank_0 = lambda msg: _log(msg) if dist.get_rank() == 0 else None
    print_rank_0(
        f"Optimizer config: lr={args.lr}, min_lr={args.min_lr}, weight_decay={args.weight_decay}, "
        f"adam_beta1={args.adam_beta1}, adam_beta2={args.adam_beta2}, clip_grad={args.clip_grad}"
    )
    opt_config = OptimizerConfig(lr=args.lr, min_lr=args.min_lr, bf16=True)

    # 5. Build config container
    _log("building config")
    cfg = _build_config(
        mimo_provider,
        data_provider,
        opt_config,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        train_iters=args.train_iters,
        log_interval=args.log_interval,
        wandb_project=args.wandb_project,
        wandb_exp_name=args.wandb_exp_name,
        wandb_entity=args.wandb_entity,
        wandb_save_dir=args.wandb_save_dir,
        lr_warmup_iters=args.lr_warmup_iters,
    )

    # Configure checkpointing from CLI args
    if args.checkpoint_interval is not None:
        cfg.checkpoint.save_interval = args.checkpoint_interval
    if args.checkpoint_dir is not None:
        cfg.checkpoint.save = args.checkpoint_dir
    if args.load_checkpoint is not None:
        cfg.checkpoint.load = args.load_checkpoint

    # Pre-cache the HF tokenizer on rank 0 before all ranks try simultaneously
    if rank == 0:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dist.barrier()

    # 6. Run training
    _log("launching pretrain()")
    pretrain(cfg, forward_step_func)

    _log("PASSED")

    # 7. Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
