# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


from __future__ import annotations

import argparse
import logging
import os
import random
import sys

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
# Parallelism config (8 GPUs: TP=4 for both modules)
# ---------------------------------------------------------------------------

from megatron.bridge.models.mimo.mimo_config import (
    MimoParallelismConfig,
    ModuleParallelismConfig,
)


def _build_parallelism_config() -> MimoParallelismConfig:
    return MimoParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=0,
            ),
            "images": ModuleParallelismConfig(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=4,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

from megatron.bridge.data.mimo.hf_provider import HFMimoDatasetProvider


def _llava_preprocess(example, dataset_root):
    """Convert LLaVA conversations format to plain text and resolve image paths."""
    conversations = example.get("conversations", [])
    text_parts = [turn.get("value", "") for turn in conversations]
    example["text"] = " ".join(text_parts).replace("<image>", "").strip()
    # Resolve relative image paths to absolute paths
    if "image" in example and example["image"] and not os.path.isabs(example["image"]):
        example["image"] = os.path.join(dataset_root, example["image"])
    return example


def _build_hf_data_provider(dataset_root: str) -> HFMimoDatasetProvider:
    """Build an HFMimoDatasetProvider for liuhaotian/LLaVA-Pretrain."""
    provider = HFMimoDatasetProvider(
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

    return provider


def _wrap_iter(loader_iter):
    """Adapt data-loader batches for the MIMO model.

    Transforms:
    - modality_inputs["images"]["pixel_values"] → modality_inputs["images"]["clip"]["x"]
      so VisionModalitySubmodules.encode() finds the "clip" encoder key and
      CLIPViTModel.forward() receives ``x=...``.
    - Sets attention_mask=None (not needed for this test).
    - Generates loss_mask if not present.
    """
    for batch in loader_iter:
        # Move tensors to GPU
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(non_blocking=True)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.cuda(non_blocking=True)
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, torch.Tensor):
                                value[k][kk] = vv.cuda(non_blocking=True)

        # Rewrap modality_inputs: {"images": {"pixel_values": t}} → {"images": {"clip": {"x": t}}}
        # Cast to bfloat16 to match model weights
        mi = batch.get("modality_inputs")
        if mi and "images" in mi:
            pv = mi["images"].get("pixel_values")
            if pv is not None:
                mi["images"] = {"clip": {"x": pv.to(torch.bfloat16)}}

        # Ensure loss_mask exists
        if "loss_mask" not in batch or batch["loss_mask"] is None:
            batch["loss_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)

        # Drop attention_mask (not needed)
        batch["attention_mask"] = None

        yield batch


def _build_data_iterators(cfg, mimo_infra):
    """Build data iterators compatible with setup_mimo's build_data_iterators_fn.

    Signature: (cfg, mimo_infra) -> (train_iter, valid_iter)
    Uses build_mimo_data_loaders which auto-detects MIMO path via cfg.model.
    """
    from megatron.bridge.data.mimo.loaders import build_mimo_data_loaders
    from megatron.bridge.training.state import TrainState

    train_state = TrainState()

    # Compute sample counts
    train_samples = cfg.train.train_iters * cfg.train.global_batch_size
    valid_samples = 0
    test_samples = 0

    train_loader, _, _ = build_mimo_data_loaders(
        cfg=cfg,
        train_state=train_state,
        mimo_provider=cfg.dataset,
        train_samples=max(train_samples, 10),  # min 10 samples
        valid_samples=valid_samples,
        test_samples=test_samples,
    )

    train_iter = _wrap_iter(train_loader) if train_loader is not None else None
    valid_iter = None
    return train_iter, valid_iter


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

from megatron.core.optimizer.optimizer_config import OptimizerConfig as MCoreOptimizerConfig

from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.config import OptimizerConfig as BridgeOptimizerConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig


def _build_config(
    mimo_provider: MimoModelProvider,
    data_provider: HFMimoDatasetProvider,
    opt_config: BridgeOptimizerConfig,
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
    # Runtime patches for MIMO
    train_cfg.num_microbatches = 1
    train_cfg.grad_reduce_in_fp32 = False
    train_cfg.overlap_grad_reduce = False
    train_cfg.use_distributed_optimizer = True
    train_cfg.check_for_nan_in_grad = False
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

    cfg = ConfigContainer(
        train=train_cfg,
        model=mimo_provider,
        optimizer=opt_config,
        scheduler=scheduler_cfg,
        dataset=data_provider,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
    )
    cfg.data_parallel_size = 1
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from megatron.bridge.training.mimo_step import forward_step as mimo_forward_step
from megatron.bridge.training.pretrain_mimo import pretrain_mimo


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
    parser = argparse.ArgumentParser(description="MIMO LLaVA training")
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
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Open per-rank log file
    log_dir = os.environ.get("MIMO_LOG_DIR", "/tmp/mimo_llava_logs")
    os.makedirs(log_dir, exist_ok=True)
    _rank_log_file = open(f"{log_dir}/rank_{rank}.log", "w")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/rank_{rank}_full.log", mode="w"), logging.StreamHandler(sys.stderr)],
        force=True,
    )
    # Enable debug logging for bridge communicator to trace P2P ops
    logging.getLogger("megatron.core.pipeline_parallel.bridge_communicator").setLevel(logging.DEBUG)
    logging.getLogger("megatron.core.pipeline_parallel.multimodule_communicator").setLevel(logging.DEBUG)

    _log(f"distributed initialized (world_size={dist.get_world_size()})")

    # No parallel_state.initialize_model_parallel() — MIMO manages its own
    # parallelism via HyperCommGrids and pg_collections. Float16Module is
    # skipped (direct bf16 cast), and cross_entropy_loss_fusion=True ensures
    # the fused CE path uses pg_collection.tp instead of global parallel_state.

    # 2. Build model provider
    _log("building model specs")
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()
    mimo_parallelism_config = _build_parallelism_config()

    mimo_provider = MimoModelProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        mimo_parallelism_config=mimo_parallelism_config,
        topology={"images": ["language"], "language": []},
        use_cpu_initialization=True,
        bf16=True,
    )
    # Patch: training_log accesses config.model.num_moe_experts
    if not hasattr(mimo_provider, "num_moe_experts"):
        mimo_provider.num_moe_experts = None

    # 4. Build data provider
    _log("building data provider")
    data_provider = _build_hf_data_provider(args.dataset_root)

    # 5. Build optimizer configs
    # MCore OptimizerConfig (with __post_init__) for get_mimo_optimizer
    _log("building optimizer configs")
    print_rank_0 = lambda msg: _log(msg) if dist.get_rank() == 0 else None
    print_rank_0(
        f"Optimizer config: lr={args.lr}, min_lr={args.min_lr}, weight_decay={args.weight_decay}, "
        f"adam_beta1={args.adam_beta1}, adam_beta2={args.adam_beta2}, clip_grad={args.clip_grad}"
    )
    mcore_opt_config = MCoreOptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        clip_grad=args.clip_grad,
        bf16=True,
        use_distributed_optimizer=True,
    )
    # Bridge OptimizerConfig (deferred post_init) for ConfigContainer
    bridge_opt_config = BridgeOptimizerConfig(lr=args.lr, min_lr=args.min_lr, use_distributed_optimizer=True)

    # 6. Build config container
    _log("building config")
    cfg = _build_config(
        mimo_provider,
        data_provider,
        bridge_opt_config,
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

    # 7. Run training
    _log("launching pretrain_mimo")
    pretrain_mimo(
        cfg=cfg,
        mimo_provider=mimo_provider,
        forward_step_func=mimo_forward_step,
        build_data_iterators_fn=_build_data_iterators,
        opt_config=mcore_opt_config,
    )

    _log("PASSED")

    # 8. Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
