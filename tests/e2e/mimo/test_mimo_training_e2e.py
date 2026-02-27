# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""End-to-end MIMO training test.

Exercises the full training loop: pretrain_mimo → setup_mimo → train_mimo
on 8 GPUs with synthetic data using the real data pipeline.
LLM on ranks 0-3 (TP=4), vision encoder on ranks 4-7 (TP=4).

Run:
    torchrun --nproc_per_node=8 tests/e2e/mimo/test_mimo_training_e2e.py
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec

# ---------------------------------------------------------------------------
# Model configs (inline, no examples/ imports)
# ---------------------------------------------------------------------------
from megatron.core.transformer.transformer_config import TransformerConfig


def _make_vision_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
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
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        cross_entropy_loss_fusion=True,
    )


# CLIP ViT-Base/16 on 224×224: (224/16)^2 = 196 patches + 1 class token = 197
_ENCODER_SEQ_LEN = 197
_SPECIAL_TOKEN_ID = 32000
_VOCAB_SIZE = 50304
_SEQ_LENGTH = 256
_IMG_SIZE = 224
_PATCH_DIM = 16


def _build_model_specs():
    """Return (language_model_spec, modality_submodules_spec, special_token_ids)."""
    vision_config = _make_vision_config()
    language_config = _make_language_config()

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

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip": vision_encoder},
        },
    )

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": _VOCAB_SIZE,
            "max_sequence_length": _SEQ_LENGTH,
        },
    )

    modality_submodules_spec = {"vision": vision_submodule_spec}
    special_token_ids = {"vision": _SPECIAL_TOKEN_ID}
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
            "llm": ModuleParallelismConfig(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                data_parallel_size=1,
                rank_offset=0,
            ),
            "vision": ModuleParallelismConfig(
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

from megatron.bridge.data.mimo.mock_provider import MockMimoProvider


def _build_mock_data_provider() -> MockMimoProvider:
    """Build a MockMimoProvider with HF processor (CLIP) and tokenizer (GPT-2)."""
    provider = MockMimoProvider(
        seq_length=_SEQ_LENGTH,
        processor_paths={"vision": "openai/clip-vit-base-patch16"},
        tokenizer_path="gpt2",
        special_token_ids={"vision": _SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"vision": _ENCODER_SEQ_LEN},
        modality_configs={
            "vision": {"type": "image", "width": _IMG_SIZE, "height": _IMG_SIZE},
        },
    )
    # build_mimo_data_loaders expects these DataLoader attributes on the provider
    provider.drop_last = True
    return provider


def _wrap_iter(loader_iter):
    """Adapt data-loader batches for the MIMO model.

    Transforms:
    - modality_inputs["vision"]["pixel_values"] → modality_inputs["vision"]["clip"]["x"]
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

        # Rewrap modality_inputs: {"vision": {"pixel_values": t}} → {"vision": {"clip": {"x": t}}}
        # Cast to bfloat16 to match model weights
        mi = batch.get("modality_inputs")
        if mi and "vision" in mi:
            pv = mi["vision"].get("pixel_values")
            if pv is not None:
                mi["vision"] = {"clip": {"x": pv.to(torch.bfloat16)}}

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

    train_loader, valid_loader, _ = build_mimo_data_loaders(
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
    mock_data_provider: MockMimoProvider,
    opt_config: BridgeOptimizerConfig,
) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=1,
        train_iters=2,
    )
    # Runtime patches for MIMO
    train_cfg.num_microbatches = 1
    train_cfg.grad_reduce_in_fp32 = False
    train_cfg.overlap_grad_reduce = False
    train_cfg.use_distributed_optimizer = True
    train_cfg.check_for_nan_in_grad = False
    train_cfg.log_interval = 1

    logger_cfg = LoggerConfig()
    logger_cfg.log_interval = 1

    cfg = ConfigContainer(
        train=train_cfg,
        model=mimo_provider,
        optimizer=opt_config,
        scheduler=SchedulerConfig(),
        dataset=mock_data_provider,
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


def main():
    global _rank_log_file

    # 1. Initialize distributed first so we know rank
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

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
        use_cpu_initialization=True,
    )
    # Patch: training_log accesses config.model.num_moe_experts
    if not hasattr(mimo_provider, "num_moe_experts"):
        mimo_provider.num_moe_experts = None

    # 4. Build data provider
    _log("building data provider")
    mock_data_provider = _build_mock_data_provider()

    # 5. Build optimizer configs
    # MCore OptimizerConfig (with __post_init__) for get_mimo_optimizer
    mcore_opt_config = MCoreOptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    # Bridge OptimizerConfig (deferred post_init) for ConfigContainer
    bridge_opt_config = BridgeOptimizerConfig(lr=1e-4)

    # 6. Build config container
    _log("building config")
    cfg = _build_config(mimo_provider, mock_data_provider, bridge_opt_config)

    # 7. Run training
    _log("launching pretrain_mimo")
    pretrain_mimo(
        cfg=cfg,
        mimo_provider=mimo_provider,
        forward_step_func=mimo_forward_step,
        build_data_iterators_fn=_build_data_iterators,
        opt_config=mcore_opt_config,
        schedulers={},
    )

    _log("PASSED")

    # 8. Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
