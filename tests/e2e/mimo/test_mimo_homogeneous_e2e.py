# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""End-to-end homogeneous MIMO training test.

Exercises the standard pretrain() loop with MimoModelProvider in homogeneous
mode (mimo_parallelism_config=None). All modules (LLM + vision encoder) run
on every rank together. The LLM uses TP=4, PP=1, DP=2 across 8 GPUs.

Run:
    torchrun --nproc_per_node=8 tests/e2e/mimo/test_mimo_homogeneous_e2e.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.data.mimo.base_provider import MimoDatasetProvider
from megatron.bridge.data.mimo.collate import mimo_collate_fn
from megatron.bridge.training.config import DatasetBuildContext


_ENCODER_SEQ_LEN = 197  # (224/16)^2 = 196 patches + 1 class token
_SPECIAL_TOKEN_ID = 32000
_VOCAB_SIZE = 50304
_SEQ_LENGTH = 256
_IMG_SIZE = 224
_PATCH_DIM = 16


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
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
    )


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
# Synthetic dataset provider
# ---------------------------------------------------------------------------

from torch.utils.data import Dataset


class SyntheticMultimodalDataset(Dataset):
    """Generates random multimodal batches for testing."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        input_ids[:_ENCODER_SEQ_LEN] = _SPECIAL_TOKEN_ID
        labels = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones(self.seq_length, dtype=torch.float)
        position_ids = torch.arange(self.seq_length)
        pixel_values = torch.randn(3, _IMG_SIZE, _IMG_SIZE)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "modality_inputs": {
                "vision": {"pixel_values": pixel_values},
            },
        }


@dataclass
class SyntheticMimoDatasetProvider(MimoDatasetProvider):
    """Minimal MIMO dataset provider for pretrain()'s setup()."""

    seq_length: int = _SEQ_LENGTH
    vocab_size: int = _VOCAB_SIZE
    num_samples: int = 64
    dataloader_type: str = "cyclic"

    def build_datasets(self, context: DatasetBuildContext):
        dataset = SyntheticMultimodalDataset(
            num_samples=self.num_samples,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
        )
        dataset.collate_fn = self.get_collate_fn()
        return dataset, None, None

    def get_collate_fn(self):
        return partial(mimo_collate_fn, modality_names=["vision"])


# ---------------------------------------------------------------------------
# Forward step function
# ---------------------------------------------------------------------------

from megatron.bridge.training.mimo_step import loss_func as mimo_loss_func


def forward_step_func(data_iterator, model):
    """Forward step for homogeneous MIMO via pretrain()."""
    batch = next(data_iterator)

    input_ids = batch["input_ids"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    position_ids = batch["position_ids"].cuda(non_blocking=True)

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

    output_tensor, loss_mask = output
    if loss_mask is None:
        loss_mask = torch.ones_like(output_tensor)

    return output_tensor, partial(mimo_loss_func, loss_mask)


# ---------------------------------------------------------------------------
# Config + main
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
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.config import TokenizerConfig


_rank_log_file = None


def _log(msg):
    global _rank_log_file
    rank = dist.get_rank() if dist.is_initialized() else "?"
    line = f"[Rank {rank}] {msg}\n"
    if _rank_log_file:
        _rank_log_file.write(line)
        _rank_log_file.flush()
    print(line, end="", flush=True)


def main():
    global _rank_log_file

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    log_dir = "/tmp/mimo_homogeneous_e2e_logs"
    os.makedirs(log_dir, exist_ok=True)
    _rank_log_file = open(f"{log_dir}/rank_{rank}.log", "w")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/rank_{rank}_full.log", mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )

    _log(f"distributed initialized (world_size={dist.get_world_size()})")

    _log("building model specs")
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()

    mimo_provider = MimoModelProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        mimo_parallelism_config=None,  # Homogeneous mode
        topology={"vision": ["llm"], "llm": []},
        use_cpu_initialization=True,
        vocab_size=_VOCAB_SIZE,
        seq_length=_SEQ_LENGTH,
    )

    if not hasattr(mimo_provider, "num_moe_experts"):
        mimo_provider.num_moe_experts = None

    _log("building config")
    train_cfg = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=2,  # DP=2, so 1 micro-batch per DP rank
        train_iters=2,
    )
    train_cfg.log_interval = 1

    logger_cfg = LoggerConfig()
    logger_cfg.log_interval = 1

    dataset_provider = SyntheticMimoDatasetProvider(
        seq_length=_SEQ_LENGTH,
        vocab_size=_VOCAB_SIZE,
    )

    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        check_for_nan_in_grad=False,
    )

    cfg = ConfigContainer(
        train=train_cfg,
        model=mimo_provider,
        optimizer=OptimizerConfig(lr=1e-4, min_lr=0.0, bf16=True),
        scheduler=SchedulerConfig(start_weight_decay=0.01, end_weight_decay=0.01),
        dataset=dataset_provider,
        ddp=ddp_cfg,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
        ),
        checkpoint=CheckpointConfig(),
        validation=ValidationConfig(eval_interval=2, eval_iters=0),
    )

    # Pre-cache the HF tokenizer on rank 0 before all ranks try simultaneously
    if rank == 0:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2")
    dist.barrier()

    _log("launching pretrain()")
    pretrain(cfg, forward_step_func)

    _log("PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
