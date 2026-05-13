# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""End-to-end MIMO training test.

Exercises the full training loop: pretrain_megatron_mimo -> setup_megatron_mimo -> train_megatron_mimo
on 8 GPUs with synthetic data using the real data pipeline.
LLM on ranks 0-3 (TP=4), vision encoder on ranks 4-7 (TP=4).

Run:
    torchrun --nproc_per_node=8 examples/models/megatron_mimo/megatron_mimo_training_e2e.py
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


_ENCODER_SEQ_LEN = 197  # (224/16)^2 = 196 patches + 1 class token
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


from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


def _build_parallelism_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
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


from megatron.bridge.data.megatron_mimo.mock_provider import MockMegatronMIMOProvider


def _build_mock_data_provider() -> MockMegatronMIMOProvider:
    """Build a MockMegatronMIMOProvider with HF processor (CLIP) and tokenizer (GPT-2)."""
    provider = MockMegatronMIMOProvider(
        seq_length=_SEQ_LENGTH,
        processor_paths={"vision": "openai/clip-vit-base-patch16"},
        tokenizer_path="gpt2",
        special_token_ids={"vision": _SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"vision": _ENCODER_SEQ_LEN},
        modality_configs={
            "vision": {"type": "image", "width": _IMG_SIZE, "height": _IMG_SIZE},
        },
    )
    provider.drop_last = True
    return provider


def _wrap_iter(loader_iter):
    """Adapt data-loader batches for the MIMO model.

    Transforms:
    - modality_inputs["vision"]["pixel_values"] -> modality_inputs["vision"]["clip"]["x"]
      so VisionModalitySubmodules.encode() finds the "clip" encoder key and
      CLIPViTModel.forward() receives ``x=...``.
    - Sets attention_mask=None (not needed for this test).
    - Generates loss_mask if not present.
    """
    for batch in loader_iter:
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

        mi = batch.get("modality_inputs")
        if mi and "vision" in mi:
            pv = mi["vision"].get("pixel_values")
            if pv is not None:
                mi["vision"] = {"clip": {"x": pv.to(torch.bfloat16)}}

        if "loss_mask" not in batch or batch["loss_mask"] is None:
            batch["loss_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)

        batch["attention_mask"] = None

        yield batch


def _build_data_iterators(cfg, megatron_mimo_infra):
    """Build data iterators compatible with setup_megatron_mimo's build_data_iterators_fn."""
    from megatron.bridge.data.megatron_mimo.loaders import build_megatron_mimo_data_loaders
    from megatron.bridge.training.state import TrainState

    train_state = TrainState()

    train_samples = cfg.train.train_iters * cfg.train.global_batch_size
    valid_samples = 0
    test_samples = 0

    train_loader, valid_loader, _ = build_megatron_mimo_data_loaders(
        cfg=cfg,
        train_state=train_state,
        megatron_mimo_provider=cfg.dataset,
        train_samples=max(train_samples, 10),
        valid_samples=valid_samples,
        test_samples=test_samples,
    )

    train_iter = _wrap_iter(train_loader) if train_loader is not None else None
    valid_iter = None
    return train_iter, valid_iter


from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
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
    megatron_mimo_provider: MegatronMIMOProvider,
    mock_data_provider: MockMegatronMIMOProvider,
    opt_config: BridgeOptimizerConfig,
    log_interval: int = 1,
    wandb_project: str | None = None,
    wandb_exp_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_save_dir: str | None = None,
) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=1,
        train_iters=2,
    )
    logger_cfg = LoggerConfig()
    logger_cfg.log_interval = log_interval
    logger_cfg.wandb_project = wandb_project
    logger_cfg.wandb_exp_name = wandb_exp_name
    logger_cfg.wandb_entity = wandb_entity
    logger_cfg.wandb_save_dir = wandb_save_dir
    logger_cfg.tensorboard_dir = os.path.join(wandb_save_dir or "/tmp/tb_logs", "tb_logs") if wandb_project else None

    from megatron.core.distributed import DistributedDataParallelConfig

    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=True,
        check_for_nan_in_grad=False,
    )

    cfg = ConfigContainer(
        train=train_cfg,
        model=megatron_mimo_provider,
        optimizer=opt_config,
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        dataset=mock_data_provider,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
        ddp=ddp_cfg,
    )
    cfg.data_parallel_size = 1
    return cfg


from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo


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
    """Entry point for the MegatronMIMO training end-to-end example."""
    global _rank_log_file

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    log_dir = "/tmp/megatron_mimo_e2e_logs"
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
    logging.getLogger("megatron.core.pipeline_parallel.bridge_communicator").setLevel(logging.DEBUG)
    logging.getLogger("megatron.core.pipeline_parallel.multimodule_communicator").setLevel(logging.DEBUG)

    _log(f"distributed initialized (world_size={dist.get_world_size()})")

    succeeded = False
    try:
        _log("building model specs")
        language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()
        megatron_mimo_parallelism_config = _build_parallelism_config()

        megatron_mimo_provider = MegatronMIMOProvider(
            language_model_spec=language_model_spec,
            modality_submodules_spec=modality_submodules_spec,
            special_token_ids=special_token_ids,
            megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
            topology={"vision": ["language"], "language": []},
            use_cpu_initialization=True,
        )
        if not hasattr(megatron_mimo_provider, "num_moe_experts"):
            megatron_mimo_provider.num_moe_experts = None

        _log("building data provider")
        mock_data_provider = _build_mock_data_provider()

        opt_config = BridgeOptimizerConfig(lr=1e-4, min_lr=0.0)

        _log("building config")
        cfg = _build_config(
            megatron_mimo_provider,
            mock_data_provider,
            opt_config,
            wandb_project=os.environ.get("WANDB_PROJECT", "Megatron-Bridge-MIMO"),
            wandb_exp_name=os.environ.get("WANDB_EXP_NAME", "omni-modal-e2e-test"),
            wandb_entity=os.environ.get("WANDB_ENTITY"),
            wandb_save_dir=os.environ.get("WANDB_SAVE_DIR", "/tmp/wandb"),
        )

        _log("launching pretrain_megatron_mimo")
        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_data_iterators,
        )

        _log("PASSED")
        succeeded = True
    finally:
        if succeeded:
            dist.destroy_process_group()
        if _rank_log_file is not None:
            _rank_log_file.close()
            _rank_log_file = None


if __name__ == "__main__":
    main()
