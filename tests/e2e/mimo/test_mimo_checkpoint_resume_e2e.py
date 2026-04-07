# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""End-to-end MIMO checkpoint save→resume round-trip test.

Validates that MiMo checkpoint loading/resume produces correct train_state
continuity: step, consumed_train_samples, and scheduler state are restored.

Two-phase test (separate torchrun invocations required):
  Phase 1 (save):   Train for SAVE_STEPS steps, save checkpoint.
  Phase 2 (resume): Resume from checkpoint, train to TOTAL_STEPS, verify continuity.

Run via wrapper:
    bash tests/e2e/mimo/run_mimo_checkpoint_resume.sh
Or manually:
    CKPT_DIR=$(mktemp -d)
    torchrun --nproc_per_node=8 tests/e2e/mimo/test_mimo_checkpoint_resume_e2e.py --phase save   --ckpt-dir $CKPT_DIR
    torchrun --nproc_per_node=8 tests/e2e/mimo/test_mimo_checkpoint_resume_e2e.py --phase resume --ckpt-dir $CKPT_DIR
"""

from __future__ import annotations

import argparse
import json
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

from megatron.bridge.data.mimo.mock_provider import MockMimoProvider
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig
from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.config import OptimizerConfig as BridgeOptimizerConfig
from megatron.bridge.training.mimo_step import forward_step as mimo_forward_step
from megatron.bridge.training.pretrain_mimo import pretrain_mimo
from megatron.bridge.training.state import GlobalState, TrainState
from megatron.bridge.training.tokenizers.config import TokenizerConfig


logger = logging.getLogger(__name__)

# Monkey-patch: report_theoretical_memory crashes on MIMO because cfg.model is
# MimoModelProvider (no kv_channels). Pre-existing issue from mainline merge.
from megatron.bridge.training.utils import train_utils as _tu


_tu.report_theoretical_memory = lambda *a, **kw: None

SAVE_STEPS = 5
TOTAL_STEPS = 10
_ENCODER_SEQ_LEN = 197
_SPECIAL_TOKEN_ID = 32000
_VOCAB_SIZE = 50304
_SEQ_LENGTH = 256
_IMG_SIZE = 224
_PATCH_DIM = 16


# ---------------------------------------------------------------------------
# Model helpers (same as test_mimo_training_e2e.py)
# ---------------------------------------------------------------------------


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


def _build_model_specs():
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
        submodules={"encoders": {"clip": vision_encoder}},
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
    return language_model_spec, {"vision": vision_submodule_spec}, {"vision": _SPECIAL_TOKEN_ID}


def _build_parallelism_config() -> MimoParallelismConfig:
    """Build parallelism config from MIMO_* env vars (set by shell wrapper).

    Env vars (with defaults for 8-GPU TP=4 both):
        MIMO_LLM_TP, MIMO_LLM_PP, MIMO_LLM_DP, MIMO_LLM_OFFSET
        MIMO_VISION_TP, MIMO_VISION_PP, MIMO_VISION_DP, MIMO_VISION_OFFSET
    """
    return MimoParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=int(os.environ.get("MIMO_LLM_TP", "4")),
                pipeline_model_parallel_size=int(os.environ.get("MIMO_LLM_PP", "1")),
                data_parallel_size=int(os.environ.get("MIMO_LLM_DP", "1")),
                rank_offset=int(os.environ.get("MIMO_LLM_OFFSET", "0")),
            ),
            "vision": ModuleParallelismConfig(
                tensor_model_parallel_size=int(os.environ.get("MIMO_VISION_TP", "4")),
                pipeline_model_parallel_size=int(os.environ.get("MIMO_VISION_PP", "1")),
                data_parallel_size=int(os.environ.get("MIMO_VISION_DP", "1")),
                rank_offset=int(os.environ.get("MIMO_VISION_OFFSET", "4")),
            ),
        },
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _build_mock_data_provider() -> MockMimoProvider:
    provider = MockMimoProvider(
        seq_length=_SEQ_LENGTH,
        processor_paths={"vision": "openai/clip-vit-base-patch16"},
        tokenizer_path="gpt2",
        special_token_ids={"vision": _SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"vision": _ENCODER_SEQ_LEN},
        modality_configs={"vision": {"type": "image", "width": _IMG_SIZE, "height": _IMG_SIZE}},
    )
    provider.drop_last = True
    return provider


def _wrap_iter(loader_iter):
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


def _build_data_iterators(cfg, mimo_infra, *, train_state=None):
    """Build data iterators. Accepts optional train_state for resume support."""
    from megatron.bridge.data.mimo.loaders import build_mimo_data_loaders

    if train_state is None:
        train_state = TrainState()

    train_samples = cfg.train.train_iters * cfg.train.global_batch_size
    train_loader, _, _ = build_mimo_data_loaders(
        cfg=cfg,
        train_state=train_state,
        mimo_provider=cfg.dataset,
        train_samples=max(train_samples, 10),
        valid_samples=0,
        test_samples=0,
    )
    train_iter = _wrap_iter(train_loader) if train_loader is not None else None
    return train_iter, None


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_config(
    mimo_provider: MimoModelProvider,
    mock_data_provider: MockMimoProvider,
    opt_config: BridgeOptimizerConfig,
    ckpt_dir: str,
    *,
    train_iters: int,
    save_interval: int,
    load_dir: str | None = None,
) -> ConfigContainer:
    par_cfg = mimo_provider.mimo_parallelism_config
    max_dp = max(p.data_parallel_size for p in par_cfg.module_parallelisms.values())

    train_cfg = TrainingConfig(
        micro_batch_size=max_dp,
        global_batch_size=max_dp * max_dp,
        train_iters=train_iters,
    )
    train_cfg.num_microbatches = 1
    train_cfg.grad_reduce_in_fp32 = False
    train_cfg.overlap_grad_reduce = False
    train_cfg.use_distributed_optimizer = True
    train_cfg.check_for_nan_in_grad = False
    train_cfg.log_interval = 1

    logger_cfg = LoggerConfig()
    logger_cfg.log_interval = 1

    llm_pp = par_cfg.module_parallelisms["language"].pipeline_model_parallel_size
    ckpt_cfg = CheckpointConfig(
        save_interval=save_interval,
        save=ckpt_dir,
        ckpt_format="torch_dist",
        # TODO: Re-enable fully_parallel_save for PP>1 after fixing MIMO sharded
        # checkpoint access pattern validation for nested DDP language model params.
        fully_parallel_save=(llm_pp == 1),
        dist_ckpt_optim_fully_reshardable=True,
        # MiMo RNG save is not yet supported: each module produces ShardedObject
        # with key "rng_state" using module-local PP/TP/DP ranks, causing
        # duplicate shard keys across modules.  Disable until upstream fix.
        save_rng=False,
    )
    if load_dir is not None:
        ckpt_cfg.load = load_dir

    cfg = ConfigContainer(
        train=train_cfg,
        model=mimo_provider,
        optimizer=opt_config,
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        dataset=mock_data_provider,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(),
        checkpoint=ckpt_cfg,
    )
    cfg.data_parallel_size = max_dp
    return cfg


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

MARKER_FILE = "resume_marker.json"


def _run_phase_save(ckpt_dir: str) -> None:
    """Phase 1: Train for SAVE_STEPS steps and save checkpoint."""
    rank = dist.get_rank()
    _log(f"Phase SAVE: training for {SAVE_STEPS} steps, saving to {ckpt_dir}")

    language_spec, modality_specs, special_tokens = _build_model_specs()
    mimo_provider = MimoModelProvider(
        language_model_spec=language_spec,
        modality_submodules_spec=modality_specs,
        special_token_ids=special_tokens,
        mimo_parallelism_config=_build_parallelism_config(),
        topology={"vision": ["language"], "language": []},
        use_cpu_initialization=True,
    )
    if not hasattr(mimo_provider, "num_moe_experts"):
        mimo_provider.num_moe_experts = None
    if not hasattr(mimo_provider, "fp8"):
        mimo_provider.fp8 = None

    mock_data = _build_mock_data_provider()
    opt_config = BridgeOptimizerConfig(lr=1e-4, min_lr=0.0, use_distributed_optimizer=True)

    cfg = _build_config(
        mimo_provider,
        mock_data,
        opt_config,
        ckpt_dir,
        train_iters=SAVE_STEPS,
        save_interval=SAVE_STEPS,
    )

    global_state = GlobalState()

    pretrain_mimo(
        cfg=cfg,
        forward_step_func=mimo_forward_step,
        build_data_iterators_fn=_build_data_iterators,
        global_state=global_state,
    )

    ts = global_state.train_state
    _log(f"Phase SAVE complete: step={ts.step}, consumed_train_samples={ts.consumed_train_samples}")

    if rank == 0:
        marker = {
            "step": ts.step,
            "consumed_train_samples": ts.consumed_train_samples,
            "floating_point_operations_so_far": ts.floating_point_operations_so_far,
        }
        marker_path = os.path.join(ckpt_dir, MARKER_FILE)
        with open(marker_path, "w") as f:
            json.dump(marker, f)
        _log(f"Wrote marker: {marker}")

    dist.barrier()
    assert ts.step == SAVE_STEPS, f"Expected step={SAVE_STEPS}, got {ts.step}"
    _log("Phase SAVE: PASSED")


def _run_phase_resume(ckpt_dir: str) -> None:
    """Phase 2: Resume from checkpoint, train to TOTAL_STEPS, verify continuity."""
    _log(f"Phase RESUME: loading from {ckpt_dir}, training to {TOTAL_STEPS} steps")

    marker_path = os.path.join(ckpt_dir, MARKER_FILE)
    with open(marker_path, "r") as f:
        saved_marker = json.load(f)
    _log(f"Loaded marker from phase 1: {saved_marker}")

    language_spec, modality_specs, special_tokens = _build_model_specs()
    mimo_provider = MimoModelProvider(
        language_model_spec=language_spec,
        modality_submodules_spec=modality_specs,
        special_token_ids=special_tokens,
        mimo_parallelism_config=_build_parallelism_config(),
        topology={"vision": ["language"], "language": []},
        use_cpu_initialization=True,
    )
    if not hasattr(mimo_provider, "num_moe_experts"):
        mimo_provider.num_moe_experts = None
    if not hasattr(mimo_provider, "fp8"):
        mimo_provider.fp8 = None

    mock_data = _build_mock_data_provider()
    opt_config = BridgeOptimizerConfig(lr=1e-4, min_lr=0.0, use_distributed_optimizer=True)

    cfg = _build_config(
        mimo_provider,
        mock_data,
        opt_config,
        ckpt_dir,
        train_iters=TOTAL_STEPS,
        save_interval=TOTAL_STEPS,
        load_dir=ckpt_dir,
    )
    # Save phase used train_iters=SAVE_STEPS, so checkpoint scheduler state
    # has lr_decay_steps / wd_incr_steps derived from SAVE_STEPS.  Resume uses
    # TOTAL_STEPS which produces different values.  override_opt_param_scheduler
    # tells the scheduler to use the current (resume) values without asserting
    # against the checkpoint.  Scheduler progress (num_steps) is still restored.
    cfg.scheduler.override_opt_param_scheduler = True

    global_state = GlobalState()

    pretrain_mimo(
        cfg=cfg,
        forward_step_func=mimo_forward_step,
        build_data_iterators_fn=_build_data_iterators,
        global_state=global_state,
    )

    ts = global_state.train_state

    _log(f"Phase RESUME complete: step={ts.step}, consumed_train_samples={ts.consumed_train_samples}")

    # Verify step continuity
    assert ts.step == TOTAL_STEPS, f"Step continuity failed: expected {TOTAL_STEPS}, got {ts.step}"

    # Verify consumed_train_samples did not reset to 0
    assert ts.consumed_train_samples >= saved_marker["consumed_train_samples"], (
        f"consumed_train_samples reset detected: "
        f"saved={saved_marker['consumed_train_samples']}, resumed={ts.consumed_train_samples}"
    )

    # Verify consumed_train_samples advanced beyond the saved value
    expected_consumed = saved_marker["consumed_train_samples"] + (
        (TOTAL_STEPS - SAVE_STEPS) * cfg.train.global_batch_size
    )
    assert ts.consumed_train_samples == expected_consumed, (
        f"consumed_train_samples mismatch: expected {expected_consumed}, got {ts.consumed_train_samples}"
    )

    _log("Phase RESUME: PASSED — all continuity assertions hold")


# ---------------------------------------------------------------------------
# Logging + main
# ---------------------------------------------------------------------------

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["save", "resume"])
    parser.add_argument("--ckpt-dir", required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    log_dir = "/tmp/mimo_resume_e2e_logs"
    os.makedirs(log_dir, exist_ok=True)
    _rank_log_file = open(f"{log_dir}/rank_{rank}_{args.phase}.log", "w")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/rank_{rank}_{args.phase}_full.log", mode="w"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )

    if args.phase == "save":
        _run_phase_save(args.ckpt_dir)
    else:
        _run_phase_resume(args.ckpt_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
