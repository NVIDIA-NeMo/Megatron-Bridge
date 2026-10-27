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

"""Train Qwen3.5-35B-A3B on agentic trajectories with online packing on four GB200 GPUs.

The model (ten GDN/GDN/GDN/attention cycles plus one MTP layer) comes from the config that
``01_prepare_qwen35_35b_tokenizer.py`` wrote next to the tokenizer, so tokenizer, chat template and
architecture stay in one verified directory. Data is the ``training.jsonl`` and ``validation.jsonl``
produced by ``02_prepare_coderforge.py`` and, optionally, ``02_5_filter_coderforge.py``.

Modes (all use TP1/PP1, base CP4, EP4, micro batch 1 and per-token loss):

* ``online_static_cp``: every sample runs on the full CP4 group (``dp_balanced`` scheduler).
* ``online_dynamic_cp``: every packed bin runs on a CP1, CP2 or CP4 group sized for its longest
  sequence (``default_dynamic_cp`` scheduler). Needs Megatron-Core with dynamic CP support for hybrid
  models; see the README.

Example::

    uv run python -m torch.distributed.run --nproc_per_node=4 examples/agentic_sft/04_train_qwen35_35b_a3b.py \\
        online_dynamic_cp --data-dir agentic_sft_workspace/data/coderforge_mixed

Arguments the script does not recognize are applied as dotted ``ConfigContainer`` overrides, e.g.
``train.train_iters=20``.

Adapted from the Megatron-LM agentic SFT example by Zhongbo Zhu.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("agentic_sft_workspace")
MODES = ("online_static_cp", "online_dynamic_cp")
SEED = 1234
BASE_CONTEXT_PARALLEL_SIZE = 4


def parse_cli_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse the script's flags; everything else is returned as dotted config overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=MODES, help="Context-parallel scheduling mode.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--tokenizer-dir", type=Path, help="Default: WORKSPACE/models/qwen35_35b_a3b/tokenizer")
    parser.add_argument("--data-dir", type=Path, help="Default: WORKSPACE/data/coderforge")
    parser.add_argument("--save-dir", type=Path, help="Default: a run directory under WORKSPACE/runs")
    parser.add_argument("--seq-length", type=int, default=65536)
    parser.add_argument(
        "--max-seqlen-per-rank",
        type=int,
        help="Token capacity of one rank per bin; default seq-length / 4, so the longest sample needs CP4.",
    )
    parser.add_argument("--global-batch-size", type=int, default=32, help="Trajectories per step.")
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Keep a shallower prefix of whole GDN/attention cycles (multiple of 4); default: the prepared config.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers per rank.")
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), default="mxfp8")
    parser.add_argument("--hybrid-ep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-recompute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--optimizer-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep FP32 master weights and Adam state on the host (needs about 850 GiB of host memory per node).",
    )
    parser.add_argument("--init-mode", choices=("scratch", "checkpoint"), default="scratch")
    parser.add_argument("--pretrained-checkpoint", type=Path, help="Megatron checkpoint for --init-mode checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume from --save-dir instead of starting fresh.")
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Build and print the configuration, then exit.")
    args, overrides = parser.parse_known_args(argv)
    args.tokenizer_dir = args.tokenizer_dir or args.workspace / "models/qwen35_35b_a3b/tokenizer"
    args.data_dir = args.data_dir or args.workspace / "data/coderforge"
    if args.max_seqlen_per_rank is None:
        args.max_seqlen_per_rank = args.seq_length // BASE_CONTEXT_PARALLEL_SIZE
    if args.init_mode == "checkpoint" and args.pretrained_checkpoint is None:
        parser.error("--init-mode checkpoint requires --pretrained-checkpoint")
    if args.init_mode == "scratch" and args.pretrained_checkpoint is not None:
        parser.error("--pretrained-checkpoint requires --init-mode checkpoint")
    if args.num_layers is not None and (args.num_layers < 4 or args.num_layers % 4):
        parser.error("--num-layers must be a positive multiple of 4")
    if args.seq_length < 32 or args.seq_length % 32:
        parser.error("--seq-length must be a positive multiple of 32")
    if args.save_dir is None:
        args.save_dir = args.workspace / "runs" / run_name(args)
    return args, overrides


def run_name(args: argparse.Namespace) -> str:
    """Name a run by everything that makes two experiments incomparable, so they cannot resume each other."""
    layers = "full" if args.num_layers is None else f"l{args.num_layers}"
    return (
        f"qwen35_35b_a3b_{args.mode}_{args.init_mode}_{args.precision}_offload{int(args.optimizer_cpu_offload)}"
        f"_{layers}_s{args.seq_length}_cap{args.max_seqlen_per_rank}_gbs{args.global_batch_size}"
    )


def _set(target: Any, **fields: Any) -> None:
    """Assign declared fields only; a misspelled or unsupported field fails instead of becoming a phantom."""
    for name, value in fields.items():
        if not hasattr(target, name):
            raise ValueError(f"{type(target).__name__} has no field '{name}' (unsupported Megatron-Core pin?)")
        setattr(target, name, value)


def build_model_config(args: argparse.Namespace) -> Any:
    """Build the model config from the prepared Hugging Face text config and apply the architecture choices."""
    text_config = AutoConfig.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    if text_config.model_type != "qwen3_5_moe_text":
        raise ValueError(f"Expected a Qwen3.5 MoE text config in {args.tokenizer_dir}; run script 01 first.")
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]
    model = AutoBridge.from_hf_config(text_config).get_model_config()
    if args.num_layers is not None:
        if args.num_layers > model.num_layers:
            raise ValueError(f"--num-layers {args.num_layers} exceeds the prepared {model.num_layers} layers")
        model.num_layers = args.num_layers
        # The 3:1 GDN:attention layout is a per-layer list; keep whole cycles.
        if isinstance(model.linear_attention_freq, list):
            model.linear_attention_freq = list(model.linear_attention_freq[: args.num_layers])
    return model


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Translate the example's training contract into a Megatron Bridge configuration."""
    cfg = _sft_common()
    cfg.model = build_model_config(args)
    cfg.tokenizer.tokenizer_model = str(args.tokenizer_dir)
    model = cfg.model
    dynamic = args.mode == "online_dynamic_cp"

    # Four physical ranks; dynamic CP can regroup them as CP1, CP2 or CP4.
    _set(
        model,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=BASE_CONTEXT_PARALLEL_SIZE,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.bfloat16,
        seq_length=args.seq_length,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=True,
        max_seqlen_per_dp_cp_rank=args.max_seqlen_per_rank,
    )
    if dynamic:
        _set(model, dynamic_context_parallel=True, min_dynamic_context_parallel_size=1)

    # One online item is a selected conversation. Targets are the chat template's generation mask: assistant
    # output only, rendered once, truncated on the right, no EOS added.
    cfg.dataset = GPTSFTDatasetConfig(
        dataset_root=str(args.data_dir),
        seq_length=args.seq_length,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant", mask_source="generation"),
        enable_global_batch_packing=True,
        dataloader_type="single",
        fold_alignment_padding=True,
        num_workers=args.num_workers,
        do_validation=True,
        do_test=False,
        seed=SEED,
    )

    _set(
        cfg.train,
        micro_batch_size=1,
        global_batch_size=args.global_batch_size,
        train_iters=args.train_iters,
    )
    _set(cfg.validation, eval_interval=args.eval_interval, eval_iters=args.eval_iters)
    _set(cfg.rng, seed=SEED)
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=args.train_iters // 10,
        lr_decay_iters=args.train_iters,
        max_lr=2e-5,
        min_lr=2e-6,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.033,
        clip_grad=1.0,
    )
    _set(
        cfg.optimizer,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    # Per-token loss normalizes by supervised tokens, so gradients are summed and rescaled once.
    _set(cfg.ddp, average_in_collective=False, grad_reduce_in_fp32=True, use_distributed_optimizer=True)

    _apply_performance(cfg, args)
    _apply_checkpointing(cfg, args)
    _set(cfg.logger, log_interval=1, tensorboard_dir=str(args.save_dir / "tensorboard"))
    _set(cfg.dist, distributed_timeout_minutes=30)
    return cfg


def _apply_performance(cfg: ConfigContainer, args: argparse.Namespace) -> None:
    """Backend and memory choices; none of them changes what the model is trained on."""
    model = cfg.model
    _set(
        model,
        transformer_impl="transformer_engine",
        apply_rope_fusion=False,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_router_fusion=True,
        bias_activation_fusion=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="native",
    )
    env = {
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM": 0,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "HF_HUB_OFFLINE": 1,
        "HF_DATASETS_OFFLINE": 1,
        # The CPU optimizer needs host threads; otherwise keep BLAS single-threaded.
        "OMP_NUM_THREADS": 12 if args.optimizer_cpu_offload else 1,
        "OPENBLAS_NUM_THREADS": 1,
    }

    if args.optimizer_cpu_offload:
        _set(
            cfg.optimizer,
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            use_precision_aware_optimizer=True,
        )

    if args.hybrid_ep:
        apply_flex_dispatcher_backend(model, "hybridep")
        _set(model, moe_flex_dispatcher_num_sms=32, moe_hybridep_pad_uneven_dispatch_inputs=True)
        env.update(
            USE_MNNVL=1,
            NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=BASE_CONTEXT_PARALLEL_SIZE,
            NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128,
        )
    else:
        apply_flex_dispatcher_backend(model, None)

    if args.precision == "mxfp8":
        cfg.mixed_precision = bf16_with_mxfp8_mixed()
        if args.optimizer_cpu_offload:
            # MXFP8 compute keeps BF16 parameter storage and gather when the optimizer runs on the host.
            _set(cfg.mixed_precision, fp8_param_gather=False, reuse_grad_buf_for_mxfp8_param_ag=False)
        _set(
            model,
            moe_use_grouped_tensor=True,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
        )
        env.update(NVTE_CUTEDSL_FUSED_GROUPED_MLP=1, CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=1)
    else:
        cfg.mixed_precision = bf16_mixed()
        env.update(NVTE_CUTEDSL_FUSED_GROUPED_MLP=0, CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL=0)

    if args.full_recompute:
        _set(model, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    else:
        _set(model, recompute_granularity="selective", recompute_modules=["gdn_norm_out", "moe"])

    if args.ddp_overlap:
        _set(cfg.ddp, overlap_grad_reduce=True, overlap_param_gather=True)
        _set(cfg.optimizer, overlap_param_gather=True)
    else:
        _set(cfg.ddp, overlap_grad_reduce=False, overlap_param_gather=False)
        _set(cfg.optimizer, overlap_param_gather=False)
    cfg.env_vars = {**cfg.env_vars, **env}


def _apply_checkpointing(cfg: ConfigContainer, args: argparse.Namespace) -> None:
    """Start from scratch, from converted weights, or resume; refuse to mix runs silently."""
    save_dir = args.save_dir
    has_checkpoint = (save_dir / "latest_checkpointed_iteration.txt").is_file()
    if has_checkpoint and not args.resume:
        raise ValueError(f"A checkpoint exists in {save_dir}: use --resume or select a fresh --save-dir")
    if args.resume and not has_checkpoint:
        raise ValueError(f"--resume found no checkpoint in {save_dir}")
    _set(
        cfg.checkpoint,
        save=str(save_dir),
        save_interval=args.train_iters,
        load=str(save_dir) if args.resume else None,
        pretrained_checkpoint=None,
        load_optim=args.resume,
        load_rng=args.resume,
    )
    if args.init_mode == "checkpoint":
        # Weights only: optimizer state and RNG start fresh for fine-tuning.
        _set(cfg.checkpoint, pretrained_checkpoint=str(args.pretrained_checkpoint))


def main(argv: list[str] | None = None) -> None:
    """Build the configuration, apply dotted overrides and start training."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args, overrides = parse_cli_args(argv if argv is not None else sys.argv[1:])
    cfg = build_config(args)
    merged, excluded_fields = create_omegaconf_dict_config(cfg)
    if overrides:
        merged = parse_hydra_overrides(merged, overrides)
    apply_overrides(cfg, OmegaConf.to_container(merged, resolve=True), excluded_fields)

    if get_rank_safe() == 0:
        logger.info(
            "%s: GBS=%d trajectories, MBS=1, TP1/PP1/base CP%d/EP4, seq_length=%d, per-rank cap=%d, layers=%s+MTP1",
            args.mode,
            cfg.train.global_batch_size,
            BASE_CONTEXT_PARALLEL_SIZE,
            cfg.model.seq_length,
            cfg.model.max_seqlen_per_dp_cp_rank,
            cfg.model.num_layers,
        )
        logger.info(
            "precision=%s hybrid_ep=%s full_recompute=%s optimizer_cpu_offload=%s output=%s",
            args.precision,
            args.hybrid_ep,
            args.full_recompute,
            args.optimizer_cpu_offload,
            args.save_dir,
        )
    if args.dry_run:
        return
    args.save_dir.mkdir(parents=True, exist_ok=True)
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
