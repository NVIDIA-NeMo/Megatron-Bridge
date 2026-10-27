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

"""Train the full Qwen3.5-35B-A3B on mock agentic-length data with static or dynamic context parallelism.

The model and training configuration follow the Megatron-LM agentic SFT example by Zhongbo Zhu: the full 40-layer
Qwen3.5-35B-A3B (ten cycles of three Gated DeltaNet layers and one attention layer, plus one MTP layer), TP1/PP1/base
CP4/EP4, 64K context, micro batch 1, per-token loss, MXFP8 compute, FP32 optimizer state offloaded to the host,
HybridEP and full recompute. The model is built from the architecture embedded below, so nothing is downloaded.

Data is mock: rows of random token IDs whose lengths follow that example's ``mixed`` distribution (50% / 30% / 20% of
samples needing CP1 / CP2 / CP4), generated deterministically on the first launch together with a tiny local tokenizer.
See ``mock_agentic_data.py``.

Modes:

* ``online_static_cp``: every sample runs on the full CP4 group (``dp_balanced`` scheduler).
* ``online_dynamic_cp``: every packed bin runs on a CP1, CP2 or CP4 group sized for its longest sequence
  (``default_dynamic_cp`` scheduler). Needs Megatron-Core with dynamic CP support for hybrid models.

Example, from the repository root::

    uv run python -m torch.distributed.run --nproc_per_node=4 examples/agentic_sft/train_qwen35_35b_a3b.py \\
        online_dynamic_cp --train-iters 50

Arguments the script does not recognize are applied as dotted ``ConfigContainer`` overrides, e.g.
``train.manual_gc=true``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, Qwen3_5MoeTextConfig

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.data.sft_processing import PromptCompletionSFTPreprocessingConfig
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.callbacks import Callback, CallbackContext
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


EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

from mock_agentic_data import TOKENIZER_DIR, ensure_mock_dataset  # noqa: E402


logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("agentic_sft_workspace")
MODES = ("online_static_cp", "online_dynamic_cp")
SEED = 1234
BASE_CONTEXT_PARALLEL_SIZE = 4
VOCAB_SIZE = 248320

# The text configuration of Qwen/Qwen3.5-35B-A3B (revision 59d61f3c), with ten cycles of 3 GDN + 1 attention layers
# and one MTP layer. Embedded so that the example needs no download; pass --hf-config to use another source.
QWEN35_35B_A3B_TEXT_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "moe_intermediate_size": 512,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 512,
    "use_cache": True,
    "vocab_size": VOCAB_SIZE,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
    "num_nextn_predict_layers": 1,
    "mtp_num_layers": 1,
    "tie_word_embeddings": False,
}


class GpuMemoryHeadroom(Callback):
    """Cap PyTorch's caching allocator below the device size so NCCL and other libraries can still allocate.

    NCCL creates some communicators lazily, for example the model-parallel group's at the end of the first training step,
    when PyTorch's allocator has grown to nearly all device memory. Mixed-length batches fragment the cache more than
    uniform ones, so without headroom that allocation can fail with an out-of-memory error. With a cap, the allocator
    releases cached blocks instead of growing.
    """

    def __init__(self, fraction: float) -> None:
        """Remember the fraction of device memory PyTorch may use."""
        self.fraction = fraction

    def on_train_start(self, context: CallbackContext) -> None:
        """Apply the cap on this rank's device once the model, optimizer and data are set up."""
        torch.cuda.set_per_process_memory_fraction(self.fraction, torch.cuda.current_device())


def build_callbacks(args: argparse.Namespace) -> list[Callback]:
    """Training callbacks selected by the command line."""
    return [GpuMemoryHeadroom(args.cuda_memory_fraction)] if args.cuda_memory_fraction else []


def parse_cli_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse the script's flags; everything else is returned as dotted config overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=MODES, help="Context-parallel scheduling mode.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--data-dir", type=Path, help="Mock dataset directory; default: WORKSPACE/data/mock")
    parser.add_argument(
        "--hf-config",
        help="Hugging Face model ID or local directory whose text config replaces the embedded Qwen3.5-35B-A3B one.",
    )
    parser.add_argument("--seq-length", type=int, default=65536)
    parser.add_argument(
        "--max-seqlen-per-rank",
        type=int,
        help="Token capacity of one rank per bin; default seq-length / 4, so the longest sample needs CP4.",
    )
    parser.add_argument("--global-batch-size", type=int, default=32, help="Samples per step.")
    parser.add_argument("--train-iters", type=int, default=100)
    parser.add_argument(
        "--num-train-samples",
        type=int,
        help="Mock training rows; default: exactly train-iters x global-batch-size, so every run sees one pass.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Keep a shallower prefix of whole GDN/attention cycles (multiple of 4); default: all 40.",
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
        help="Keep FP32 master weights and Adam state on the host (needs a large host-memory allocation).",
    )
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.9,
        help="Fraction of device memory PyTorch's allocator may use, leaving the rest for NCCL; 0 disables the cap.",
    )
    parser.add_argument("--init-mode", choices=("scratch", "checkpoint"), default="scratch")
    parser.add_argument("--pretrained-checkpoint", type=Path, help="Megatron checkpoint for --init-mode checkpoint.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Write checkpoints and TensorBoard logs here (a full-model checkpoint with optimizer state is ~460 GB); "
        "default: save nothing.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from --save-dir instead of starting fresh.")
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=0, help="Validation iterations per evaluation; 0 disables.")
    parser.add_argument("--prepare-data-only", action="store_true", help="Generate the mock dataset and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print the configuration, then exit.")
    args, overrides = parser.parse_known_args(argv)
    args.data_dir = args.data_dir or args.workspace / "data/mock"
    if args.max_seqlen_per_rank is None:
        args.max_seqlen_per_rank = args.seq_length // BASE_CONTEXT_PARALLEL_SIZE
    if args.init_mode == "checkpoint" and args.pretrained_checkpoint is None:
        parser.error("--init-mode checkpoint requires --pretrained-checkpoint")
    if args.init_mode == "scratch" and args.pretrained_checkpoint is not None:
        parser.error("--pretrained-checkpoint requires --init-mode checkpoint")
    if args.resume and args.save_dir is None:
        parser.error("--resume requires --save-dir")
    if args.num_layers is not None and (args.num_layers < 4 or args.num_layers % 4):
        parser.error("--num-layers must be a positive multiple of 4")
    if args.seq_length < 32 or args.seq_length % 32:
        parser.error("--seq-length must be a positive multiple of 32")
    if not 0 <= args.cuda_memory_fraction <= 1:
        parser.error("--cuda-memory-fraction must be between 0 and 1")
    if args.train_iters < 1 or args.global_batch_size < 1 or args.eval_iters < 0 or args.eval_interval < 1:
        parser.error("train-iters, global-batch-size and eval-interval must be positive; eval-iters non-negative")
    if args.num_train_samples is None:
        args.num_train_samples = args.train_iters * args.global_batch_size
    return args, overrides


def num_validation_samples(args: argparse.Namespace) -> int:
    """Validation rows needed so every evaluation of the run finds a fresh global batch."""
    if args.eval_iters == 0:
        return 0
    evaluations = args.train_iters // args.eval_interval + 1
    return args.eval_iters * args.global_batch_size * evaluations


def _set(target: Any, **fields: Any) -> None:
    """Assign declared fields only; a misspelled or unsupported field fails instead of becoming a phantom."""
    for name, value in fields.items():
        if not hasattr(target, name):
            raise ValueError(f"{type(target).__name__} has no field '{name}' (unsupported Megatron-Core pin?)")
        setattr(target, name, value)


def load_text_config(hf_config: str | None) -> Any:
    """Return the Qwen3.5 MoE text config: the embedded 35B-A3B one, or the text config of ``hf_config``.

    Raises:
        ValueError: If the selected config is not a Qwen3.5 MoE text model.
    """
    if hf_config is None:
        return Qwen3_5MoeTextConfig(**QWEN35_35B_A3B_TEXT_CONFIG)
    config = AutoConfig.from_pretrained(hf_config)
    text_config = getattr(config, "text_config", config)
    if text_config.model_type != "qwen3_5_moe_text":
        raise ValueError(f"Expected a Qwen3.5 MoE text config from {hf_config}, got {text_config.model_type}")
    if hasattr(config, "text_config"):
        text_config.tie_word_embeddings = config.tie_word_embeddings
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]
    return text_config


def build_model_config(args: argparse.Namespace) -> Any:
    """Build the model config from the text config and apply the layer count."""
    model = AutoBridge.from_hf_config(load_text_config(args.hf_config)).get_model_config()
    if args.num_layers is not None:
        if args.num_layers > model.num_layers:
            raise ValueError(f"--num-layers {args.num_layers} exceeds the model's {model.num_layers} layers")
        model.num_layers = args.num_layers
        # The 3:1 GDN:attention layout is a per-layer list; keep whole cycles.
        if isinstance(model.linear_attention_freq, list):
            model.linear_attention_freq = list(model.linear_attention_freq[: args.num_layers])
    return model


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Translate the training contract into a Megatron Bridge configuration."""
    cfg = _sft_common()
    cfg.model = build_model_config(args)
    model = cfg.model
    # Mock rows are token IDs read back by the local tokenizer the generator writes next to them.
    _set(cfg.tokenizer, tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(args.data_dir / TOKENIZER_DIR))
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

    # Each row is a prompt and a completion; only the completion is supervised. Rows are unpacked samples that the
    # global-batch scheduler packs per step across the data- and context-parallel ranks.
    cfg.dataset = GPTSFTDatasetConfig(
        dataset_root=str(args.data_dir),
        seq_length=args.seq_length,
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input", completion_column="output", separator="", loss_mode="completion", add_eos=False
        ),
        enable_global_batch_packing=True,
        dataloader_type="single",
        fold_alignment_padding=True,
        num_workers=args.num_workers,
        do_validation=args.eval_iters > 0,
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
    _set(cfg.logger, log_interval=1)
    if args.save_dir is not None:
        _set(cfg.logger, tensorboard_dir=str(args.save_dir / "tensorboard"))
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
    """Start from scratch, from converted weights, or resume; save nothing unless asked."""
    save_dir = args.save_dir
    if save_dir is None:
        _set(cfg.checkpoint, save=None, load=None, pretrained_checkpoint=None, load_optim=False, load_rng=False)
    else:
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


def prepare_data(args: argparse.Namespace, vocab_size: int) -> dict[str, Any]:
    """Generate the mock dataset for this run (once per node) and return its manifest."""
    return ensure_mock_dataset(
        args.data_dir,
        seq_length=args.seq_length,
        num_train=args.num_train_samples,
        num_validation=num_validation_samples(args),
        vocab_size=vocab_size,
        seed=SEED,
    )


def main(argv: list[str] | None = None) -> None:
    """Build the configuration, generate the mock data, apply dotted overrides and start training."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    args, overrides = parse_cli_args(argv if argv is not None else sys.argv[1:])
    cfg = build_config(args)
    merged, excluded_fields = create_omegaconf_dict_config(cfg)
    if overrides:
        merged = parse_hydra_overrides(merged, overrides)
    apply_overrides(cfg, OmegaConf.to_container(merged, resolve=True), excluded_fields)

    if get_rank_safe() == 0:
        logger.info(
            "%s: GBS=%d samples, MBS=1, TP1/PP1/base CP%d/EP4, seq_length=%d, per-rank cap=%d, layers=%s+MTP1",
            args.mode,
            cfg.train.global_batch_size,
            BASE_CONTEXT_PARALLEL_SIZE,
            cfg.model.seq_length,
            cfg.model.max_seqlen_per_dp_cp_rank,
            cfg.model.num_layers,
        )
        logger.info(
            "precision=%s hybrid_ep=%s full_recompute=%s optimizer_cpu_offload=%s mock data=%s (%d train rows)",
            args.precision,
            args.hybrid_ep,
            args.full_recompute,
            args.optimizer_cpu_offload,
            args.data_dir,
            args.num_train_samples,
        )
    if args.dry_run:
        return
    manifest = prepare_data(args, cfg.model.vocab_size)
    if args.prepare_data_only:
        lengths = manifest["splits"]["training"]["runtime_lengths"]
        logger.info("Mock dataset ready: %d rows, %d runtime tokens, in %s", len(lengths), sum(lengths), args.data_dir)
        return
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    pretrain(config=cfg, forward_step_func=forward_step, callbacks=build_callbacks(args))


if __name__ == "__main__":
    main()
