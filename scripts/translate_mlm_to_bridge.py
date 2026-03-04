#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Bidirectional config translator: Megatron-LM ←→ Megatron Bridge.

Direction 1 — MLM → Bridge (default):
  Input:   YAML config or raw MLM CLI args
  Output:  Bridge Hydra overrides, standalone recipe, or launch command

Direction 2 — Bridge → MLM (``--reverse``):
  Input:   Recipe name (``--recipe``) and/or CLI overrides (``--args``)
  Output:  MLM pretrain_gpt.py CLI args or launch command

Examples:
  # MLM → Bridge: from YAML
  python scripts/translate_mlm_to_bridge.py \\
      --yaml model_configs/benchmarking/DeepSeek-V3.yaml

  # MLM → Bridge: from raw args
  python scripts/translate_mlm_to_bridge.py \\
      --args "--num-layers 61 --hidden-size 7168 --bf16 --swiglu"

  # MLM → Bridge: generate standalone recipe
  python scripts/translate_mlm_to_bridge.py \\
      --yaml DeepSeek-V3.yaml --emit recipe --recipe-name deepseek_v3

  # Bridge → MLM: recipe + overrides (most common)
  python scripts/translate_mlm_to_bridge.py --reverse \\
      --recipe llama32_1b_pretrain_config \\
      --args "train.train_iters=1000 model.tensor_model_parallel_size=2"

  # Bridge → MLM: recipe only (all defaults)
  python scripts/translate_mlm_to_bridge.py --reverse \\
      --recipe vanilla_gpt_pretrain_config

  # Bridge → MLM: overrides only (no recipe)
  python scripts/translate_mlm_to_bridge.py --reverse \\
      --args "model.num_layers=32 model.activation_func=silu model.gated_linear_unit=true"

  # Bridge → MLM: full command
  python scripts/translate_mlm_to_bridge.py --reverse --emit command \\
      --recipe llama32_1b_pretrain_config --nproc 8
"""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any


try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
#  Mapping tables:  MLM arg  →  (bridge_path, transform)
#
#  bridge_path uses dotted notation matching ConfigContainer hierarchy.
#  transform is one of:
#    - None               : pass value through as-is
#    - "flag"             : boolean flag (no value in CLI, maps to True)
#    - "flag_invert"      : boolean flag that inverts semantics
#    - "swiglu"           : special combo: gated_linear_unit + silu activation
#    - callable           : custom transformer  (value) -> list[(path, val)]
# ---------------------------------------------------------------------------

# fmt: off
ARG_MAP: dict[str, tuple[str, Any]] = {
    # ── Model architecture ──────────────────────────────────────────────
    "num-layers":                        ("model.num_layers",                     None),
    "hidden-size":                       ("model.hidden_size",                    None),
    "ffn-hidden-size":                   ("model.ffn_hidden_size",                None),
    "num-attention-heads":               ("model.num_attention_heads",            None),
    "num-query-groups":                  ("model.num_query_groups",               None),
    "kv-channels":                       ("model.kv_channels",                    None),
    "max-position-embeddings":           ("model.seq_length",                     None),
    "position-embedding-type":           ("model.position_embedding_type",        None),
    "rotary-base":                       ("model.rotary_base",                    None),
    "rotary-percent":                    ("model.rotary_percent",                 None),
    "normalization":                     ("model.normalization",                  None),
    "layernorm-epsilon":                 ("model.layernorm_epsilon",              "alias"),
    "norm-epsilon":                      ("model.layernorm_epsilon",              None),
    "init-method-std":                   ("model.init_method_std",               None),
    "make-vocab-size-divisible-by":      ("model.make_vocab_size_divisible_by",   None),
    "vocab-size":                        ("tokenizer.vocab_size",                 None),
    "padded-vocab-size":                 ("model.vocab_size",                     None),

    "window-size":                       ("model.window_size",                    None),
    "attention-backend":                 ("model.attention_backend",              None),
    "rotary-seq-len-interpolation-factor": ("model.seq_len_interpolation_factor", None),
    "attention-softmax-in-fp32":         ("model.attention_softmax_in_fp32",      "flag"),
    "fp16-lm-cross-entropy":             ("model.fp16_lm_cross_entropy",         "flag"),
    "calculate-per-token-loss":          ("model.calculate_per_token_loss",       "flag"),

    # Boolean architecture flags
    "swiglu":                            (None,                                   "swiglu"),
    "squared-relu":                      (None,                                   "squared_relu"),
    "disable-bias-linear":               ("model.add_bias_linear",                "flag_invert"),
    "add-qkv-bias":                      ("model.add_qkv_bias",                  "flag"),
    "untie-embeddings-and-output-weights": ("model.share_embeddings_and_output_weights", "flag_invert"),
    "qk-layernorm":                      ("model.qk_layernorm",                  "flag"),
    "group-query-attention":             (None,                                   "skip"),  # implied by num-query-groups
    "use-flash-attn":                    (None,                                   "skip"),  # Bridge default
    "sequence-parallel":                 ("model.sequence_parallel",              "flag"),
    "cross-entropy-loss-fusion":         ("model.cross_entropy_loss_fusion",      "flag"),
    "cross-entropy-fusion-impl":         ("model.cross_entropy_fusion_impl",      None),

    # ── MLA (Multi-Latent Attention) ────────────────────────────────────
    "multi-latent-attention":            ("model.multi_latent_attention",         "flag"),
    "q-lora-rank":                       ("model.q_lora_rank",                   None),
    "kv-lora-rank":                      ("model.kv_lora_rank",                  None),
    "qk-head-dim":                       ("model.qk_head_dim",                   None),
    "qk-pos-emb-head-dim":              ("model.qk_pos_emb_head_dim",           None),
    "v-head-dim":                        ("model.v_head_dim",                    None),
    "rotary-scaling-factor":             ("model.rotary_scaling_factor",          None),
    "mscale":                            ("model.mscale",                        None),
    "mscale-all-dim":                    ("model.mscale_all_dim",                None),

    # ── MoE ─────────────────────────────────────────────────────────────
    "num-experts":                       ("model.num_moe_experts",               None),
    "moe-layer-freq":                    ("model.moe_layer_freq",                None),
    "moe-ffn-hidden-size":               ("model.moe_ffn_hidden_size",           None),
    "moe-shared-expert-intermediate-size": ("model.moe_shared_expert_intermediate_size", None),
    "moe-router-topk":                   ("model.moe_router_topk",              None),
    "moe-router-load-balancing-type":    ("model.moe_router_load_balancing_type", None),
    "moe-aux-loss-coeff":                ("model.moe_aux_loss_coeff",            None),
    "moe-router-group-topk":             ("model.moe_router_group_topk",         None),
    "moe-router-num-groups":             ("model.moe_router_num_groups",         None),
    "moe-router-topk-scaling-factor":    ("model.moe_router_topk_scaling_factor", None),
    "moe-router-score-function":         ("model.moe_router_score_function",     None),
    "moe-router-enable-expert-bias":     ("model.moe_router_enable_expert_bias", "flag"),
    "moe-router-bias-update-rate":       ("model.moe_router_bias_update_rate",   None),
    "moe-router-dtype":                  ("model.moe_router_dtype",              None),
    "moe-grouped-gemm":                  ("model.moe_grouped_gemm",             "flag"),
    "moe-token-dispatcher-type":         ("model.moe_token_dispatcher_type",     None),
    "moe-permute-fusion":                ("model.moe_permute_fusion",            "flag"),
    "moe-router-fusion":                 ("model.moe_router_fusion",             "flag"),

    # ── MTP ─────────────────────────────────────────────────────────────
    "mtp-num-layers":                    ("model.mtp_num_layers",                None),
    "mtp-loss-scaling-factor":           ("model.mtp_loss_scaling_factor",       None),

    # ── Parallelism ─────────────────────────────────────────────────────
    "tensor-model-parallel-size":        ("model.tensor_model_parallel_size",    None),
    "pipeline-model-parallel-size":      ("model.pipeline_model_parallel_size",  None),
    "pipeline-model-parallel-layout":    ("model.pipeline_model_parallel_layout", None),
    "context-parallel-size":             ("model.context_parallel_size",         None),
    "expert-model-parallel-size":        ("model.expert_model_parallel_size",    None),
    "expert-tensor-parallel-size":       ("model.expert_tensor_parallel_size",   None),
    "virtual-pipeline-model-parallel-size": ("model.virtual_pipeline_model_parallel_size", None),
    "use-distributed-optimizer":         ("ddp.use_distributed_optimizer",       "flag"),

    # ── Training ────────────────────────────────────────────────────────
    "micro-batch-size":                  ("train.micro_batch_size",              None),
    "global-batch-size":                 ("train.global_batch_size",             None),
    "train-iters":                       ("train.train_iters",                   None),
    "train-samples":                     ("train.train_samples",                 None),
    "exit-duration-in-mins":             ("train.exit_duration_in_mins",         None),
    "exit-interval":                     ("train.exit_interval",                 None),
    "skip-train":                        ("train.skip_train",                    "flag"),
    "manual-gc":                         ("train.manual_gc",                     "flag"),
    "manual-gc-interval":                ("train.manual_gc_interval",            None),
    "seq-length":                        ("dataset.sequence_length",             None),
    "dataloader-type":                   ("dataset.dataloader_type",             None),
    "num-dataset-builder-threads":       ("dataset.num_dataset_builder_threads", None),

    # ── Optimizer / regularization ──────────────────────────────────────
    "lr":                                ("optimizer.lr",                        None),
    "min-lr":                            ("optimizer.min_lr",                    None),
    "adam-beta1":                        ("optimizer.adam_beta1",                None),
    "adam-beta2":                        ("optimizer.adam_beta2",                None),
    "adam-eps":                          ("optimizer.adam_eps",                  None),
    "weight-decay":                      ("optimizer.weight_decay",             None),
    "clip-grad":                         ("optimizer.clip_grad",                None),
    "decoupled-lr":                      ("optimizer.decoupled_lr",             None),
    "decoupled-min-lr":                  ("optimizer.decoupled_min_lr",         None),
    "attention-dropout":                 ("model.attention_dropout",             None),
    "hidden-dropout":                    ("model.hidden_dropout",               None),

    # ── LR scheduler ────────────────────────────────────────────────────
    "lr-decay-style":                    ("scheduler.lr_decay_style",           None),
    "lr-warmup-iters":                   ("scheduler.lr_warmup_iters",          None),
    "lr-warmup-samples":                 ("scheduler.lr_warmup_samples",        None),
    "lr-warmup-fraction":                ("scheduler.lr_warmup_fraction",       None),
    "lr-warmup-init":                    ("scheduler.lr_warmup_init",           None),
    "lr-decay-iters":                    ("scheduler.lr_decay_iters",           None),
    "lr-decay-samples":                  ("scheduler.lr_decay_samples",         None),
    "lr-wsd-decay-style":               ("scheduler.lr_wsd_decay_style",       None),
    "lr-wsd-decay-iters":               ("scheduler.lr_wsd_decay_iters",       None),
    "lr-wsd-decay-samples":             ("scheduler.lr_wsd_decay_samples",     None),
    "override-opt-param-scheduler":     ("scheduler.override_opt_param_scheduler", "flag"),
    "use-checkpoint-opt-param-scheduler": ("scheduler.use_checkpoint_opt_param_scheduler", "flag"),
    "start-weight-decay":               ("scheduler.start_weight_decay",       None),
    "end-weight-decay":                  ("scheduler.end_weight_decay",         None),
    "weight-decay-incr-style":           ("scheduler.weight_decay_incr_style",  None),

    # ── Data / dataset ──────────────────────────────────────────────────
    "data-path":                         ("dataset.blend",                      "data_path"),
    "split":                             ("dataset.split",                      "split"),
    "data-cache-path":                   ("dataset.path_to_cache",             None),
    "num-workers":                       ("dataset.num_workers",                None),
    "no-mmap-bin-files":                 ("dataset.mmap_bin_files",             "flag_invert"),
    "no-create-attention-mask-in-dataloader": ("dataset.skip_getting_attention_mask_from_dataset", "flag"),
    "reset-position-ids":                ("dataset.reset_position_ids",         "flag"),
    "reset-attention-mask":              ("dataset.reset_attention_mask",       "flag"),
    "eod-mask-loss":                     ("dataset.eod_mask_loss",              "flag"),

    # ── Tokenizer ───────────────────────────────────────────────────────
    "tokenizer-type":                    ("tokenizer.tokenizer_type",           None),
    "tokenizer-model":                   ("tokenizer.tokenizer_model",          None),
    "vocab-file":                        ("tokenizer.vocab_file",               None),
    "merge-file":                        ("tokenizer.merge_file",               None),

    # ── Validation ──────────────────────────────────────────────────────
    "eval-iters":                        ("validation.eval_iters",              None),
    "eval-interval":                     ("validation.eval_interval",           None),

    # ── Checkpoint ──────────────────────────────────────────────────────
    "save":                              ("checkpoint.save",                    None),
    "load":                              ("checkpoint.load",                    None),
    "save-interval":                     ("checkpoint.save_interval",           None),
    "finetune":                          ("checkpoint.finetune",                "flag"),
    "pretrained-checkpoint":             ("checkpoint.pretrained_checkpoint",   None),
    "no-save-optim":                     ("checkpoint.save_optim",             "flag_invert"),
    "no-load-optim":                     ("checkpoint.load_optim",             "flag_invert"),
    "no-load-rng":                       ("checkpoint.load_rng",               "flag_invert"),
    "auto-detect-ckpt-format":           (None,                                "skip"),
    "use-checkpoint-args":               ("checkpoint.use_checkpoint_args",     "flag"),
    "exit-on-missing-checkpoint":        ("checkpoint.exit_on_missing_checkpoint", "flag"),
    "async-save":                        ("checkpoint.async_save",             "flag"),
    "ckpt-fully-parallel-load":          ("checkpoint.fully_parallel_load",    "flag"),
    "dist-ckpt-strictness":              ("checkpoint.dist_ckpt_strictness",   None),
    "ckpt-format":                       ("checkpoint.ckpt_format",            None),

    # ── DDP ─────────────────────────────────────────────────────────────
    "overlap-grad-reduce":               ("ddp.overlap_grad_reduce",           "flag"),
    "overlap-param-gather":              ("ddp.overlap_param_gather",          "flag"),
    "no-check-for-nan-in-loss-and-grad": ("ddp.check_for_nan_in_grad",        "flag_invert"),
    "grad-reduce-in-fp32":               ("ddp.grad_reduce_in_fp32",          "flag"),

    # ── Precision ───────────────────────────────────────────────────────
    "bf16":                              ("mixed_precision._bf16",             "flag"),
    "fp16":                              ("mixed_precision._fp16",             "flag"),

    # ── Logger ──────────────────────────────────────────────────────────
    "log-interval":                      ("logger.log_interval",               None),
    "log-timers-to-tensorboard":         ("logger.log_timers_to_tensorboard",  "flag"),
    "log-memory-to-tensorboard":         ("logger.log_memory_to_tensorboard",  "flag"),
    "log-validation-ppl-to-tensorboard": ("logger.log_validation_ppl_to_tensorboard", "flag"),
    "log-throughput":                    ("logger.log_throughput",              "flag"),
    "log-num-zeros-in-grad":             ("logger.log_num_zeros_in_grad",      "flag"),
    "log-params-norm":                   ("logger.log_params_norm",            "flag"),
    "tensorboard-dir":                   ("logger.tensorboard_dir",            None),
    "wandb-project":                     ("logger.wandb_project",              None),
    "wandb-exp-name":                    ("logger.wandb_exp_name",             None),
    "logging-level":                     ("logger.logging_level",              None),

    # ── Recompute ───────────────────────────────────────────────────────
    "recompute-granularity":             ("model.recompute_granularity",       None),
    "recompute-method":                  ("model.recompute_method",            None),
    "recompute-num-layers":              ("model.recompute_num_layers",        None),

    # ── Mock data ───────────────────────────────────────────────────────
    "mock-data":                         (None,                                 "skip"),  # Bridge uses blend=null

    # ── RNG ──────────────────────────────────────────────────────────────
    "seed":                              ("rng.seed",                           None),

    # ── Misc (informational / noop in Bridge) ───────────────────────────
    "use-mcore-models":                  (None,                                 "skip"),
    "transformer-impl":                  (None,                                 "skip"),
    "distributed-timeout-minutes":       (None,                                 "skip"),
    "enable-experimental":               (None,                                 "skip"),
}
# fmt: on

# Known flags that take no value in MLM CLI (auto-detected from ARG_MAP transforms)
FLAG_ARGS = {k for k, (_, t) in ARG_MAP.items() if t in ("flag", "flag_invert", "swiglu", "squared_relu", "skip")}


# ---------------------------------------------------------------------------
#  Parsing helpers
# ---------------------------------------------------------------------------


def _try_numeric(val: str) -> int | float | str:
    """Attempt int, then float, else return string."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _try_parse_value(val: str) -> Any:
    """Parse a value string into its Python representation."""
    if val.lower() in ("true",):
        return True
    if val.lower() in ("false",):
        return False
    if val.lower() in ("none", "null"):
        return None

    # Try ast.literal_eval for lists/tuples (e.g., moe_layer_freq)
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple, dict)):
            return parsed
        return parsed
    except (ValueError, SyntaxError):
        pass

    return _try_numeric(val)


def parse_yaml_config(path: str) -> dict[str, Any]:
    """Parse a Megatron-LM style YAML into a flat arg dict.

    Expected YAML structure:
        ENV_VARS:
          KEY: VALUE
        MODEL_ARGS:
          --arg-name: value
          --flag-arg: true   # or just present
    """
    if not HAS_YAML:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = yaml.safe_load(f)

    model_args = data.get("MODEL_ARGS", data)
    env_vars = data.get("ENV_VARS", {})

    parsed: dict[str, Any] = {}
    for key, val in model_args.items():
        clean_key = key.lstrip("-")
        if isinstance(val, bool):
            if val:
                parsed[clean_key] = True
        elif val is None:
            parsed[clean_key] = True
        else:
            parsed[clean_key] = val

    return parsed, env_vars


def parse_raw_args(args_str: str) -> dict[str, Any]:
    """Parse a raw MLM CLI string into a flat arg dict."""
    tokens = shlex.split(args_str)
    parsed: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            # Check if it's a flag arg (no value)
            if key in FLAG_ARGS:
                parsed[key] = True
                i += 1
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                parsed[key] = _try_parse_value(tokens[i + 1])
                i += 2
            else:
                parsed[key] = True
                i += 1
        else:
            i += 1

    return parsed, {}


# ---------------------------------------------------------------------------
#  Translation engine
# ---------------------------------------------------------------------------


class TranslationResult:
    """Container for the translation output."""

    def __init__(self):
        self.overrides: OrderedDict[str, Any] = OrderedDict()
        self.skipped: list[tuple[str, Any]] = []
        self.unknown: list[tuple[str, Any]] = []
        self.notes: list[str] = []
        self.env_vars: dict[str, str] = {}
        self.uses_mla = False
        self.uses_moe = False
        self.raw_args: dict[str, Any] = {}

    def add_override(self, path: str, value: Any):
        self.overrides[path] = value

    def add_note(self, note: str):
        self.notes.append(note)


def translate(args: dict[str, Any], env_vars: dict[str, str] | None = None) -> TranslationResult:
    """Translate parsed MLM args into Bridge overrides."""
    result = TranslationResult()
    result.raw_args = args
    result.env_vars = env_vars or {}

    # Detect MLA / MoE usage
    result.uses_mla = "multi-latent-attention" in args
    result.uses_moe = "num-experts" in args

    if result.uses_mla:
        result.add_note("MLA detected: use MLAModelProvider instead of GPTModelProvider in recipe")
    if result.uses_moe:
        result.add_note(
            f"MoE detected ({args.get('num-experts', '?')} experts): "
            "ensure moe_token_dispatcher_type is set appropriately"
        )

    for arg_name, arg_val in args.items():
        if arg_name not in ARG_MAP:
            result.unknown.append((arg_name, arg_val))
            continue

        bridge_path, transform = ARG_MAP[arg_name]

        # Handle special transforms first (these may have bridge_path=None)
        if transform == "swiglu":
            result.add_override("model.gated_linear_unit", True)
            result.add_override("model.activation_func", "silu")
            result.add_note("swiglu: set model.gated_linear_unit=true + model.activation_func=silu")
        elif transform == "squared_relu":
            result.add_override("model.activation_func", "squared_relu")
            result.add_note("squared_relu: set model.activation_func=squared_relu")
        elif transform == "data_path":
            # Bridge blend expects ([paths], weights_or_null)
            if isinstance(arg_val, str):
                paths = arg_val.split() if " " in str(arg_val) else [arg_val]
            elif isinstance(arg_val, (list, tuple)):
                paths = list(arg_val)
            else:
                paths = [str(arg_val)]
            result.add_override(bridge_path, (paths, None))
        elif transform == "split":
            # Normalize to comma-separated string (may arrive as tuple from ast.literal_eval)
            if isinstance(arg_val, (list, tuple)):
                split_str = ",".join(str(x) for x in arg_val)
            else:
                split_str = str(arg_val)
            result.add_override(bridge_path, split_str)
        elif transform == "skip" or bridge_path is None:
            result.skipped.append((arg_name, arg_val))
        elif transform == "flag":
            result.add_override(bridge_path, True)
        elif transform == "flag_invert":
            result.add_override(bridge_path, False)
        elif transform is None:
            result.add_override(bridge_path, arg_val)
        else:
            result.add_override(bridge_path, arg_val)

    # Handle bf16/fp16 → mixed_precision string
    if "mixed_precision._bf16" in result.overrides:
        del result.overrides["mixed_precision._bf16"]
        result.add_override("mixed_precision", "bf16_mixed")
    if "mixed_precision._fp16" in result.overrides:
        del result.overrides["mixed_precision._fp16"]
        result.add_override("mixed_precision", "16-mixed")

    # Clean up internal-only keys
    for key in list(result.overrides.keys()):
        if "._" in key:
            del result.overrides[key]

    # Warn about sequence_parallel + TP=1 (Bridge validates, MLM silently ignores)
    tp = result.overrides.get("model.tensor_model_parallel_size", 1)
    sp = result.overrides.get("model.sequence_parallel")
    if sp is True and (tp is None or int(tp) <= 1):
        result.add_note(
            "sequence_parallel=true requires tensor_model_parallel_size > 1 in Bridge. "
            "Set model.sequence_parallel=false or increase TP."
        )

    return result


# ---------------------------------------------------------------------------
#  Output formatters
# ---------------------------------------------------------------------------


def _format_value_for_override(val: Any, key: str = "") -> str:
    """Format a Python value for Hydra CLI override syntax."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        if key == "dataset.split":
            return f"\"'{val}'\""
        if " " in val or "," in val:
            return f"'{val}'"
        return val
    if isinstance(val, tuple) and len(val) == 2 and key == "dataset.blend":
        # Bridge blend format: ([paths_list], weights_or_null)
        paths, weights = val
        paths_str = ",".join(str(p) for p in paths)
        if weights is None:
            return f'"[[{paths_str}],null]"'
        weights_str = ",".join(str(w) for w in weights)
        return f'"[[{paths_str}],[{weights_str}]]"'
    if isinstance(val, (list, tuple)):
        return repr(val)
    if val is None:
        return "null"
    return str(val)


def emit_overrides(result: TranslationResult) -> str:
    """Emit Hydra-style overrides for use with run_recipe.py."""
    lines = []

    lines.append("# ═══════════════════════════════════════════════════════════")
    lines.append("# Megatron Bridge Overrides (translated from MLM args)")
    lines.append("# ═══════════════════════════════════════════════════════════")

    if result.notes:
        lines.append("#")
        for note in result.notes:
            lines.append(f"# NOTE: {note}")
        lines.append("#")

    # Group overrides by section
    sections: dict[str, list[tuple[str, Any]]] = OrderedDict()
    for path, val in result.overrides.items():
        section = path.split(".")[0]
        sections.setdefault(section, []).append((path, val))

    section_labels = {
        "model": "Model Architecture",
        "train": "Training",
        "optimizer": "Optimizer",
        "scheduler": "LR Scheduler",
        "dataset": "Dataset",
        "tokenizer": "Tokenizer",
        "validation": "Validation",
        "checkpoint": "Checkpoint",
        "ddp": "Distributed Data Parallel",
        "logger": "Logging",
        "mixed_precision": "Mixed Precision",
    }

    for section, entries in sections.items():
        label = section_labels.get(section, section)
        lines.append(f"\n# ── {label} {'─' * max(1, 50 - len(label))}")
        for path, val in entries:
            lines.append(f"  {path}={_format_value_for_override(val, key=path)} \\")

    if result.unknown:
        lines.append("\n# ── Unknown args (not mapped) ─────────────────────────")
        for arg_name, arg_val in result.unknown:
            lines.append(f"#   --{arg_name} {arg_val}")

    if result.skipped:
        lines.append("\n# ── Skipped args (not needed in Bridge) ───────────────")
        for arg_name, arg_val in result.skipped:
            lines.append(f"#   --{arg_name}")

    return "\n".join(lines)


def emit_command(result: TranslationResult, base_recipe: str = "vanilla_gpt_pretrain_config") -> str:
    """Emit the full run_recipe.py launch command."""
    lines = []

    if result.env_vars:
        lines.append("# Environment variables")
        for k, v in result.env_vars.items():
            lines.append(f"export {k}={v}")
        lines.append("")

    lines.append("torchrun --nproc_per_node=8 scripts/training/run_recipe.py \\")
    lines.append(f"  --recipe {base_recipe} \\")

    for path, val in result.overrides.items():
        lines.append(f"  {path}={_format_value_for_override(val, key=path)} \\")

    # Remove trailing backslash from last line
    if lines and lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]

    return "\n".join(lines)


def emit_recipe(result: TranslationResult, recipe_name: str = "custom_model") -> str:
    """Generate a standalone Bridge recipe Python file."""
    uses_mla = result.uses_mla
    provider_cls = "MLAModelProvider" if uses_mla else "GPTModelProvider"
    provider_import_path = "megatron.bridge.models.mla_provider" if uses_mla else "megatron.bridge.models.gpt_provider"

    # Separate model vs other overrides
    model_fields: dict[str, Any] = {}
    train_fields: dict[str, Any] = {}
    opt_fields: dict[str, Any] = {}
    sched_fields: dict[str, Any] = {}
    dataset_fields: dict[str, Any] = {}
    tokenizer_fields: dict[str, Any] = {}
    validation_fields: dict[str, Any] = {}
    checkpoint_fields: dict[str, Any] = {}
    ddp_fields: dict[str, Any] = {}
    logger_fields: dict[str, Any] = {}
    misc_overrides: dict[str, Any] = {}
    mixed_precision_str: str | None = None

    for path, val in result.overrides.items():
        parts = path.split(".", 1)
        section = parts[0]
        field = parts[1] if len(parts) > 1 else None

        if section == "model" and field:
            model_fields[field] = val
        elif section == "train" and field:
            train_fields[field] = val
        elif section == "optimizer" and field:
            opt_fields[field] = val
        elif section == "scheduler" and field:
            sched_fields[field] = val
        elif section == "dataset" and field:
            dataset_fields[field] = val
        elif section == "tokenizer" and field:
            tokenizer_fields[field] = val
        elif section == "validation" and field:
            validation_fields[field] = val
        elif section == "checkpoint" and field:
            checkpoint_fields[field] = val
        elif section == "ddp" and field:
            ddp_fields[field] = val
        elif section == "logger" and field:
            logger_fields[field] = val
        elif section == "mixed_precision":
            if field is None:
                mixed_precision_str = str(val)
            else:
                misc_overrides[path] = val
        else:
            misc_overrides[path] = val

    # Build model constructor kwargs
    def _fmt_val(v: Any) -> str:
        if isinstance(v, str):
            if v.startswith("torch.") or v.startswith("F."):
                return v
            return repr(v)
        if isinstance(v, bool):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return repr(v)
        if v is None:
            return "None"
        return repr(v)

    # Handle activation_func: map short string names to F.xxx references
    _ACTIVATION_TO_IMPORT = {
        "silu": "F.silu",
        "gelu": "F.gelu",
        "relu": "F.relu",
        "sigmoid": "F.sigmoid",
        "tanh": "torch.tanh",
        "squared_relu": "squared_relu",
        "relu2": "squared_relu",
        "fast_gelu": "fast_gelu",
    }
    activation_func = model_fields.pop("activation_func", None)
    activation_func_code = _ACTIVATION_TO_IMPORT.get(str(activation_func), None) if activation_func else None

    model_kwargs_lines = []
    for k, v in model_fields.items():
        model_kwargs_lines.append(f"        {k}={_fmt_val(v)},")

    if activation_func_code:
        model_kwargs_lines.append(f"        activation_func={activation_func_code},")

    # Optimizer kwargs
    opt_kwargs = {
        "max_lr": opt_fields.pop("lr", "3e-4"),
        "min_lr": opt_fields.pop("min_lr", None),
        "adam_beta1": opt_fields.pop("adam_beta1", 0.9),
        "adam_beta2": opt_fields.pop("adam_beta2", 0.95),
        "adam_eps": opt_fields.pop("adam_eps", 1e-8),
        "weight_decay": opt_fields.pop("weight_decay", 0.1),
        "clip_grad": opt_fields.pop("clip_grad", 1.0),
    }

    lr_warmup_iters = sched_fields.pop("lr_warmup_iters", None)
    lr_warmup_samples = sched_fields.pop("lr_warmup_samples", None)
    lr_decay_iters = sched_fields.pop("lr_decay_iters", None)
    lr_decay_samples = sched_fields.pop("lr_decay_samples", None)
    lr_decay_style = sched_fields.pop("lr_decay_style", "cosine")
    lr_warmup_init = sched_fields.pop("lr_warmup_init", None)

    use_samples = (lr_warmup_samples is not None) or (lr_decay_samples is not None)

    # Tokenizer
    tok_type = tokenizer_fields.get("tokenizer_type", "HuggingFaceTokenizer")
    tok_model = tokenizer_fields.get("tokenizer_model", None)

    # Build recipe file content
    lines = []
    lines.append('"""')
    lines.append(f"Auto-generated Bridge recipe for {recipe_name}")
    lines.append("Translated from Megatron-LM pretrain_gpt.py arguments.")
    lines.append('"""')
    lines.append("")
    lines.append("import os")
    lines.append("from typing import Optional")
    lines.append("")
    lines.append("import torch")
    lines.append("import torch.nn.functional as F")
    lines.append("from megatron.core.distributed import DistributedDataParallelConfig")
    lines.append("")
    lines.append(f"from {provider_import_path} import {provider_cls}")
    lines.append("from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths")
    lines.append(
        "from megatron.bridge.recipes.utils.optimizer_utils import "
        + (
            "distributed_fused_adam_with_cosine_annealing_samples"
            if use_samples
            else "distributed_fused_adam_with_cosine_annealing"
        )
    )
    lines.append("from megatron.bridge.training.config import (")
    lines.append("    CheckpointConfig,")
    lines.append("    ConfigContainer,")
    lines.append("    GPTDatasetConfig,")
    lines.append("    LoggerConfig,")
    lines.append("    RNGConfig,")
    lines.append("    TokenizerConfig,")
    lines.append("    TrainingConfig,")
    lines.append("    ValidationConfig,")
    lines.append(")")
    lines.append("")
    lines.append("")

    # model_config function
    lines.append("def model_config(")
    lines.append(f"    tensor_parallelism: int = {model_fields.get('tensor_model_parallel_size', 1)},")
    lines.append(f"    pipeline_parallelism: int = {model_fields.get('pipeline_model_parallel_size', 1)},")
    lines.append(f"    context_parallelism: int = {model_fields.get('context_parallel_size', 1)},")
    if result.uses_moe:
        lines.append(f"    expert_parallelism: int = {model_fields.get('expert_model_parallel_size', 1)},")
    lines.append(f"    sequence_parallelism: bool = {model_fields.get('sequence_parallel', False)},")
    lines.append(f") -> {provider_cls}:")
    lines.append(f'    """Configure the {recipe_name} model."""')
    lines.append(f"    return {provider_cls}(")

    # Emit model fields (excluding parallelism fields already in function signature)
    parallelism_fields = {
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "virtual_pipeline_model_parallel_size",
    }
    for k, v in model_fields.items():
        if k in parallelism_fields:
            continue
        lines.append(f"        {k}={_fmt_val(v)},")

    if activation_func_code:
        lines.append(f"        activation_func={activation_func_code},")

    # Add parallelism from function args
    lines.append("        tensor_model_parallel_size=tensor_parallelism,")
    lines.append("        pipeline_model_parallel_size=pipeline_parallelism,")
    lines.append("        context_parallel_size=context_parallelism,")
    if result.uses_moe:
        lines.append("        expert_model_parallel_size=expert_parallelism,")
    lines.append("        sequence_parallel=sequence_parallelism,")
    vpp = model_fields.get("virtual_pipeline_model_parallel_size")
    if vpp:
        lines.append(f"        virtual_pipeline_model_parallel_size={vpp},")

    lines.append("    )")
    lines.append("")
    lines.append("")

    # pretrain_config function
    lines.append(f"def {recipe_name}_pretrain_config() -> ConfigContainer:")
    lines.append(f'    """Pre-training configuration for {recipe_name}."""')
    lines.append('    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")')
    lines.append(f'    run_output_dir = os.path.join(base_output_dir, "{recipe_name}")')
    lines.append('    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")')
    lines.append('    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")')
    lines.append("")

    # Optimizer setup
    opt_func_name = (
        "distributed_fused_adam_with_cosine_annealing_samples"
        if use_samples
        else "distributed_fused_adam_with_cosine_annealing"
    )
    lines.append(f"    opt_config, scheduler = {opt_func_name}(")
    if use_samples:
        if lr_warmup_samples is not None:
            lines.append(f"        lr_warmup_samples={lr_warmup_samples},")
        if lr_decay_samples is not None:
            lines.append(f"        lr_decay_samples={lr_decay_samples},")
    else:
        if lr_warmup_iters is not None:
            lines.append(f"        lr_warmup_iters={lr_warmup_iters},")
        if lr_decay_iters is not None:
            lines.append(f"        lr_decay_iters={lr_decay_iters},")
    lines.append(f"        max_lr={opt_kwargs['max_lr']},")
    if opt_kwargs["min_lr"] is not None:
        lines.append(f"        min_lr={opt_kwargs['min_lr']},")
    lines.append(f"        adam_beta1={opt_kwargs['adam_beta1']},")
    lines.append(f"        adam_beta2={opt_kwargs['adam_beta2']},")
    lines.append(f"        adam_eps={opt_kwargs['adam_eps']},")
    lines.append(f"        weight_decay={opt_kwargs['weight_decay']},")
    lines.append(f"        clip_grad={opt_kwargs['clip_grad']},")
    if lr_decay_style != "cosine":
        lines.append(f"        lr_decay_style={repr(lr_decay_style)},")
    lines.append("    )")

    if lr_warmup_init is not None:
        lines.append(f"    scheduler.lr_warmup_init = {lr_warmup_init}")

    # Remaining scheduler fields
    for k, v in sched_fields.items():
        lines.append(f"    scheduler.{k} = {_fmt_val(v)}")

    lines.append("")

    # Dataset blend
    seq_len = dataset_fields.get("seq_length", 4096)
    lines.append("    cfg = ConfigContainer(")
    lines.append("        model=model_config(),")
    lines.append("        train=TrainingConfig(")
    for k in [
        "train_iters",
        "train_samples",
        "global_batch_size",
        "micro_batch_size",
        "exit_duration_in_mins",
        "manual_gc",
        "manual_gc_interval",
    ]:
        if k in train_fields:
            lines.append(f"            {k}={_fmt_val(train_fields[k])},")
    lines.append("        ),")

    lines.append("        validation=ValidationConfig(")
    for k in ["eval_interval", "eval_iters"]:
        if k in validation_fields:
            lines.append(f"            {k}={_fmt_val(validation_fields[k])},")
    lines.append("        ),")

    lines.append("        optimizer=opt_config,")
    lines.append("        scheduler=scheduler,")

    # DDP
    lines.append("        ddp=DistributedDataParallelConfig(")
    ddp_defaults = {
        "check_for_nan_in_grad": True,
        "grad_reduce_in_fp32": True,
        "overlap_grad_reduce": False,
        "overlap_param_gather": False,
        "average_in_collective": True,
        "use_distributed_optimizer": False,
    }
    ddp_merged = {**ddp_defaults, **ddp_fields}
    for k, v in ddp_merged.items():
        lines.append(f"            {k}={_fmt_val(v)},")
    lines.append("        ),")

    # Dataset
    lines.append("        dataset=GPTDatasetConfig(")
    lines.append("            random_seed=1234,")
    ds_defaults = {
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "num_dataset_builder_threads": 1,
        "data_sharding": True,
        "dataloader_type": "single",
    }
    ds_merged = {**ds_defaults}
    for k, v in dataset_fields.items():
        if k == "seq_length":
            ds_merged["sequence_length"] = v
        else:
            ds_merged[k] = v
    if "sequence_length" not in ds_merged:
        ds_merged["sequence_length"] = seq_len
    for k, v in ds_merged.items():
        lines.append(f"            {k}={_fmt_val(v)},")
    lines.append("            skip_getting_attention_mask_from_dataset=True,")
    lines.append("        ),")

    # Logger
    lines.append("        logger=LoggerConfig(")
    log_defaults = {"log_interval": 10, "tensorboard_dir": None}
    log_merged = {**log_defaults, **logger_fields}
    if log_merged.get("tensorboard_dir") is None:
        lines.append(f"            log_interval={log_merged.get('log_interval', 10)},")
        lines.append("            tensorboard_dir=tensorboard_dir,")
    else:
        for k, v in log_merged.items():
            lines.append(f"            {k}={_fmt_val(v)},")
    for k in [
        "log_timers_to_tensorboard",
        "log_memory_to_tensorboard",
        "log_throughput",
        "log_validation_ppl_to_tensorboard",
    ]:
        if k in logger_fields:
            lines.append(f"            {k}={_fmt_val(logger_fields[k])},")
    lines.append("        ),")

    # Tokenizer
    lines.append("        tokenizer=TokenizerConfig(")
    lines.append(f"            tokenizer_type={repr(tok_type)},")
    if tok_model:
        lines.append(f"            tokenizer_model={repr(tok_model)},")
    lines.append("        ),")

    # Checkpoint
    lines.append("        checkpoint=CheckpointConfig(")
    ckpt_defaults = {
        "save_interval": 500,
        "ckpt_format": "torch_dist",
        "fully_parallel_save": True,
    }
    ckpt_merged = {**ckpt_defaults, **checkpoint_fields}
    for k, v in ckpt_merged.items():
        if k == "save" and v and "$" in str(v):
            lines.append(f"            # save={repr(v)},  # Contains env var - set via override")
            continue
        if k == "load" and v and "$" in str(v):
            lines.append(f"            # load={repr(v)},  # Contains env var - set via override")
            continue
        lines.append(f"            {k}={_fmt_val(v)},")
    lines.append("        ),")

    lines.append("        rng=RNGConfig(seed=1234),")

    # Mixed precision
    mp = mixed_precision_str or "bf16_mixed"
    lines.append(f"        mixed_precision={repr(mp)},")

    lines.append("    )")
    lines.append("")
    lines.append("    return cfg")

    return "\n".join(lines)


# ===========================================================================
#  Bridge → MLM reverse direction
# ===========================================================================

# ---------------------------------------------------------------------------
#  Reverse mapping tables
# ---------------------------------------------------------------------------

# Auto-build reverse of ARG_MAP: bridge_path -> (mlm_arg_name, reverse_transform)
_REVERSE_ARG_MAP: dict[str, tuple[str, str | None]] = {}
for _mlm_arg, (_bridge_path, _transform) in ARG_MAP.items():
    if _bridge_path is None or _transform == "skip":
        continue
    if "._" in _bridge_path:
        continue
    if _transform in ("swiglu", "squared_relu", "data_path", "split", "alias"):
        continue
    _rev = None
    if _transform == "flag":
        _rev = "flag"
    elif _transform == "flag_invert":
        _rev = "flag_invert"
    _REVERSE_ARG_MAP[_bridge_path] = (_mlm_arg, _rev)

# Extra Bridge→MLM boolean flags: Bridge True → emit MLM --flag
BRIDGE_BOOL_TO_MLM: dict[str, str] = {
    "model.use_te_rng_tracker": "te-rng-tracker",
    "model.layernorm_zero_centered_gamma": "apply-layernorm-1p",
    "training.gradient_accumulation_fusion": "gradient-accumulation-fusion",
    "training.cross_entropy_loss_fusion": "cross-entropy-loss-fusion",
    "training.masked_softmax_fusion": "masked-softmax-fusion",
    "training.bias_dropout_fusion": "bias-dropout-fusion",
    "training.tp_comm_overlap": "tp-comm-overlap",
    "checkpoint.use_persistent_ckpt_worker": "use-persistent-ckpt-worker",
    "checkpoint.fully_parallel_load": "ckpt-fully-parallel-load",
    "logging.log_progress": "log-progress",
    "model.fp8_param": "fp8-param-gather",
    "model.overlap_p2p_comm_warmup_flush": "overlap-p2p-communication-warmup-flush",
}

# Extra Bridge→MLM inverse booleans: Bridge False → emit MLM --flag
BRIDGE_INVERSE_BOOL_TO_MLM: dict[str, str] = {
    "training.tp_comm_overlap_split_ag": "disable-tp-comm-split-ag",
    "training.tp_comm_overlap_split_rs": "disable-tp-comm-split-rs",
    "training.tp_comm_overlap_rs_dgrad": "disable-tp-comm-bulk-dgrad",
    "training.tp_comm_overlap_bulk_wgrad": "disable-tp-comm-bulk-wgrad",
    "model.tp_comm_overlap_rs": "disable-tp-comm-overlap-rs",
    "checkpoint.fully_parallel_save": "no-ckpt-fully-parallel-save",
    "model.share_embeddings_and_output_weights": "untie-embeddings-and-output-weights",
    "rerun_state_machine.check_for_nan_in_loss": "no-check-for-nan-in-loss-and-grad",
    "model.apply_rope_fusion": "no-rope-fusion",
    "model.add_bias_linear": "disable-bias-linear",
    "model.barrier_with_L1_time": "no-barrier-with-level-1-timing",
    "model.batch_p2p_comm": "no-overlap-p2p-communication",
    "model.overlap_p2p_comm": "no-overlap-p2p-communication",
    "model.perform_initialization": "no-initialization",
    "model.tp_comm_bulk_dgrad": "disable-tp-comm-bulk-dgrad",
    "model.tp_comm_bulk_wgrad": "disable-tp-comm-bulk-wgrad",
    "model.tp_comm_overlap_ag": "disable-tp-comm-overlap-ag",
}

# Bridge-only fields with no MLM equivalent — skip silently
BRIDGE_IGNORE_KEYS: frozenset[str] = frozenset(
    {
        "_target_",
        "rng",
        "rerun_state_machine",
        "train",
        "model",
        "optimizer",
        "ddp",
        "scheduler",
        "dataset",
        "logger",
        "tokenizer",
        "checkpoint",
        "dist",
        "ft",
        "straggler",
        "nvrx_straggler",
        "profiling",
        "peft",
        "comm_overlap",
        "mixed_precision",
        "inprocess_restart",
        "logger.filter_warnings",
        "logger.modules_to_filter",
        "logger.set_level_for_all_loggers",
        "tokenizer.image_tag_type",
        "tokenizer.special_tokens",
        "tokenizer.tokenizer_prompt_format",
        "model.timers",
        "optimizer.timers",
        "model.finalize_model_grads_func",
        "model.grad_scale_func",
        "model.grad_sync_func",
        "model.no_sync_func",
        "model.param_sync_func",
        "model.cpu_offloading",
        "model.cpu_offloading_activations",
        "model.cpu_offloading_double_buffering",
        "model.cpu_offloading_num_layers",
        "model.cpu_offloading_weights",
        "optimizer.store_param_remainders",
        "model.activation_func_fp8_input_store",
        "model.batch_p2p_sync",
        "model.cuda_graph_retain_backward_graph",
        "model.cuda_graph_use_single_mempool",
        "model.deallocate_pipeline_outputs",
        "model.disable_parameter_transpose_cache",
        "model.enable_autocast",
        "model.fp8_dot_product_attention",
        "model.fp8_multi_head_attention",
        "model.hetereogenous_dist_checkpoint",
        "model.heterogeneous_block_specs",
        "model.memory_efficient_layer_norm",
        "model.moe_router_topk_limited_devices",
        "model.moe_token_dropping",
        "model.mtp_enabled",
        "model.num_microbatches_with_partial_activation_checkpoints",
        "model.output_layer_init_method.mean",
        "model.output_layer_init_method.std",
        "model.parallel_output",
        "model.scatter_embedding_sequence_parallel",
        "model.should_pad_vocab",
        "model.softmax_scale",
        "model.tp_comm_atomic_ag",
        "model.tp_comm_atomic_rs",
        "model.tp_comm_overlap_disable_fc1",
        "model.tp_comm_overlap_disable_qkv",
        "model.tp_only_amax_red",
        "model.use_kitchen",
        "model.use_mamba_mem_eff_path",
        "model.use_transformer_engine_full_layer_spec",
        "model.use_transformer_engine_op_fuser",
        "model.variable_seq_lengths",
    }
)

# Extra direct-value mappings for Bridge YAML paths not in ARG_MAP
BRIDGE_TO_MLM_EXTRA: dict[str, str] = {
    "model.num_layers_in_first_pipeline_stage": "decoder-first-pipeline-num-layers",
    "model.num_layers_in_last_pipeline_stage": "decoder-last-pipeline-num-layers",
    "model.microbatch_group_size_per_vp_stage": "microbatch-group-size-per-virtual-pipeline-stage",
    "model.seq_len_interpolation_factor": "rotary-seq-len-interpolation-factor",
    "model.init_method.std": "init-method-std",
    "training.micro_batch_size": "micro-batch-size",
    "training.global_batch_size": "global-batch-size",
    "training.sequence_parallel": "sequence-parallel",
    "training.recompute_modules": "recompute-modules",
    "training.recompute_granularity": "recompute-granularity",
    "training.cuda_graph_scope": "cuda-graph-scope",
    "training.cuda_graph_warmup_steps": "cuda-graph-warmup-steps",
    "logging.tensorboard_dir": "tensorboard-dir",
    "checkpoint.save_dir": "save",
    "checkpoint.save_interval": "save-interval",
    "checkpoint.non_persistent_algo": "non-persistent-local-ckpt-algo",
    "checkpoint.non_persistent_type": "non-persistent-ckpt-type",
    "checkpoint.local_dir": "non-persistent-local-ckpt-dir",
    "checkpoint.global_dir": "non-persistent-global-ckpt-dir",
    "tokenizer.model": "tokenizer-model",
    "tokenizer.type": "tokenizer-type",
    "tokenizer.vocab_size": "vocab-size",
    "tokenizer.vocab_extra_ids": "vocab-extra-ids",
    "tokenizer.tiktoken_num_special_tokens": "tiktoken-num-special-tokens",
    "data.split": "split",
    "data.path": "data-path",
}


# ---------------------------------------------------------------------------
#  MLM argparse introspection
# ---------------------------------------------------------------------------


def _build_mlm_arg_index() -> dict[str, Any]:
    """Build an index of MLM argparse actions keyed by flag name (without --)."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_megatron_arguments(p)
    return {arg.option_strings[0][2:]: arg for arg in p._actions if arg.option_strings}


try:
    from megatron.training.arguments import add_megatron_arguments

    _MLM_ARG_INDEX: dict[str, Any] | None = _build_mlm_arg_index()
except ImportError:
    _MLM_ARG_INDEX = None


def _to_flag_and_value(name: str, val: Any) -> str:
    """Format an MLM arg as a CLI snippet using argparse introspection.

    Returns ``"--flag value"`` for value args, ``"--flag"`` for store_true,
    or ``""`` when the arg should not be emitted.
    """
    if _MLM_ARG_INDEX is None:
        if isinstance(val, bool):
            return f"--{name}" if val else ""
        if val is None:
            return ""
        if isinstance(val, (list, tuple)):
            return f"--{name} {' '.join(map(str, val))}" if val else ""
        return f"--{name} {val}"

    act = _MLM_ARG_INDEX.get(name)
    if act is None:
        if isinstance(val, bool):
            return f"--{name}" if val else ""
        if val is None:
            return ""
        return f"--{name} {val}"

    # store_true detection
    if getattr(act, "nargs", None) is None and act.option_strings and act.type is None and act.const is True:
        return f"--{name}" if bool(val) else ""

    if isinstance(val, (list, tuple)):
        if not val:
            return ""
        return f"--{name} {' '.join(map(str, val))}"

    if isinstance(val, bool):
        return f"--{name}" if val else ""
    if val is None:
        return ""
    if act.type is not None:
        return f"--{name} {act.type(val)}"
    return f"--{name} {val}"


# ---------------------------------------------------------------------------
#  Bridge override parsing
# ---------------------------------------------------------------------------


def parse_bridge_overrides(overrides_str: str) -> dict[str, Any]:
    """Parse a Bridge override string into a flat dict.

    Input format: ``"section.field=value section2.field2=value2"``
    """
    parsed: dict[str, Any] = {}
    for token in shlex.split(overrides_str):
        if "=" not in token:
            continue
        key, _, raw_val = token.partition("=")
        parsed[key] = _try_parse_value(raw_val)
    return parsed


# ---------------------------------------------------------------------------
#  Recipe loading (Bridge → flat dict)
# ---------------------------------------------------------------------------


def _load_recipe_to_flat_dict(recipe_name: str) -> dict[str, Any]:
    """Load a Bridge recipe by name and flatten it to a ``section.field`` dict.

    Requires ``megatron.bridge`` to be importable.

    Raises:
        ImportError: If megatron.bridge is not installed.
        ValueError: If the recipe is not found.
    """
    import dataclasses

    import megatron.bridge.recipes as recipes

    try:
        from megatron.bridge.utils.activation_map import callable_to_str as _act_to_str
    except ImportError:
        _act_to_str = None

    if not hasattr(recipes, recipe_name):
        available = [n for n in dir(recipes) if "config" in n]
        raise ValueError(
            f"Recipe '{recipe_name}' not found in megatron.bridge.recipes.\nAvailable (sample): {available[:20]}"
        )

    cfg = getattr(recipes, recipe_name)()

    def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                if f.name.startswith("_"):
                    continue
                val = getattr(obj, f.name)
                key = f"{prefix}.{f.name}" if prefix else f.name
                if dataclasses.is_dataclass(val) and not isinstance(val, type):
                    out.update(_flatten(val, key))
                elif callable(val) and not isinstance(val, type):
                    if _act_to_str is not None:
                        try:
                            out[key] = _act_to_str(val)
                        except (ValueError, KeyError):
                            pass
                elif isinstance(val, dict):
                    for dk, dv in val.items():
                        out[f"{key}.{dk}"] = dv
                else:
                    out[key] = val
        else:
            if prefix:
                out[prefix] = obj
        return out

    return _flatten(cfg)


# ---------------------------------------------------------------------------
#  Reverse translation engine (Bridge → MLM)
# ---------------------------------------------------------------------------


class ReverseTranslationResult:
    """Container for Bridge→MLM translation output."""

    def __init__(self):
        self.mlm_args: OrderedDict[str, Any] = OrderedDict()
        self.skipped: list[tuple[str, Any]] = []
        self.unknown: list[tuple[str, Any]] = []
        self.notes: list[str] = []

    def add_arg(self, mlm_name: str, value: Any = None):
        """Add an MLM arg. ``value=None`` means a boolean flag (no value)."""
        self.mlm_args[mlm_name] = value

    def add_note(self, note: str):
        self.notes.append(note)


def translate_bridge_to_mlm(overrides: dict[str, Any]) -> ReverseTranslationResult:
    """Translate Bridge overrides into MLM pretrain_gpt.py CLI args."""
    result = ReverseTranslationResult()
    consumed: set[str] = set()

    # --- Special: activation_func + gated_linear_unit → --swiglu ---------
    act_func = overrides.get("model.activation_func")
    gated = overrides.get("model.gated_linear_unit", False)
    if act_func is not None:
        consumed.add("model.activation_func")
        act_str = str(act_func)
        is_silu = any(s in act_str for s in ("silu",))
        is_squared_relu = any(s in act_str for s in ("squared_relu", "relu2"))

        if is_silu and gated:
            result.add_arg("swiglu")
            consumed.add("model.gated_linear_unit")
            result.add_note("swiglu: model.activation_func=silu + model.gated_linear_unit=true → --swiglu")
        elif is_squared_relu:
            result.add_arg("squared-relu")
            result.add_note("squared_relu: model.activation_func=squared_relu → --squared-relu")
        elif "gelu" not in act_str:
            result.add_note(f"activation_func={act_func}: no direct MLM flag, may need manual handling")

    if "model.gated_linear_unit" in overrides and "model.gated_linear_unit" not in consumed:
        consumed.add("model.gated_linear_unit")

    # --- Special: mixed_precision → --bf16 / --fp16 ----------------------
    mp = overrides.get("mixed_precision")
    if mp is not None:
        consumed.add("mixed_precision")
        if mp == "bf16_mixed":
            result.add_arg("bf16")
        elif mp in ("16-mixed", "fp16_mixed"):
            result.add_arg("fp16")
        else:
            result.add_note(f"mixed_precision={mp}: unknown precision mode")

    # --- Special: dataset.blend → --data-path ----------------------------
    blend = overrides.get("dataset.blend")
    if blend is not None:
        consumed.add("dataset.blend")
        if isinstance(blend, (list, tuple)) and len(blend) == 2:
            paths = blend[0] if isinstance(blend[0], (list, tuple)) else [blend[0]]
            result.add_arg("data-path", " ".join(str(p) for p in paths))
        elif isinstance(blend, str):
            # May arrive as "[[/path],null]" from CLI — extract paths
            s = blend.strip().strip('"').strip("'")
            if s.startswith("[[") and "]" in s:
                inner = re.findall(r"\[([^\[\]]+)\]", s)
                if inner:
                    paths = [p.strip() for p in inner[0].split(",") if p.strip()]
                    result.add_arg("data-path", " ".join(paths))
                else:
                    result.add_arg("data-path", s)
            else:
                result.add_arg("data-path", s)

    # --- Special: dataset.split → --split --------------------------------
    split = overrides.get("dataset.split")
    if split is not None:
        consumed.add("dataset.split")
        if isinstance(split, (list, tuple)):
            result.add_arg("split", ",".join(str(x) for x in split))
        else:
            result.add_arg("split", str(split))

    # --- Main loop -------------------------------------------------------
    for bridge_key, val in overrides.items():
        if bridge_key in consumed:
            continue

        # 1. Check auto-generated reverse of ARG_MAP
        if bridge_key in _REVERSE_ARG_MAP:
            mlm_name, rev_transform = _REVERSE_ARG_MAP[bridge_key]
            if rev_transform == "flag":
                if val:
                    result.add_arg(mlm_name)
            elif rev_transform == "flag_invert":
                if not val:
                    result.add_arg(mlm_name)
            else:
                result.add_arg(mlm_name, val)
            continue

        # 2. Check extra direct-value map
        if bridge_key in BRIDGE_TO_MLM_EXTRA:
            mlm_name = BRIDGE_TO_MLM_EXTRA[bridge_key]
            result.add_arg(mlm_name, val)
            continue

        # 3. Check extra boolean maps
        if bridge_key in BRIDGE_BOOL_TO_MLM:
            if val:
                result.add_arg(BRIDGE_BOOL_TO_MLM[bridge_key])
            continue

        if bridge_key in BRIDGE_INVERSE_BOOL_TO_MLM:
            if not val:
                result.add_arg(BRIDGE_INVERSE_BOOL_TO_MLM[bridge_key])
            continue

        # 4. Check ignore list
        if bridge_key in BRIDGE_IGNORE_KEYS:
            result.skipped.append((bridge_key, val))
            continue
        if any(bridge_key.startswith(pfx + ".") for pfx in BRIDGE_IGNORE_KEYS):
            result.skipped.append((bridge_key, val))
            continue
        if bridge_key.endswith("._target_") or bridge_key.startswith("model.generation_config"):
            result.skipped.append((bridge_key, val))
            continue

        # 5. Heuristic: try field_name with underscores → hyphens
        field_name = bridge_key.rsplit(".", 1)[-1]
        mlm_guess = field_name.replace("_", "-")
        if _MLM_ARG_INDEX and mlm_guess in _MLM_ARG_INDEX:
            frag = _to_flag_and_value(mlm_guess, val)
            if frag:
                result.add_arg(mlm_guess, val if not isinstance(val, bool) else None)
            continue

        # 6. Unknown
        result.unknown.append((bridge_key, val))

    return result


# ---------------------------------------------------------------------------
#  MLM output emitters
# ---------------------------------------------------------------------------


def emit_mlm_args(result: ReverseTranslationResult) -> str:
    """Emit MLM CLI args, one per line."""
    lines = []
    lines.append("# ═══════════════════════════════════════════════════════════")
    lines.append("# Megatron-LM pretrain_gpt.py args (translated from Bridge)")
    lines.append("# ═══════════════════════════════════════════════════════════")

    if result.notes:
        lines.append("#")
        for note in result.notes:
            lines.append(f"# NOTE: {note}")
        lines.append("#")

    for mlm_name, val in result.mlm_args.items():
        frag = _to_flag_and_value(mlm_name, True if val is None else val)
        if frag:
            lines.append(f"  {frag} \\")

    if result.unknown:
        lines.append("\n# ── Unknown Bridge keys (not mapped) ─────────────────────")
        for key, val in result.unknown:
            lines.append(f"#   {key}={val}")

    if result.skipped:
        lines.append(f"\n# ── Skipped: {len(result.skipped)} Bridge-only keys (no MLM equivalent)")

    return "\n".join(lines)


def emit_mlm_command(
    result: ReverseTranslationResult,
    script_path: str = "pretrain_gpt.py",
    nproc: int = 8,
) -> str:
    """Emit full torchrun + pretrain_gpt.py launch command."""
    lines = []
    lines.append(f"torchrun --nproc_per_node={nproc} {script_path} \\")

    for mlm_name, val in result.mlm_args.items():
        frag = _to_flag_and_value(mlm_name, True if val is None else val)
        if frag:
            lines.append(f"  {frag} \\")

    # Remove trailing backslash from last line
    if lines and lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for bidirectional MLM <-> Bridge config translation."""
    parser = argparse.ArgumentParser(
        description="Translate between Megatron-LM args and Megatron Bridge overrides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples (MLM → Bridge, default):
          python scripts/translate_mlm_to_bridge.py --yaml DeepSeek-V3.yaml
          python scripts/translate_mlm_to_bridge.py --args "--num-layers 32 --hidden-size 4096 --bf16 --swiglu"
          python scripts/translate_mlm_to_bridge.py --yaml DeepSeek-V3.yaml --emit recipe --recipe-name deepseek_v3

        Examples (Bridge → MLM, --reverse):
          python scripts/translate_mlm_to_bridge.py --reverse --recipe llama32_1b_pretrain_config
          python scripts/translate_mlm_to_bridge.py --reverse --recipe llama32_1b_pretrain_config \\
              --args "train.train_iters=1000 model.tensor_model_parallel_size=2"
          python scripts/translate_mlm_to_bridge.py --reverse --emit command --recipe llama32_1b_pretrain_config
          python scripts/translate_mlm_to_bridge.py --reverse --args "model.num_layers=32 model.hidden_size=4096"
        """),
    )
    parser.add_argument("--yaml", type=str, help="Path to Megatron-LM YAML config (MLM→Bridge)")
    parser.add_argument("--args", type=str, help="Raw args string (MLM flags or Bridge overrides)")
    parser.add_argument(
        "--recipe",
        type=str,
        help="Bridge recipe name for --reverse (e.g., llama32_1b_pretrain_config). "
        "Loads recipe defaults; combine with --args to add overrides on top.",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse direction: translate Bridge overrides → MLM args",
    )
    parser.add_argument(
        "--emit",
        choices=["overrides", "recipe", "command"],
        default="overrides",
        help="Output format (default: overrides). "
        "For --reverse: 'overrides' emits MLM args, 'command' emits full torchrun command",
    )
    parser.add_argument("--recipe-name", type=str, default="custom_model", help="Recipe name for --emit recipe")
    parser.add_argument(
        "--base-recipe",
        type=str,
        default="vanilla_gpt_pretrain_config",
        help="Base recipe name for --emit command (MLM→Bridge direction)",
    )
    parser.add_argument(
        "--script-path",
        type=str,
        default="pretrain_gpt.py",
        help="Path to pretrain_gpt.py for --reverse --emit command",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="nproc_per_node for --emit command",
    )
    parser.add_argument("--output", "-o", type=str, help="Write output to file instead of stdout")

    cli_args = parser.parse_args()

    # ── Input validation ──
    if cli_args.reverse:
        if not cli_args.recipe and not cli_args.args and not cli_args.yaml:
            parser.error("--reverse requires at least one of: --recipe, --args, --yaml")
    else:
        if not cli_args.yaml and not cli_args.args:
            parser.error("MLM→Bridge direction requires --yaml or --args")
        if cli_args.recipe:
            parser.error("--recipe is only used with --reverse")

    if cli_args.reverse:
        # ── Bridge → MLM direction ──
        bridge_overrides: dict[str, Any] = {}

        # 1) Load recipe defaults as base
        if cli_args.recipe:
            try:
                bridge_overrides = _load_recipe_to_flat_dict(cli_args.recipe)
                print(
                    f"# Loaded recipe '{cli_args.recipe}': {len(bridge_overrides)} fields",
                    file=sys.stderr,
                )
            except ImportError:
                print(
                    "ERROR: megatron.bridge must be importable to use --recipe. "
                    "Run from a Bridge environment or use --args instead.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # 2) Merge CLI overrides on top
        if cli_args.yaml:
            if not HAS_YAML:
                print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
                sys.exit(1)
            with open(cli_args.yaml) as f:
                raw = yaml.safe_load(f)
            bridge_overrides.update(_flatten_dict(raw))
        elif cli_args.args:
            bridge_overrides.update(parse_bridge_overrides(cli_args.args))

        result = translate_bridge_to_mlm(bridge_overrides)

        if cli_args.emit == "command":
            output = emit_mlm_command(result, script_path=cli_args.script_path, nproc=cli_args.nproc)
        else:
            output = emit_mlm_args(result)

        _write_output(output, cli_args.output)

        # Summary
        print("\n# Summary:", file=sys.stderr)
        if cli_args.recipe:
            print(f"#   Recipe:     {cli_args.recipe}", file=sys.stderr)
        print(f"#   Translated: {len(result.mlm_args)} MLM args", file=sys.stderr)
        if result.skipped:
            print(f"#   Skipped:    {len(result.skipped)} (Bridge-only, no MLM equivalent)", file=sys.stderr)
        if result.unknown:
            print(f"#   Unknown:    {len(result.unknown)} (no mapping found)", file=sys.stderr)
            for key, val in result.unknown:
                print(f"#     {key}={val}", file=sys.stderr)
        if result.notes:
            for note in result.notes:
                print(f"#   Note: {note}", file=sys.stderr)
    else:
        # ── MLM → Bridge direction (existing) ──
        if cli_args.yaml:
            parsed_args, env_vars = parse_yaml_config(cli_args.yaml)
        else:
            parsed_args, env_vars = parse_raw_args(cli_args.args)

        result = translate(parsed_args, env_vars)

        if cli_args.emit == "overrides":
            output = emit_overrides(result)
        elif cli_args.emit == "command":
            output = emit_command(result, base_recipe=cli_args.base_recipe)
        elif cli_args.emit == "recipe":
            output = emit_recipe(result, recipe_name=cli_args.recipe_name)
        else:
            output = emit_overrides(result)

        _write_output(output, cli_args.output)

        # Summary
        print("\n# Summary:", file=sys.stderr)
        print(f"#   Translated: {len(result.overrides)} overrides", file=sys.stderr)
        if result.skipped:
            print(f"#   Skipped:    {len(result.skipped)} (not needed in Bridge)", file=sys.stderr)
        if result.unknown:
            print(f"#   Unknown:    {len(result.unknown)} (no mapping found)", file=sys.stderr)
            for arg, val in result.unknown:
                print(f"#     --{arg} = {val}", file=sys.stderr)
        if result.notes:
            for note in result.notes:
                print(f"#   Note: {note}", file=sys.stderr)


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _write_output(output: str, output_path: str | None) -> None:
    """Write output to file or stdout."""
    if output_path:
        Path(output_path).write_text(output + "\n")
        print(f"Written to {output_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
