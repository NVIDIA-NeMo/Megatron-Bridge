# mbridge_to_megatron.py
# Converts an MBridge-style YAML config to a Megatron-LM bash launcher.
# Works with your provided add_megatron_arguments() (2025 NVIDIA file).
#
# Usage:
#   python mbridge_to_megatron.py run_config.yaml out.sh \
#       --script-path /opt/megatron-lm/pretrain_gpt.py \
#       --save-dir /ckpts/qwen3_8b \
#       --local-ckpt-dir /localssd/np_ckpts \
#       --data-prefix /data/corpus_prefix \
#       --use-replication
#
# Notes:
# - This script doesn’t guess your hardware parallelism. Adjust TP/PP/DP after generation if needed.
# - It auto-fixes inverse flags and list args; auto-adds missing required args.

import sys, os, argparse, yaml, re
from collections import defaultdict
from typing import Any, Dict, Tuple

# Make sure we can import Megatron-LM args
sys.path.insert(0, "/opt/megatron-lm")

from megatron.training.arguments import add_megatron_arguments  # noqa: E402

'''
def process_act_func(arg, val, mbridge_config):
    if val == 'torch.nn.functional.silu':
        if mbridge_config["model.gated_linear_unit"]:
            if mbridge_config["model.bias_activation_fusion"]:
                return f"swiglu"
            else:
                return "swiglu\n\t--no-bias-swiglu-fusion"
    elif val == 'torch.nn.functional.gelu':
        # I believe gelu is default
        if not mbridge_config["model.bias_activation_fusion"]:
            return "no-bias-gelu-fusion"
    else:
        raise NotImplementedError
'''


def process_act_func(arg, val, mbridge_config):
    """
    Map MBridge activation target to Megatron flags.
    - Never raise: return [] on unknowns.
    - Use `.get()` to avoid KeyErrors if YAML omits fields.+    - Return a *list of flags* so each flag becomes its own array element.
    """
    s = str(val) if val is not None else ""
    gated = bool(mbridge_config.get("model.gated_linear_unit", False))
    bias_fusion = bool(mbridge_config.get("model.bias_activation_fusion", True))

    def has(name: str) -> bool:
        return name in s or s.endswith(name)

    # Gated SiLU ▒~F~R SWiGLU (optionally disable bias fusion)
    if has("silu"):
        if gated:
            out = ["swiglu"]
            if not bias_fusion:
                out.append("no-bias-swiglu-fusion")
            return out
        # Plain SiLU: do nothing (assume handled elsewhere / default)
        return []

    # GELU: default is gelu; only emit fusion disable if requested
    if has("gelu"):
        return [] if bias_fusion else ["no-bias-gelu-fusion"]

    # Other/unknown activation: no-op
    return


def query_groups_sideeffect(arg, val, mbridge_config):
    """Set --group-query-attention if num_query_groups != num_attention_heads."""
    num_heads = mbridge_config.get("model.num_attention_heads", None)
    if num_heads and isinstance(num_heads, int) and num_heads != val:
        return f"group-query-attention\n\t--num-query-groups {val}"

# ---------- Maps for args that don't match 1:1 ----------
EXTRA_ARG_MAP = {
    # pipeline split helpers
    "model.num_layers_in_first_pipeline_stage": "decoder-first-pipeline-num-layers",
    "model.num_layers_in_last_pipeline_stage": "decoder-last-pipeline-num-layers",
    "model.num_moe_experts": "num-experts",
    "model.virtual_pipeline_model_parallel_size": "num-virtual-stages-per-pipeline-rank",
    "model.layernorm_epsilon": "norm-epsilon",
    "model.seq_len_interpolation_factor": "rotary-seq-len-interpolation-factor",
    "model.init_method.std": "init-method-std",
    "model.microbatch_group_size_per_vp_stage": "microbatch-group-size-per-virtual-pipeline-stage",
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

# Boolean flags that map directly to Megatron store_true flags
EXTRA_ARG_BOOL_MAP = {
    "model.use_te_rng_tracker": "te-rng-tracker",
    "model.layernorm_zero_centered_gamma": "apply-layernorm-1p",
    "training.sequence_parallel": "sequence-parallel",
    "training.gradient_accumulation_fusion": "gradient-accumulation-fusion",
    "training.cross_entropy_loss_fusion": "cross-entropy-loss-fusion",
    "training.masked_softmax_fusion": "masked-softmax-fusion",
    "training.bias_dropout_fusion": "bias-dropout-fusion",
    "training.tp_comm_overlap": "tp-comm-overlap",
    "checkpoint.use_persistent_ckpt_worker": "use-persistent-ckpt-worker",
    "logging.log_progress": "log-progress",
    "checkpoint.fully_parallel_load": "ckpt-fully-parallel-load",
    "comm_overlap": "tp-comm-overlap",
    "model.fp8_param": "fp8-param-gather",
    "model.overlap_p2p_comm_warmup_flush": "overlap-p2p-communication-warmup-flush",
}

# Inverse boolean flags (MBridge True ⇒ Megatron uses a disabling flag or untie)
EXTRA_ARG_INVERSE_BOOL_MAP = {
    # TP overlap “split”/“rs/ag” have disable-* in Megatron. Leave unset to enable.
    "training.tp_comm_overlap_split_ag": "disable-tp-comm-split-ag",
    "training.tp_comm_overlap_split_rs": "disable-tp-comm-split-rs",
    "training.tp_comm_overlap_rs_dgrad": "disable-tp-comm-bulk-dgrad",  # if False, we emit disable
    "training.tp_comm_overlap_bulk_wgrad": "disable-tp-comm-bulk-wgrad",
    "model.tp_comm_overlap_rs": "disable-tp-comm-overlap-rs",
    # ckpt parallel save inverse
    "checkpoint.fully_parallel_save": "no-ckpt-fully-parallel-save",
    # embedding tie
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

# Flags that can be safely ignored
TO_IGNORE_KEYS = {
    # Prefixes/Objects for building Megatron Bridge subconfigs
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
    "rng._target_",
    "rerun_state_machine._target_",
    "train._target_",
    "model._target_",
    "optimizer._target_",
    "ddp._target_",
    "scheduler._target_",
    "dataset._target_",
    "logger._target_",
    "tokenizer._target_",
    "checkpoint._target_",
    "dist._target_",
    "ft._target_",
    "straggler._target_",
    "nvrx_straggler._target_",
    "profiling._target_",
    "peft._target_",
    "comm_overlap._target_",
    "mixed_precision._target_",
    "inprocess_restart._target_",
    # MBridge unique logging settings
    "logger.filter_warnings",
    "logger.modules_to_filter",
    "logger.set_level_for_all_loggers",
    # Mbridge Multimodal tokenizer settings
    "tokenizer.image_tag_type",
    "tokenizer.special_tokens",
    "tokenizer.tokenizer_prompt_format",
    # built at runtime
    "model.timers",
    "optimizer.timers",
    "model.finalize_model_grads_func",
    "model.grad_scale_func",
    "model.grad_sync_func",
    "model.no_sync_func",
    "model.param_sync_func",
    # No MLM args for model weight/activation cpu offloading. args exist for optim offload
    "model.cpu_offloading",
    "model.cpu_offloading_activations",
    "model.cpu_offloading_double_buffering",
    "model.cpu_offloading_num_layers",
    "model.cpu_offloading_weights",
    # No MLM arg exists
    "optimizer.store_param_remainders",
    "model.activation_func_fp8_input_store",
    "model.batch_p2p_sync",
    "model.cuda_graph_retain_backward_graph",
    "model.cuda_graph_use_single_mempool",
    "model.deallocate_pipeline_outputs",  # hardcoded to true in arguments.py
    "model.disable_parameter_transpose_cache",
    "model.enable_autocast",
    "model.fp8_dot_product_attention",
    "model.fp8_multi_head_attention",
    "model.hetereogenous_dist_checkpoint",
    "model.heterogeneous_block_specs",  # automatically set on TransformerConfig based on other heterogeneous args
    "model.memory_efficient_layer_norm",
    "model.moe_router_topk_limited_devices",
    "model.moe_token_dropping",  # unused
    "model.mtp_enabled",  # unused, from mbridge
    "model.num_microbatches_with_partial_activation_checkpoints",
    "model.output_layer_init_method.mean",  # hardcoded to 0.0
    "model.output_layer_init_method.std",  # hardcoded based on init_method_std
    "model.parallel_output",  # hardcoded to true in gpt_builders.py
    "model.scatter_embedding_sequence_parallel",
    "model.should_pad_vocab",  # mbridge specific
    "model.softmax_scale",
    "model.tp_comm_atomic_ag",  # deprecated
    "model.tp_comm_atomic_rs",  # deprecated
    "model.tp_comm_overlap_disable_fc1",
    "model.tp_comm_overlap_disable_qkv",
    "model.tp_only_amax_red",
    "model.use_kitchen",  # will be set automatically if kitchen config is specified
    "model.use_mamba_mem_eff_path",
    "model.use_transformer_engine_full_layer_spec",  # mbridge specific
    "model.use_transformer_engine_op_fuser",  # mbridge specific
    "model.variable_seq_lengths",  # hardcoded false
}

EXTRA_ARG_MISC = {
    "model.activation_func._target_": process_act_func,
    "model.num_query_groups": query_groups_sideeffect,
}


def filter_safe_to_ignore(missing, mbridge_config):
    filtered = []
    for key, val in missing.items():
        is_gen_config = key.startswith("model.generation_config")
        can_ignore = any([key == ignore_key for ignore_key in TO_IGNORE_KEYS])
        if is_gen_config or can_ignore:
            print(f"Safe to ignore missing key {key} with value {val}")
            continue

        filtered.append((key, val))
    for key, value in filtered:
        print("Missing:", key, value)

    return filtered


# Known groups -> bash array name
GROUP_ORDER = [
    "checkpoint",
    "logging",
    "training",
    "model",
    "optimizer",
    "rng",
    "tokenizer",
    "data",
    "rerun_state_machine",
    "other",
]


# Helper: flatten nested dict using dotted keys
def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def get_megatron_arg_index():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_megatron_arguments(p)
    mlm_arg_name_to_arg = {arg.option_strings[0][2:]: arg for arg in p._actions}
    return mlm_arg_name_to_arg


MEG_ARGS = get_megatron_arg_index()


def to_flag_and_value(name: str, val: Any) -> str:
    """
    Build a CLI snippet: "--flag value" or just "--flag" for store_true.
    We assume `name` is already the Megatron CLI form with hyphens.
    """
    # Store_true (presence only). We detect by checking action type.
    act = MEG_ARGS.get(name)
    if act is None:
        # Allow unknowns to go out as "--name value"
        if isinstance(val, bool):
            return f"--{name}" if val else ""
        if val is None:
            return ""
        return f"--{name} {val}"

    # If flag expects no value (store_true / store_false style):
    if (
        getattr(act, "nargs", None) is None
        and act.option_strings
        and act.type is None
        and act.const is True
    ):
        # Common case: `action='store_true'` → supply only when True.
        return f"--{name}" if bool(val) else ""

    # Lists: Megatron takes repeated nargs or space-separated after flag for many.
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return ""
        # Flatten to space-separated after single flag
        items = " ".join(map(str, val))
        return f"--{name} {items}"

    # Booleans for non-store_true args are rare; pass as true/false literal
    if isinstance(val, bool):
        return f"--{name}" if val else ""

    if val is None:
        return ""
    if act.type is not None:
        return f"--{name} {act.type(val)}"
    return f"--{name} {val}"


def normalize_listish(v: Any) -> Any:
    # Turn "['core_attn']" or "core_attn" or ["core_attn"] into ["core_attn"]
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            try:
                parsed = eval(s, {"__builtins__": {}}, {})
                return (
                    list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
                )
            except Exception:
                return [s]
        # split by comma / whitespace if needed
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return [v]


def build_matches(cfg: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    matches = {}
    missing = {}

    flat = flatten(cfg)
    for fullkey, val in flat.items():
        # Special value cleanup for recompute-modules etc.
        if fullkey.endswith("recompute_modules"):
            val = normalize_listish(val)
        if fullkey.endswith("cp_comm_type"):
            val = normalize_listish(val)

        # Map via explicit maps first
        if fullkey in EXTRA_ARG_MAP:
            meg_key = EXTRA_ARG_MAP[fullkey]
            frag = to_flag_and_value(meg_key, val)
            if frag:
                matches[fullkey] = frag
            continue

        if fullkey in EXTRA_ARG_BOOL_MAP:
            if bool(val):
                matches[fullkey] = f"--{EXTRA_ARG_BOOL_MAP[fullkey]}"
            continue

        if fullkey in EXTRA_ARG_INVERSE_BOOL_MAP:
            # Emit inverse flag only when value is False (or when "untie" semantics)
            inv_flag = EXTRA_ARG_INVERSE_BOOL_MAP[fullkey]
            # share_embeddings True → do NOT emit untie flag; False → emit untie flag
            if "untie-embeddings" in inv_flag:
                if not bool(val):
                    matches[fullkey] = f"--{inv_flag}"
            else:
                if not bool(val):
                    matches[fullkey] = f"--{inv_flag}"
            continue

        if fullkey in EXTRA_ARG_MISC:
            mapping = EXTRA_ARG_MISC[fullkey](fullkey, val, flat)
            if mapping:
                matches[fullkey] = f"--{mapping}"
                continue

        # Heuristic: try exact Megatron dest name with hyphens
        meg_key_guess = fullkey.split(".")[-1].replace("_", "-")
        if meg_key_guess in MEG_ARGS:
            # Handle lists
            if isinstance(val, list):
                frag = to_flag_and_value(meg_key_guess, val)
            elif isinstance(val, bool):
                frag = to_flag_and_value(meg_key_guess, val)
            else:
                frag = to_flag_and_value(meg_key_guess, val)
            if frag:
                matches[fullkey] = frag
        else:
            # remember for post-processing
            missing[fullkey] = val

    return matches, missing


def remove_duplicate_args(matches: Dict[str, str]):
    existing = set()
    filtered_matches = {}
    for key, val in matches.items():
        if val in existing:
            continue
        existing.add(val)
        filtered_matches[key] = val
    return filtered_matches


def delete_by_regex(dictionary: Dict, pattern: str):
    regex = re.compile(pattern)
    for key in dictionary.keys():
        if regex.search(key):
            del dictionary[key]
            return


def ad_hoc_filters(matches: Dict[str, str]) -> Dict[str, str]:
    reverse_mapping = {val: key for key, val in matches.items()}
    if "--config-logger-dir " in reverse_mapping:  # Empty path is not supported
        del reverse_mapping["--config-logger-dir "]
    if "--weight-decay-incr-style constant" in reverse_mapping:
        delete_by_regex(reverse_mapping, "--end-weight-decay.+")
        delete_by_regex(reverse_mapping, "--start-weight-decay.+")
    delete_by_regex(reverse_mapping, "--local-rank.+")
    matches = {val: key for key, val in reverse_mapping.items()}
    return matches


def postprocess(
    matches: Dict[str, str], missing: Dict[str, Any], overrides
) -> Dict[str, str]:
    """
    Add required args and fix common gaps:
      - max-position-embeddings if seq-length given
      - CUDA graphs impl if scope set
      - replication + local non-persistent checkpointing
      - data path, save dir, tokenizer defaults (from CLI overrides)
    """

    # Pull simple presence helpers
    def has(flag_prefix: str) -> bool:
        return any(
            s.endswith(f"--{flag_prefix}") or f"--{flag_prefix} " in s
            for s in matches.values()
        )

    # Ensure seq-length → max-position-embeddings
    seq_len = None
    for k, v in matches.items():
        if v.startswith("--seq-length "):
            seq_len = int(v.split()[1])
            break
    if seq_len is not None and not has("max-position-embeddings"):
        matches["__auto.max_position_embeddings"] = (
            f"--max-position-embeddings {seq_len}"
        )

    # CUDA Graphs scope needs an impl
    #if has("cuda-graph-scope") and not has("cuda-graph-impl"):
    #    matches["__auto.cuda_graph_impl"] = "--cuda-graph-impl local"

    # Replication/non-persistent local ckpt
    if overrides.use_replication:
        matches["__auto.replication"] = "--replication"
        if not has("replication-factor"):
            matches["__auto.replication_factor"] = (
                f"--replication-factor {overrides.replication_factor}"
            )
        if not has("replication-jump"):
            matches["__auto.replication_jump"] = (
                f"--replication-jump {overrides.replication_jump}"
            )
        # force local non-persistent ckpt
        matches["__auto.np_type"] = "--non-persistent-ckpt-type local"
        if overrides.local_ckpt_dir:
            matches["__auto.np_dir"] = (
                f"--non-persistent-local-ckpt-dir {overrides.local_ckpt_dir}"
            )
        if not has("non-persistent-local-ckpt-algo"):
            matches["__auto.np_algo"] = (
                "--non-persistent-local-ckpt-algo fully_parallel"
            )

    # save/checkpoint dir override
    if overrides.save_dir and not any(
        v.startswith("--save ") for v in matches.values()
    ):
        matches["__auto.save"] = f"--save {overrides.save_dir}"
        if not has("save-interval"):
            matches["__auto.save_interval"] = (
                f"--save-interval {overrides.save_interval}"
            )

    # tokenizer defaults
    if overrides.tokenizer_model and not any(
        v.startswith("--tokenizer-model ") for v in matches.values()
    ):
        matches["__auto.tok_model"] = f"--tokenizer-model {overrides.tokenizer_model}"
    if overrides.tokenizer_type and not any(
        v.startswith("--tokenizer-type ") for v in matches.values()
    ):
        matches["__auto.tok_type"] = f"--tokenizer-type {overrides.tokenizer_type}"

    # data source
    if overrides.data_prefix and not any(
        v.startswith("--data-path ") for v in matches.values()
    ):
        matches["__auto.data_path"] = f"--data-path {overrides.data_prefix}"
        if not has("split"):
            matches["__auto.split"] = f"--split {overrides.data_split}"

    # training baseline if missing
    if not has("micro-batch-size"):
        matches["__auto.mbs"] = f"--micro-batch-size {overrides.micro_batch_size}"
    if not any(v.startswith("--global-batch-size ") for v in matches.values()):
        matches["__auto.gbs"] = f"--global-batch-size {overrides.global_batch_size}"

    # Dtype defaults
    if overrides.use_bf16 and not has("bf16") and not has("fp16"):
        matches["__auto.bf16"] = "--bf16"

    # Remove duplicates
    matches = remove_duplicate_args(matches)

    # Additional filters
    matches = ad_hoc_filters(matches)

    return matches


def group_into_bash_arrays(matches: Dict[str, str]) -> Dict[str, list]:
    groups = defaultdict(list)
    # Group by top-level MBridge prefix if present
    for k in sorted(matches):
        cli = matches[k]
        if not cli:
            continue
        top = k.split(".", 1)[0] if "." in k else "other"
        # normalize names for arrays
        name = {
            "checkpoint": "CHECKPOINT_ARGS",
            "logging": "LOGGER_ARGS",
            "training": "TRAINING_ARGS",
            "model": "MODEL_ARGS",
            "optimizer": "OPTIMIZER_ARGS",
            "rng": "RNG_ARGS",
            "tokenizer": "TOKENIZER_ARGS",
            "data": "DATA_ARGS",
            "rerun_state_machine": "RERUN_STATE_MACHINE_ARGS",
            "other": "OTHER_ARGS",
        }.get(top.lower(), "OTHER_ARGS")
        groups[name].append(cli)
    return groups


def render_bash(groups: Dict[str, list], overrides) -> str:
    # Keep arrays in a predictable order
    ordered = []
    for key in [
        "CHECKPOINT_ARGS",
        "LOGGER_ARGS",
        "TRAINING_ARGS",
        "MODEL_ARGS",
        "OPTIMIZER_ARGS",
        "RNG_ARGS",
        "TOKENIZER_ARGS",
        "DATA_ARGS",
        "RERUN_STATE_MACHINE_ARGS",
        "OTHER_ARGS",
    ]:
        if key in groups:
            arr = "\n\t".join(groups[key])
            ordered.append(f"{key}=(\n\t{arr}\n)\n")
    # torchrun env
    ds = f"""#!/bin/bash

PRETRAIN_SCRIPT_PATH={overrides.script_path}

# torchrun launcher (edit if multi-node)
MASTER_ADDR="${{MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${{MASTER_PORT:-29500}}"
NNODES="${{NNODES:-1}}"
NODE_RANK="${{NODE_RANK:-0}}"
NPROC_PER_NODE="${{NPROC_PER_NODE:-{overrides.nproc_per_node}}}"

DISTRIBUTED_ARGS=(
  --nnodes "${{NNODES}}"
  --nproc-per-node "${{NPROC_PER_NODE}}"
  --node-rank "${{NODE_RANK}}"
  --master-addr "${{MASTER_ADDR}}"
  --master-port "${{MASTER_PORT}}"
)

DATA_ARGS=(
    "--mock-data"
    "--tokenizer-type NullTokenizer"
    "--vocab-size 128256" 
    "--tiktoken-pattern v2" 
    "--split '99,1,0'"
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    "--num-workers 1"
    )

{''.join(ordered)}
set -x
torchrun ${{DISTRIBUTED_ARGS[@]}} \\
\t"$PRETRAIN_SCRIPT_PATH" \\
\t${{CHECKPOINT_ARGS[@]}} \\
\t${{LOGGER_ARGS[@]}} \\
\t${{TRAINING_ARGS[@]}} \\
\t${{MODEL_ARGS[@]}} \\
\t${{OPTIMIZER_ARGS[@]}} \\
\t${{RNG_ARGS[@]}} \\
\t${{TOKENIZER_ARGS[@]}} \\
\t${{DATA_ARGS[@]}} \\
\t${{RERUN_STATE_MACHINE_ARGS[@]}} \\
\t${{OTHER_ARGS[@]}}
"""
    return ds


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("out_path")
    ap.add_argument("--script-path", default="/opt/megatron-lm/pretrain_gpt.py")
    ap.add_argument("--save-dir", default="qwen3_8b_megatron")
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--local-ckpt-dir", default="")
    ap.add_argument("--data-prefix", default="")
    ap.add_argument("--data-split", default="969,30,1")
    ap.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tokenizer-type", default="HuggingFaceTokenizer")
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--global-batch-size", type=int, default=64)
    ap.add_argument("--use-bf16", action="store_true", default=True)
    ap.add_argument("--use-replication", action="store_true")
    ap.add_argument("--replication-factor", type=int, default=2)
    ap.add_argument("--replication-jump", type=int, default=1)
    ap.add_argument("--nproc-per-node", type=int, default=8)
    return ap.parse_args()


def main():
    args = parse_cli()
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    matches, missing = build_matches(cfg)
    missing = filter_safe_to_ignore(missing, cfg)
    matches = postprocess(matches, missing, args)
    groups = group_into_bash_arrays(matches)
    bash = render_bash(groups, args)

    with open(args.out_path, "w") as f:
        f.write(bash)
    print(f"Wrote launcher: {args.out_path}")


if __name__ == "__main__":
    main()
