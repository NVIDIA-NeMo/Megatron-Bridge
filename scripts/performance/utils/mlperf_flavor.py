# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MLPerf v6.0 apples-to-apples flavor (--mlperf_flavor): resolves shape, dataset, mounts, and parity env vars.

Shapes here mirror the canonical NVIDIA MLPerf v6.0 training submission configs on GB200.
Validated end-to-end against MLPerf v6.0 reference (Llama 3.1 8B, 8/16/32/64 GPU FP8 + FP4).
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# (model_recipe_name, compute_dtype, num_gpus, gpu) -> (TP, PP, VP, CP, MBS, GBS, parity_mode)
# Shapes derived from MLPerf v6.0 NVIDIA submission configs (per-gpu source-of-truth differs):
#   - GB200 8B 8 GPU:   config_GB200_2x4x2xtp1pp1cp1_8b[_fp4].sh       (TP1/PP1/CP1, MBS=2, GBS=16)
#   - GB200 8B 72 GPU:  config_GB200_18x4x1xtp1pp1cp2_8b.sh            (TP1/PP1/CP2, MBS=1)
#   - GB200 8B 128 FP8: down-scaled config_GB200_128x4x1xtp2pp1cp4_8b.sh (TP2/PP1/CP4, MBS=1)
#   - GB200 405B:       config_GB200_128x4x{112,128}xtp4pp8cp2_cg_fp4.sh (TP4/PP8/VP8/CP2, MBS=1)
#   - GB300 405B:       config_GB300_128x4x56xtp2pp8cp2_cg_fp4.sh      (TP2/PP8/VP8/CP2, MBS=1)
# Note: 16/32/64/128 GPU entries for 8B are down-scaled from canonical 72/512 GPU shapes (same TP/PP/CP).
_MLPERF_SHAPES: Dict[Tuple[str, str, int, str], Tuple[int, int, Optional[int], int, int, int, str]] = {
    # 8B PP=1 -> VP=None (Megatron rejects VP>0 with PP=1; matches INTERLEAVED_PIPELINE=null in optimized.git).
    ("llama3_8b",   "fp8_cs",   8,   "gb200"): (1, 1, None, 1, 2, 16,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   16,  "gb200"): (1, 1, None, 2, 1, 8,    "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   32,  "gb200"): (1, 1, None, 2, 1, 16,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   64,  "gb200"): (1, 1, None, 2, 1, 32,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   72,  "gb200"): (1, 1, None, 2, 1, 36,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   128, "gb200"): (2, 1, None, 4, 1, 16,   "F16_ATTN"),
    ("llama3_8b",   "nvfp4",    8,   "gb200"): (1, 1, None, 1, 2, 16,   "FP4_ATTN"),
    ("llama3_8b",   "nvfp4",    16,  "gb200"): (1, 1, None, 1, 1, 16,   "FP4_ATTN"),
    ("llama3_8b",   "nvfp4",    32,  "gb200"): (1, 1, None, 1, 1, 32,   "FP4_ATTN"),
    ("llama3_8b",   "nvfp4",    64,  "gb200"): (1, 1, None, 1, 1, 64,   "FP4_ATTN"),
    # 405B entries deferred: GB200 + GB300 not yet validated against MLPerf-canonical GBS=896 shapes.
    # 8B FP8 GB300 — same shapes as GB200 (canonical optimized.git configs use identical TP/PP/CP/MBS).
    ("llama3_8b",   "fp8_cs",   8,   "gb300"): (1, 1, None, 1, 2, 16,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   16,  "gb300"): (1, 1, None, 2, 1, 8,    "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   32,  "gb300"): (1, 1, None, 2, 1, 16,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   64,  "gb300"): (1, 1, None, 2, 1, 32,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   72,  "gb300"): (1, 1, None, 2, 1, 36,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   128, "gb300"): (2, 1, None, 4, 1, 16,   "F16_ATTN"),
    # 8B FP4 GB300 — same shapes as GB200.
    ("llama3_8b",   "nvfp4",    8,   "gb300"): (1, 1, None, 1, 2, 16,   "FP4_ATTN"),
    ("llama3_8b",   "nvfp4",    16,  "gb300"): (1, 1, None, 1, 1, 16,   "FP4_ATTN"),
    ("llama3_8b",   "nvfp4",    32,  "gb300"): (1, 1, None, 1, 1, 32,   "FP4_ATTN"),
    ("llama3_8b",   "nvfp4",    64,  "gb300"): (1, 1, None, 1, 1, 64,   "FP4_ATTN"),
    ("llama3_8b",   "nvfp4",    72,  "gb300"): (1, 1, None, 1, 1, 72,   "FP4_ATTN"),
}


def _resolve_dataset(model_recipe_name: str) -> Tuple[str, str]:
    """Return (data_prefix, index_cache_dir) for the model size.

    Expects MLPERF_DATA_ROOT to contain <size>/c4-train.en_6_text_document.{bin,idx}.
    On clusters with the shared MLPerf training C4 preproc, set:
        MLPERF_DATA_ROOT=/lustre/share/coreai_mlperf_training/data/c4
    (same layout that MLPerf v6.0 reference configs expect).
    """
    data_root = os.environ.get("MLPERF_DATA_ROOT")
    if not data_root:
        raise RuntimeError(
            "--mlperf_flavor requires MLPERF_DATA_ROOT pointing to a dir containing "
            "<size>/c4-train.en_6_text_document.{bin,idx}."
        )

    size = "8b" if "8b" in model_recipe_name else "405b" if "405b" in model_recipe_name else None
    if size is None:
        raise RuntimeError(f"--mlperf_flavor: cannot map model_recipe_name '{model_recipe_name}' to a dataset size.")

    prefix = os.path.join(data_root, size, "c4-train.en_6_text_document")
    if not (os.path.isfile(prefix + ".bin") and os.path.isfile(prefix + ".idx")):
        raise RuntimeError(
            f"--mlperf_flavor: MLPerf C4 dataset not found at {prefix}.{{bin,idx}}. "
            "Set MLPERF_DATA_ROOT to the directory containing <size>/c4-train.en_6_text_document.{bin,idx}."
        )

    index_cache_dir = os.environ.get("MLPERF_INDEX_CACHE_DIR") or os.path.join(
        os.environ.get("LLMB_INSTALL", "/tmp"), ".cache", f"mlperf_idx_{size}"
    )
    os.makedirs(index_cache_dir, exist_ok=True)
    return prefix, index_cache_dir


def apply_mlperf_flavor(args: argparse.Namespace) -> None:
    """Mutate args in place to apply MLPerf v6.0 apples-to-apples configuration; sets MLPERF_PARITY_* env vars for perf_plugins/overrides to pick up."""
    # Shape table is derived from MLPerf v6.0 NVIDIA GB200 submission configs.
    # Other GPU types have different MLPerf shapes and need separate validation.
    if args.gpu not in ("gb200", "gb300"):
        raise RuntimeError(
            f"--mlperf_flavor currently only supports gpu in (gb200, gb300), got '{args.gpu}'. Other GPU types not yet validated."
        )
    key = (args.model_recipe_name, args.compute_dtype, args.num_gpus, args.gpu)
    shape = _MLPERF_SHAPES.get(key)
    if shape is None:
        raise RuntimeError(
            f"--mlperf_flavor: no MLPerf shape mapping for {key}. "
            f"Supported: {sorted(_MLPERF_SHAPES.keys())}"
        )
    tp, pp, vp, cp, mbs, gbs, parity_mode = shape

    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.virtual_pipeline_model_parallel_size = vp
    args.context_parallel_size = cp
    args.micro_batch_size = mbs
    args.global_batch_size = gbs

    # Dataset: MLPerf C4 shards via rp2 if MLPERF_DATA_ROOT is set; otherwise keep MBridge default (mock).
    # C4 is optional - the shape + parity knobs are the primary apples-to-apples levers; the C4 dataset
    # typically adds +4-8% TFLOPS/GPU over mock, but the comparison is still meaningful without it.
    if os.environ.get("MLPERF_DATA_ROOT"):
        data_prefix, index_cache_dir = _resolve_dataset(args.model_recipe_name)
        args.data = "rp2"
        args.dataset_paths = [data_prefix]
        args.index_mapping_dir = index_cache_dir
        # Bind-mount data + index cache into container (Lustre paths aren't auto-mounted by NeMo-Run).
        data_root = os.path.dirname(os.path.dirname(data_prefix))
        if args.custom_mounts is None:
            args.custom_mounts = []
        for mount in (data_root, index_cache_dir):
            if mount not in args.custom_mounts:
                args.custom_mounts.append(mount)
        dataset_msg = f"data={data_prefix}"
    else:
        dataset_msg = "data=MBridge default (mock; set MLPERF_DATA_ROOT to use MLPerf C4 for full apples-to-apples)"

    # Set MLPERF_PARITY_* env vars so perf_plugins (host-side) + overrides.py (in-container) pick them up.
    env_var = {"F16_ATTN": "MLPERF_PARITY_F16_ATTN", "FP4_ATTN": "MLPERF_PARITY_FP4_ATTN", "405B": "MLPERF_PARITY_405B"}[parity_mode]
    os.environ[env_var] = "1"

    # Forward shape + data to inner script (setup_experiment.py forwards original sys.argv).
    shape_argv = [
        "--tensor_model_parallel_size", str(tp),
        "--pipeline_model_parallel_size", str(pp),
        "--context_parallel_size", str(cp),
        "--micro_batch_size", str(mbs),
        "--global_batch_size", str(gbs),
    ]
    if vp is not None:
        shape_argv.extend(["--virtual_pipeline_model_parallel_size", str(vp)])
    sys.argv.extend(shape_argv)
    if os.environ.get("MLPERF_DATA_ROOT"):
        sys.argv.extend([
            "--data", "rp2",
            "--dataset_paths", data_prefix,
            "--index_mapping_dir", index_cache_dir,
        ])

    logger.info(
        f"--mlperf_flavor (v6.0): {args.model_recipe_name}/{args.compute_dtype}/{args.num_gpus} -> "
        f"TP={tp} PP={pp} VP={vp} CP={cp} MBS={mbs} GBS={gbs} parity={parity_mode} {dataset_msg}"
    )
