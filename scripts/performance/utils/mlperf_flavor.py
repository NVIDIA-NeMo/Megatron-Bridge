# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MLPerf v5.1 apples-to-apples flavor (--mlperf_flavor): resolves shape, dataset, mounts, and parity env vars."""

import argparse
import logging
import os
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# (model_recipe_name, compute_dtype, num_gpus) -> (TP, PP, VP, CP, MBS, GBS, parity_mode); shapes derived from v5.1 NVIDIA submission configs.
_MLPERF_V51_SHAPES: Dict[Tuple[str, str, int], Tuple[int, int, int, int, int, int, str]] = {
    ("llama3_8b",   "fp8_cs",   8):   (1, 1, 1, 1, 1, 8,    "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   16):  (1, 1, 1, 2, 1, 8,    "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   32):  (1, 1, 1, 2, 1, 16,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   64):  (1, 1, 1, 2, 1, 32,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   72):  (1, 1, 1, 2, 1, 36,   "F16_ATTN"),
    ("llama3_8b",   "fp8_cs",   128): (2, 1, 1, 4, 1, 16,   "F16_ATTN"),
    ("llama3_8b",   "nvfp4",    8):   (1, 1, 1, 1, 2, 16,   "FP4_ATTN"),
    ("llama31_405b","fp8_cs",   256): (4, 8, 8, 2, 1, 576,  "405B"),
    ("llama31_405b","fp8_cs",   512): (4, 8, 8, 2, 1, 1152, "405B"),
    ("llama31_405b","nvfp4",    256): (4, 8, 8, 2, 1, 576,  "405B"),
    ("llama31_405b","nvfp4",    512): (4, 8, 8, 2, 1, 1152, "405B"),
}


def _resolve_dataset(model_recipe_name: str) -> Tuple[str, str]:
    """Return (data_prefix, index_cache_dir) for the model size."""
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
    """Mutate args in place to apply MLPerf v5.1 apples-to-apples configuration; sets MLPERF_PARITY_* env vars for perf_plugins/overrides to pick up."""
    # Shape table is derived from v5.1 GB200 reference configs; other GPU types have different v5.1 shapes and need separate validation.
    if args.gpu != "gb200":
        raise RuntimeError(
            f"--mlperf_flavor currently only supports gpu=gb200, got '{args.gpu}'. Other GPU types not yet validated."
        )
    key = (args.model_recipe_name, args.compute_dtype, args.num_gpus)
    shape = _MLPERF_V51_SHAPES.get(key)
    if shape is None:
        raise RuntimeError(
            f"--mlperf_flavor: no v5.1 shape mapping for {key}. "
            f"Supported: {sorted(_MLPERF_V51_SHAPES.keys())}"
        )
    tp, pp, vp, cp, mbs, gbs, parity_mode = shape

    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.virtual_pipeline_model_parallel_size = vp
    args.context_parallel_size = cp
    args.micro_batch_size = mbs
    args.global_batch_size = gbs

    # Dataset: MLPerf C4 shards via rp2; tokenizer left at MBridge default (same vocab as preprocessed shards).
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

    # Set MLPERF_PARITY_* env vars so perf_plugins (host-side) + overrides.py (in-container) pick them up.
    env_var = {"F16_ATTN": "MLPERF_PARITY_F16_ATTN", "FP4_ATTN": "MLPERF_PARITY_FP4_ATTN", "405B": "MLPERF_PARITY_405B"}[parity_mode]
    os.environ[env_var] = "1"

    logger.info(
        f"--mlperf_flavor: {args.model_recipe_name}/{args.compute_dtype}/{args.num_gpus} -> "
        f"TP={tp} PP={pp} VP={vp} CP={cp} MBS={mbs} GBS={gbs} parity={parity_mode} data={data_prefix}"
    )
