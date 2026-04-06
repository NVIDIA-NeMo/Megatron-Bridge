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

"""Compute-node NUMA binding wrapper.

Runs once per rank at launch time, detects GPU-to-NUMA locality via sysfs,
and execs the real training command under numactl. Replaces the previous
hardcoded GPU-family heuristic with runtime topology detection.

Usage:
    python3 numa_bind.py --mode auto -- torchrun ...
    python3 numa_bind.py --mode override --override-file topology.yaml -- torchrun ...
    python3 numa_bind.py --mode off -- torchrun ...
"""

import argparse
import ast
import logging
import operator
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Restricted expression evaluator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST, rank: int) -> int:
    """Recursively evaluate an AST node with only safe integer operations."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, rank)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and node.id == "rank":
        return rank
    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left, rank)
        right = _eval_node(node.right, rank)
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_eval_node(node.operand, rank))
    raise ValueError(
        f"Unsupported expression element: {type(node).__name__}. "
        "Only integer literals, 'rank', and +, -, *, //, % are allowed."
    )


def eval_binding_expr(expr: str, rank: int) -> str:
    """Evaluate a binding expression template for a given rank.

    Supports two forms:
      - Plain value (no braces): returned as-is, e.g. "0" -> "0"
      - Template with {expr}: evaluates the expression, e.g. "{rank // 2}" -> "1"

    Multiple templates in one string are supported for comma-separated values,
    e.g. "{rank * 16}, {rank * 16 + 1}" -> "0, 1" for rank=0.
    """
    if "{" not in str(expr):
        return str(expr)

    def _replace_match(text: str, rank: int) -> str:
        result = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                end = text.index("}", i)
                inner = text[i + 1 : end].strip()
                tree = ast.parse(inner, mode="eval")
                value = _eval_node(tree, rank)
                result.append(str(value))
                i = end + 1
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    return _replace_match(str(expr), rank)


# ---------------------------------------------------------------------------
# YAML override file handling
# ---------------------------------------------------------------------------

VALID_BINDING_KEYS = {"cpunodebind", "membind", "physcpubind"}


def load_override_file(path: str) -> Dict[str, Any]:
    """Load and return the parsed YAML override file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Override file must be a YAML mapping, got {type(data).__name__}")
    return data


def resolve_override_binding(data: Dict[str, Any], rank: int) -> Dict[str, str]:
    """Resolve binding for a specific rank from override file data.

    Resolution order:
      1. Per-rank entry in 'ranks' (if present) — used as-is, no template evaluation
      2. Template in 'binding' — evaluated with the rank value
    """
    result = {}

    # Evaluate templates from 'binding' section
    binding = data.get("binding", {})
    for key, expr in binding.items():
        if key not in VALID_BINDING_KEYS:
            raise ValueError(f"Unknown binding key '{key}'. Valid keys: {VALID_BINDING_KEYS}")
        result[key] = eval_binding_expr(expr, rank)

    # Apply per-rank overrides (these take precedence)
    ranks = data.get("ranks", {})
    rank_override = ranks.get(rank, ranks.get(str(rank), {}))
    if rank_override:
        for key, value in rank_override.items():
            if key not in VALID_BINDING_KEYS:
                raise ValueError(f"Unknown binding key '{key}'. Valid keys: {VALID_BINDING_KEYS}")
            result[key] = str(value)

    return result


def validate_override_file(path: str) -> None:
    """Validate an override file at setup time (before Slurm submission).

    Checks schema structure and template syntax without needing compute-node
    access. Raises ValueError on problems.
    """
    data = load_override_file(path)

    # Validate 'binding' section
    binding = data.get("binding", {})
    if not isinstance(binding, dict):
        raise ValueError("'binding' must be a mapping")
    for key, expr in binding.items():
        if key not in VALID_BINDING_KEYS:
            raise ValueError(f"Unknown binding key '{key}'. Valid keys: {VALID_BINDING_KEYS}")
        # Verify template syntax parses with a dummy rank
        try:
            eval_binding_expr(expr, rank=0)
        except Exception as e:
            raise ValueError(f"Invalid expression for '{key}': {expr!r} — {e}") from e

    # Validate 'ranks' section
    ranks = data.get("ranks", {})
    if not isinstance(ranks, dict):
        raise ValueError("'ranks' must be a mapping")
    for rank_id, rank_binding in ranks.items():
        if not isinstance(rank_binding, dict):
            raise ValueError(f"ranks[{rank_id}] must be a mapping")
        for key in rank_binding:
            if key not in VALID_BINDING_KEYS:
                raise ValueError(
                    f"Unknown binding key '{key}' in ranks[{rank_id}]. "
                    f"Valid keys: {VALID_BINDING_KEYS}"
                )

    # Warn about unknown top-level keys
    known_keys = {"binding", "ranks"}
    unknown = set(data.keys()) - known_keys
    if unknown:
        raise ValueError(f"Unknown top-level keys: {unknown}. Valid keys: {known_keys}")


# ---------------------------------------------------------------------------
# GPU-to-NUMA auto-detection
# ---------------------------------------------------------------------------


def detect_numa_node(local_rank: int) -> Optional[int]:
    """Detect the NUMA node for a GPU by its local rank via sysfs.

    Uses nvidia-smi to get the PCI bus ID, then reads the kernel's
    numa_node attribute from sysfs. Returns None if detection fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
                "-i",
                str(local_rank),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return None

        bdf = result.stdout.strip().lower()
        if not bdf:
            logger.warning("nvidia-smi returned empty PCI bus ID for rank %d", local_rank)
            return None

        # nvidia-smi may return an 8-digit domain (e.g., 00000000:19:00.0)
        # but sysfs uses a 4-digit domain (e.g., 0000:19:00.0). Normalize.
        parts = bdf.split(":")
        if len(parts) >= 3 and len(parts[0]) == 8:
            bdf = parts[0][4:] + ":" + ":".join(parts[1:])

        numa_path = Path(f"/sys/bus/pci/devices/{bdf}/numa_node")
        if not numa_path.exists():
            logger.warning("sysfs path not found: %s", numa_path)
            return None

        node = int(numa_path.read_text().strip())
        if node < 0:
            logger.warning("sysfs reports NUMA node %d for %s (no NUMA info available)", node, bdf)
            return None

        return node

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out for rank %d", local_rank)
        return None
    except (OSError, ValueError) as e:
        logger.warning("NUMA detection failed for rank %d: %s", local_rank, e)
        return None


# ---------------------------------------------------------------------------
# Build numactl arguments
# ---------------------------------------------------------------------------


def build_numactl_args(binding: Dict[str, str]) -> List[str]:
    """Convert a binding dict to numactl command-line arguments."""
    args = []
    if "cpunodebind" in binding:
        args.extend(["--cpunodebind", binding["cpunodebind"]])
    if "membind" in binding:
        args.extend(["--membind", binding["membind"]])
    if "physcpubind" in binding:
        args.extend(["--physcpubind", binding["physcpubind"]])
    return args


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse wrapper CLI arguments."""
    parser = argparse.ArgumentParser(
        description="NUMA binding wrapper for GPU training jobs",
        usage="numa_bind.py --mode {auto,off,override} [options] -- <command...>",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "off", "override"],
        required=True,
        help="Binding mode: auto (detect via sysfs), off (skip), override (use YAML file)",
    )
    parser.add_argument(
        "--override-file",
        type=str,
        default=None,
        help="Path to YAML override file (required for --mode override)",
    )
    parser.add_argument(
        "--hard-fail",
        action="store_true",
        default=False,
        help="Exit with error instead of warning when detection fails (useful for testing)",
    )

    # Split on '--' to separate wrapper args from the real command
    if "--" in argv:
        sep = argv.index("--")
        wrapper_argv = argv[:sep]
        command = argv[sep + 1 :]
    else:
        wrapper_argv = argv
        command = []

    args = parser.parse_args(wrapper_argv)
    args.command = command
    return args


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point. Resolves NUMA binding and execs the command."""
    logging.basicConfig(
        level=logging.INFO,
        format="[numa_bind] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not args.command:
        logger.error("No command specified after '--'")
        sys.exit(1)

    # Mode: off — just exec the command
    if args.mode == "off":
        logger.info("mode=off, skipping NUMA binding")
        os.execvp(args.command[0], args.command)

    slurm_localid = os.environ.get("SLURM_LOCALID")
    if slurm_localid is None:
        logger.error("SLURM_LOCALID not set — numa_bind.py must run within a Slurm job")
        sys.exit(1)
    local_rank = int(slurm_localid)

    if args.mode == "override":
        if not args.override_file:
            logger.error("--override-file is required when --mode=override")
            sys.exit(1)

        data = load_override_file(args.override_file)
        binding = resolve_override_binding(data, local_rank)

    else:  # auto
        node = detect_numa_node(local_rank)
        if node is None:
            if args.hard_fail:
                logger.error("NUMA detection failed for rank %d (--hard-fail enabled)", local_rank)
                sys.exit(1)
            logger.warning(
                "rank=%d: could not detect NUMA node, running without binding", local_rank
            )
            os.execvp(args.command[0], args.command)
        binding = {"cpunodebind": str(node), "membind": str(node)}

    numa_args = build_numactl_args(binding)
    if not numa_args:
        logger.warning("rank=%d: no binding flags resolved, running without numactl", local_rank)
        os.execvp(args.command[0], args.command)

    full_cmd = ["numactl"] + numa_args + args.command
    logger.info("rank=%d mode=%s -> %s", local_rank, args.mode, " ".join(full_cmd))
    os.execvp("numactl", full_cmd)


if __name__ == "__main__":
    main()
