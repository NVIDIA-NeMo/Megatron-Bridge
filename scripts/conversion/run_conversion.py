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
"""Run CPU or distributed GPU checkpoint conversion inside the job environment."""

import argparse
import logging

import cpu_backend
import gpu_backend
from arguments import build_parser
from utils import resolve_hf_commit_revision, resolve_hf_model_revision


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure deterministic logging for the conversion worker CLI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate worker arguments for direct invocations and submitted jobs."""
    for name in ("tp", "pp", "ep", "etp"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1.")
    distributed_timeout_minutes = getattr(args, "distributed_timeout_minutes", None)
    if distributed_timeout_minutes is not None and distributed_timeout_minutes < 1:
        raise ValueError("--distributed-timeout-minutes must be at least 1.")
    if args.device == "cpu" and any(getattr(args, name) != 1 for name in ("tp", "pp", "ep", "etp")):
        raise ValueError("CPU conversion requires TP=PP=EP=ETP=1.")
    if args.command == "export":
        if args.save_every_n_ranks < 1:
            raise ValueError("--save-every-n-ranks must be at least 1.")
        distributed_save = args.distributed_save if args.distributed_save is not None else args.device == "gpu"
        if args.device == "cpu" and distributed_save:
            raise ValueError("--distributed-save is only supported by the GPU backend.")
        if not distributed_save and args.save_every_n_ranks != 1:
            raise ValueError("--save-every-n-ranks requires --distributed-save.")
        if args.device == "cpu" and args.export_weight_dtype is not None:
            raise ValueError("--export-weight-dtype is only supported by the GPU backend.")


def _run_import(args: argparse.Namespace) -> None:
    """Dispatch an import to the selected backend."""
    common_args = {
        "hf_model": args.hf_model,
        "hf_revision": args.hf_revision,
        "megatron_path": args.megatron_path,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "overwrite": args.overwrite,
    }
    if args.device == "cpu":
        cpu_backend.import_checkpoint(**common_args)
        return
    gpu_backend.import_checkpoint(
        **common_args,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
    )


def _run_export(args: argparse.Namespace) -> None:
    """Dispatch an export to the selected backend."""
    common_args = {
        "hf_model": args.hf_model,
        "megatron_path": args.megatron_path,
        "hf_path": args.hf_path,
        "show_progress": not args.no_progress,
        "strict": not args.not_strict,
        "trust_remote_code": args.trust_remote_code,
        "overwrite": args.overwrite,
    }
    if args.device == "cpu":
        cpu_backend.export_checkpoint(**common_args)
        return
    distributed_save = args.distributed_save if args.distributed_save is not None else True
    gpu_backend.export_checkpoint(
        **common_args,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        torch_dtype=args.torch_dtype,
        distributed_save=distributed_save,
        save_every_n_ranks=args.save_every_n_ranks,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
        export_weight_dtype=args.export_weight_dtype,
    )


def _run_roundtrip(args: argparse.Namespace) -> None:
    """Run distributed round-trip weight validation on the GPU backend."""
    gpu_backend.roundtrip_checkpoint(
        hf_model=args.hf_model,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        trust_remote_code=args.trust_remote_code,
        distributed_timeout_minutes=args.distributed_timeout_minutes,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse worker arguments and run checkpoint conversion."""
    args = build_parser(include_execution=False).parse_args(argv)
    _validate_args(args)
    if args.hf_revision is not None:
        if args.command == "import":
            args.hf_revision = resolve_hf_commit_revision(args.hf_model, args.hf_revision)
            logger.info("Resolved Hugging Face import to immutable revision %s", args.hf_revision)
        else:
            args.hf_model = resolve_hf_model_revision(args.hf_model, args.hf_revision)
            logger.info(
                "Resolved Hugging Face revision %s to immutable local snapshot %s",
                args.hf_revision,
                args.hf_model,
            )
    logger.info("Selected %s backend for %s conversion", args.device.upper(), args.command)
    if args.command == "import":
        _run_import(args)
    elif args.command == "export":
        _run_export(args)
    else:
        _run_roundtrip(args)


if __name__ == "__main__":
    _configure_logging()
    main()
