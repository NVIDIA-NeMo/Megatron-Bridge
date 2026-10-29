#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Generic MegatronMIMO conversion CLI: HF <-> MIMO.

Loads an HF model via ``AutoBridge``, resolves MIMO route metadata from the
standard bridge/provider or an explicit conversion spec, and imports or exports
a MegatronMIMO distributed checkpoint.

Usage:
    # HF -> MIMO checkpoint.
    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/conversion/convert_megatron_mimo.py import \\
            --hf-model Qwen/Qwen3.5-0.8B \\
            --megatron-path /tmp/qwen35_mimo/ckpt \\
            --component language=tp=1 \\
            --component images=tp=1

    # MIMO checkpoint -> HF checkpoint.
    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/conversion/convert_megatron_mimo.py export \\
            --hf-model Qwen/Qwen3.5-0.8B \\
            --megatron-path /tmp/qwen35_mimo/ckpt \\
            --hf-path /tmp/qwen35_mimo/hf_export \\
            --component language=tp=1 \\
            --component images=tp=1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

from megatron.bridge.models.megatron_mimo import (
    MegatronMIMOBridge,
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


logger = logging.getLogger(__name__)


_COMPONENT_KEY_TO_FIELD = {
    "tp": "tensor_model_parallel_size",
    "pp": "pipeline_model_parallel_size",
    "dp": "data_parallel_size",
    "cp": "context_parallel_size",
    "ep": "expert_model_parallel_size",
    "etp": "expert_tensor_parallel_size",
    "rank_offset": "rank_offset",
}


def _parse_component_spec(raw: str) -> tuple[str, ModuleParallelismConfig]:
    """Parse one ``--component`` flag value into ``(name, ModuleParallelismConfig)``.

    Format: ``<name>=tp=N[,pp=N,dp=N,cp=N,ep=N,etp=N,rank_offset=N]``.
    """
    if "=" not in raw:
        raise ValueError(
            f"--component value {raw!r} must be of the form 'name=tp=N[,pp=N,dp=N,cp=N,ep=N,etp=N,rank_offset=N]'"
        )
    name, _, rest = raw.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"--component value {raw!r} has empty component name")

    kwargs: dict[str, int] = {}
    for token in rest.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--component value {raw!r}: token {token!r} must be 'key=value'")
        key, _, value = token.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in _COMPONENT_KEY_TO_FIELD:
            raise ValueError(
                f"--component value {raw!r}: unknown key {key!r}. Supported keys: {sorted(_COMPONENT_KEY_TO_FIELD)}"
            )
        try:
            kwargs[_COMPONENT_KEY_TO_FIELD[key]] = int(value)
        except ValueError as exc:
            raise ValueError(f"--component value {raw!r}: value for {key!r} must be int, got {value!r}") from exc

    return name, ModuleParallelismConfig(**kwargs)


def _build_parallelism_config(
    component_specs: list[str], world_size: int | None = None
) -> MegatronMIMOParallelismConfig:
    """Build ``MegatronMIMOParallelismConfig`` from repeated ``--component`` flags.

    When ``world_size`` is given and a component omits ``dp=``, the CLI
    auto-assigns a non-colocated layout: equal share of ranks per component
    with sequential ``rank_offset``. Components with ``dp=`` set are left
    alone (user owns the layout).
    """
    if not component_specs:
        raise ValueError(
            "At least one --component flag is required (e.g. --component language=tp=1 --component vision=tp=1)"
        )

    module_parallelisms: dict[str, ModuleParallelismConfig] = {}
    for raw in component_specs:
        name, parallelism = _parse_component_spec(raw)
        if name in module_parallelisms:
            raise ValueError(f"--component {name!r} specified more than once")
        module_parallelisms[name] = parallelism

    if world_size is not None:
        _auto_fill_layout(module_parallelisms, world_size)

    return MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)


def _auto_fill_layout(module_parallelisms: dict[str, ModuleParallelismConfig], world_size: int) -> None:
    """Fill missing ``dp`` and ``rank_offset`` for non-colocated heterogeneous layout."""
    if all(parallelism.data_parallel_size is not None for parallelism in module_parallelisms.values()):
        return

    num_components = len(module_parallelisms)
    if world_size % num_components != 0:
        raise ValueError(
            f"world_size ({world_size}) is not divisible by number of components ({num_components}); "
            f"specify dp=N and rank_offset=N explicitly for non-uniform layouts."
        )
    ranks_per_component = world_size // num_components

    offset = 0
    for name, parallelism in module_parallelisms.items():
        if parallelism.data_parallel_size is None:
            mp = parallelism.total_model_parallel_size
            if ranks_per_component % mp != 0:
                raise ValueError(
                    f"--component {name!r}: ranks per component ({ranks_per_component}) "
                    f"is not divisible by total_model_parallel_size ({mp}); "
                    f"specify dp=N explicitly."
                )
            parallelism.data_parallel_size = ranks_per_component // mp
            parallelism.rank_offset = offset
        # Whether auto-filled or user-supplied, advance offset by this
        # component's ranks so the next auto-filled component lands after it.
        offset += parallelism.total_ranks


def run_import(
    *,
    hf_model: str,
    component_specs: list[str],
    trust_remote_code: bool,
    torch_dtype: torch.dtype,
    megatron_path: str,
) -> None:
    """HF -> MIMO checkpoint."""
    parallelism_config = _build_parallelism_config(component_specs, world_size=dist.get_world_size())

    is_rank_zero = dist.get_rank() == 0
    if is_rank_zero:
        logger.info("Importing HF -> MIMO for %s", hf_model)

    bridge = MegatronMIMOBridge.from_hf_pretrained(
        hf_model,
        parallelism_config=parallelism_config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if is_rank_zero:
        source_bridge = bridge._model_bridge
        logger.info(
            "Source bridge: %s; routes: %s",
            type(source_bridge).__name__,
            [(r.name, r.source_prefix, r.target_module_path) for r in bridge.routes],
        )

    bridge.import_ckpt(
        megatron_path,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs={"trust_remote_code": True} if trust_remote_code else None,
    )
    if is_rank_zero:
        logger.info("HF -> MIMO conversion complete (wrote MIMO dist-checkpoint to %s)", megatron_path)


def run_export(
    *,
    hf_model: str,
    component_specs: list[str],
    trust_remote_code: bool,
    torch_dtype: torch.dtype,
    hf_path: str,
    megatron_path: str,
) -> None:
    """MIMO checkpoint -> HF checkpoint."""
    parallelism_config = _build_parallelism_config(component_specs, world_size=dist.get_world_size())

    is_rank_zero = dist.get_rank() == 0

    bridge = MegatronMIMOBridge.from_hf_pretrained(
        hf_model,
        parallelism_config=parallelism_config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if is_rank_zero:
        logger.info("Exporting MIMO -> HF; loading MIMO ckpt from %s", megatron_path)

    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_path,
        show_progress=is_rank_zero,
    )
    if is_rank_zero:
        logger.info("MIMO -> HF export complete (wrote %s)", hf_path)


def _init_distributed() -> None:
    """Initialize ``torch.distributed`` from torchrun environment variables."""
    if dist.is_initialized():
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


_DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags between import and export subcommands."""
    parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or local path.")
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        dest="component",
        help=(
            "Per-component parallelism: 'name=tp=N[,pp=N,dp=N,cp=N,etp=N,rank_offset=N]'. "
            "Repeat once per component. Names must match the route table declared by the "
            "MIMO metadata resolved from --hf-model."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=list(_DTYPE_MAP),
        help="Model parameter dtype.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing custom modeling/tokenizer code from the HF repo.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level.",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, init torch.distributed, dispatch subcommand."""
    parser = argparse.ArgumentParser(
        description="Generic MegatronMIMO conversion (HF <-> MIMO checkpoint).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Conversion direction.")

    import_parser = subparsers.add_parser("import", help="HF -> MIMO.")
    _add_common_args(import_parser)
    import_parser.add_argument(
        "--megatron-path",
        required=True,
        help=("Write the MIMO dist-checkpoint to this directory after the HF->MIMO conversion."),
    )

    export_parser = subparsers.add_parser("export", help="MIMO -> HF.")
    _add_common_args(export_parser)
    export_parser.add_argument(
        "--megatron-path",
        required=True,
        help=(
            "Path to a MIMO dist-checkpoint to load before export. Either a parent "
            "directory (with iter_* subfolders) or an iter folder directly."
        ),
    )
    export_parser.add_argument(
        "--hf-path",
        required=True,
        help=(
            "Write the exported HF checkpoint to this directory. Artifacts "
            "(config.json, tokenizer files, modeling files) are copied from the "
            "source; safetensors shards preserve the source's sharding layout."
        ),
    )

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    logging.basicConfig(level=getattr(logging, args.log_level))
    torch_dtype = _DTYPE_MAP[args.torch_dtype]

    _init_distributed()
    try:
        if args.command == "import":
            run_import(
                hf_model=args.hf_model,
                component_specs=args.component,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch_dtype,
                megatron_path=args.megatron_path,
            )
        elif args.command == "export":
            run_export(
                hf_model=args.hf_model,
                component_specs=args.component,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch_dtype,
                hf_path=args.hf_path,
                megatron_path=args.megatron_path,
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
