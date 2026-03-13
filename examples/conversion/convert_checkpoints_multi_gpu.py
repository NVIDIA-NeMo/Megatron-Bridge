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
Multi-GPU Checkpoint Conversion: Import (HF -> Megatron) and Export (Megatron -> HF).

This is the distributed counterpart of convert_checkpoints.py. It supports tensor,
pipeline, and expert parallelism for models that do not fit on a single GPU or
require a sharded Megatron checkpoint.

Usage examples:

  # Import a HuggingFace MoE model to Megatron format with expert parallelism
  uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
      --hf-model Qwen/Qwen3-30B-A3B \
      --megatron-path ./checkpoints/qwen3_30b_a3b \
      --tp 1 --ep 8

  # Export a distributed Megatron checkpoint back to HuggingFace format
  uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/convert_checkpoints_multi_gpu.py export \
      --hf-model Qwen/Qwen3-30B-A3B \
      --megatron-path ./checkpoints/qwen3_30b_a3b \
      --hf-path ./exports/qwen3_30b_a3b_hf \
      --tp 1 --ep 8

  # Import with pipeline parallelism
  uv run python -m torch.distributed.run --nproc_per_node=4 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
      --hf-model meta-llama/Llama-3.1-70B \
      --megatron-path ./checkpoints/llama31_70b \
      --tp 2 --pp 2

  # Multi-node import via Slurm srun
  srun --ntasks-per-node=8 ... python \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
      --hf-model moonshotai/Moonlight-16B-A3B \
      --megatron-path ./checkpoints/moonlight \
      --tp 2 --ep 8
"""

import argparse
import os
import sys

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import print_rank_0


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(DTYPE_MAP)}.")
    return DTYPE_MAP[name]


def _check_distributed():
    if os.environ.get("WORLD_SIZE") is None:
        print("This script must be launched with torchrun or srun. Example:")
        print(f"  torchrun --nproc_per_node <gpus> {sys.argv[0]} import --hf-model <id> --megatron-path <path>")
        sys.exit(1)


@torchrun_main
def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    device: str = "cuda",
) -> None:
    """Import a HuggingFace model and save it as a distributed Megatron checkpoint."""
    _check_distributed()
    dtype = _parse_dtype(torch_dtype)

    print_rank_0(f"Importing: {hf_model} -> {megatron_path}")
    print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}  device={device}")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )

    model_provider = bridge.to_megatron_provider(load_weights=True, device=device)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    hf_tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    if trust_remote_code:
        hf_tokenizer_kwargs["trust_remote_code"] = True

    print_rank_0(f"Saving Megatron checkpoint to: {megatron_path}")
    bridge.save_megatron_model(
        megatron_model,
        megatron_path,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )
    print_rank_0(f"Import complete: {megatron_path}")


@torchrun_main
def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    strict: bool = False,
    show_progress: bool = True,
    distributed_save: bool = False,
    save_every_n_ranks: int = 1,
) -> None:
    """Export a distributed Megatron checkpoint to HuggingFace format."""
    _check_distributed()
    dtype = _parse_dtype(torch_dtype)

    print_rank_0(f"Exporting: {megatron_path} -> {hf_path}")
    print_rank_0(f"  TP={tp}  PP={pp}  EP={ep}  ETP={etp}  dtype={torch_dtype}")
    print_rank_0(f"  distributed_save={distributed_save}  save_every_n_ranks={save_every_n_ranks}")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )

    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    mp_overrides = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
    }

    print_rank_0(f"Loading Megatron checkpoint from: {megatron_path}")
    megatron_model = bridge.load_megatron_model(
        megatron_path,
        mp_overrides=mp_overrides,
        wrap_with_ddp=False,
    )
    megatron_model = [m.cuda() for m in megatron_model]

    print_rank_0(f"Saving HuggingFace model to: {hf_path}")
    bridge.save_hf_pretrained(
        megatron_model,
        hf_path,
        show_progress=show_progress,
        strict=strict,
        distributed_save=distributed_save,
        save_every_n_ranks=save_every_n_ranks,
    )
    print_rank_0(f"Export complete: {hf_path}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--torch-dtype",
        choices=list(DTYPE_MAP),
        default="bfloat16",
        help="Model precision (default: bfloat16)",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")


def main():
    """Parse CLI arguments and dispatch to import or export."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU checkpoint conversion between HuggingFace and Megatron formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # Import: HF -> Megatron
    import_parser = subparsers.add_parser("import", help="Import HuggingFace model to distributed Megatron checkpoint")
    _add_common_args(import_parser)
    import_parser.add_argument("--megatron-path", required=True, help="Directory to save the Megatron checkpoint")

    # Export: Megatron -> HF
    export_parser = subparsers.add_parser(
        "export", help="Export distributed Megatron checkpoint to HuggingFace format"
    )
    _add_common_args(export_parser)
    export_parser.add_argument("--megatron-path", required=True, help="Directory containing the Megatron checkpoint")
    export_parser.add_argument("--hf-path", required=True, help="Directory to save the HuggingFace model")
    export_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    export_parser.add_argument(
        "--not-strict", action="store_true", help="Allow source and target to have different keys"
    )
    export_parser.add_argument(
        "--distributed-save",
        action="store_true",
        help="Each rank saves its assigned shards independently (reduces rank-0 memory pressure)",
    )
    export_parser.add_argument(
        "--save-every-n-ranks",
        type=int,
        default=1,
        help="Only every N-th rank writes files (reduces I/O, only with --distributed-save)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "import":
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            strict=not args.not_strict,
            show_progress=not args.no_progress,
            distributed_save=args.distributed_save,
            save_every_n_ranks=args.save_every_n_ranks,
        )


if __name__ == "__main__":
    main()
