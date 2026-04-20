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
Megatron-FSDP DTensor Checkpoint Conversion:
Usage examples:

  # Import HF model to Megatron-FSDP DTensor checkpoint
  uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/conversion/fsdp/convert_checkpoints_fsdp.py import \
  --hf-model Qwen/Qwen2.5-7B-Instruct \
  --megatron-path ./checkpoints/qwen25_7b_fsdp_dtensor \
  --tp 2 --cp 1 --ep 1 \
  --ckpt-format fsdp_dtensor

  # Export Megatron checkpoint to HuggingFace
  uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/conversion/fsdp/convert_checkpoints_fsdp.py export \
  --hf-model Qwen/Qwen2.5-7B-Instruct \
  --megatron-path ./checkpoints/qwen25_7b_fsdp_dtensor \
  --hf-path exports/qwen25_7b_hf \
  --tp 2 --cp 1 --ep 1 \
  --ckpt-format fsdp_dtensor
"""

import argparse
import os
import sys

import torch
from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.checkpointing import load_checkpoint
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer, LoggerConfig, OptimizerConfig
from megatron.bridge.training.model_load_save import save_megatron_model as save_native_megatron_model
from megatron.bridge.training.state import GlobalState
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


def _check_distributed() -> None:
    if os.environ.get("WORLD_SIZE") is None:
        print("This script must be launched with torchrun or srun. Example:")
        print(
            f"  uv run python -m torch.distributed.run --nproc_per_node <gpus> "
            f"{sys.argv[0]} import --hf-model <id> --megatron-path <path>"
        )
        sys.exit(1)


def _check_world_size(tp: int, cp: int, ep: int) -> None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as err:
        raise ValueError("Invalid WORLD_SIZE environment variable.") from err

    mp_size = tp * cp * ep
    if mp_size <= 0:
        raise ValueError(f"Invalid parallel sizes: tp={tp}, cp={cp}, ep={ep}")
    if world_size % mp_size != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by tp*cp*ep ({mp_size}). Got tp={tp}, cp={cp}, ep={ep}."
        )


def _build_fsdp_distributed_model(bridge: AutoBridge, tp: int, cp: int, ep: int, dtype: torch.dtype):
    """Build and return a Megatron-FSDP wrapped model list."""
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.context_parallel_size = cp
    model_provider.expert_model_parallel_size = ep
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype
    model_provider.gradient_accumulation_fusion = False
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        check_for_nan_in_grad=True,
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads_params",
    )
    megatron_model = model_provider.provide_distributed_model(
        ddp_config=ddp_config,
        use_megatron_fsdp=True,
        use_torch_fsdp2=False,
        overlap_param_gather_with_optimizer_step=False,
        data_parallel_random_init=False,
    )
    return model_provider, ddp_config, megatron_model


@torchrun_main
def import_hf_to_megatron_fsdp(
    hf_model: str,
    megatron_path: str,
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    low_memory_save: bool = True,
    ckpt_format: str = "fsdp_dtensor",
) -> None:
    """Import a HuggingFace model and save it as a DTensor checkpoint."""
    _check_distributed()
    _check_world_size(tp=tp, cp=cp, ep=ep)
    dtype = _parse_dtype(torch_dtype)

    print_rank_0(f"Importing: {hf_model} -> {megatron_path}")
    print_rank_0(f"  TP={tp}  CP={cp}  EP={ep}  dtype={torch_dtype}  ckpt_format={ckpt_format}")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )

    _, _, megatron_model = _build_fsdp_distributed_model(bridge, tp=tp, cp=cp, ep=ep, dtype=dtype)

    bridge.load_hf_weights(megatron_model)

    hf_tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        hf_tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    if trust_remote_code:
        hf_tokenizer_kwargs["trust_remote_code"] = True

    effective_low_memory_save = low_memory_save
    if ckpt_format == "fsdp_dtensor" and low_memory_save:
        # fsdp_dtensor save path requires the live model object.
        print_rank_0("low_memory_save is not supported with fsdp_dtensor. Forcing low_memory_save=False.")
        effective_low_memory_save = False

    print_rank_0(f"Saving Megatron checkpoint to: {megatron_path}")
    save_native_megatron_model(
        megatron_model,
        megatron_path,
        ckpt_format=ckpt_format,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        low_memory_save=effective_low_memory_save,
    )
    print_rank_0(f"Import complete: {megatron_path}")


@torchrun_main
def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    ckpt_format: str = "fsdp_dtensor",
    strict: bool = False,
    show_progress: bool = True,
    distributed_save: bool = False,
    save_every_n_ranks: int = 1,
) -> None:
    """Export Megatron checkpoint to HuggingFace format."""
    _check_distributed()
    _check_world_size(tp=tp, cp=cp, ep=ep)
    dtype = _parse_dtype(torch_dtype)

    print_rank_0(f"Exporting: {megatron_path} -> {hf_path}")
    print_rank_0(f"  TP={tp}  CP={cp}  EP={ep}  dtype={torch_dtype}  ckpt_format={ckpt_format}")
    print_rank_0(f"  distributed_save={distributed_save}  save_every_n_ranks={save_every_n_ranks}")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )

    print_rank_0(f"Loading Megatron checkpoint from: {megatron_path}")
    if ckpt_format == "fsdp_dtensor":
        # Build an FSDP-wrapped model and load with the training checkpoint loader.
        model_provider, ddp_config, megatron_model = _build_fsdp_distributed_model(
            bridge, tp=tp, cp=cp, ep=ep, dtype=dtype
        )

        state = GlobalState()
        state.cfg = ConfigContainer(
            model=model_provider,
            train=None,
            optimizer=OptimizerConfig(use_distributed_optimizer=False),
            ddp=ddp_config,
            scheduler=None,
            dataset=None,
            logger=LoggerConfig(),
            tokenizer=None,
            checkpoint=CheckpointConfig(
                load=megatron_path,
                finetune=True,
                load_optim=False,
                load_rng=False,
                ckpt_format=ckpt_format,
            ),
            dist=None,
        )
        load_checkpoint(
            state=state,
            model=megatron_model,
            optimizer=None,
            opt_param_scheduler=None,
            strict=strict,
        )
    else:
        mp_overrides = {
            "tensor_model_parallel_size": tp,
            "context_parallel_size": cp,
            "expert_model_parallel_size": ep,
            "pipeline_dtype": dtype,
        }
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
    parser.add_argument("--cp", type=int, default=1, help="Context parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument(
        "--torch-dtype",
        choices=list(DTYPE_MAP),
        default="bfloat16",
        help="Model precision (default: bfloat16)",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")
    parser.add_argument(
        "--ckpt-format",
        choices=["torch_dist", "fsdp_dtensor"],
        default="fsdp_dtensor",
        help="Megatron checkpoint format to load/save (default: fsdp_dtensor for Megatron-FSDP)",
    )


def main() -> None:
    """Megatron-FSDP DTensor Checkpoint Conversion"""
    parser = argparse.ArgumentParser(
        description="HuggingFace and Megatron-FSDP DTensor checkpoint conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    import_parser = subparsers.add_parser(
        "import", help="Import HuggingFace model to Megatron-FSDP DTensor checkpoint"
    )
    _add_common_args(import_parser)
    import_parser.add_argument("--megatron-path", required=True, help="Directory to save the DTensor checkpoint")
    import_parser.add_argument(
        "--no-low-memory-save",
        action="store_true",
        help="Disable low-memory save mode (keeps model alive after save)",
    )

    export_parser = subparsers.add_parser("export", help="Export DTensor checkpoint to HuggingFace format")
    _add_common_args(export_parser)
    export_parser.add_argument("--megatron-path", required=True, help="Directory containing the DTensor checkpoint")
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
        import_hf_to_megatron_fsdp(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            tp=args.tp,
            cp=args.cp,
            ep=args.ep,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            low_memory_save=not args.no_low_memory_save,
            ckpt_format=args.ckpt_format,
        )
    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            tp=args.tp,
            cp=args.cp,
            ep=args.ep,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            ckpt_format=args.ckpt_format,
            strict=not args.not_strict,
            show_progress=not args.no_progress,
            distributed_save=args.distributed_save,
            save_every_n_ranks=args.save_every_n_ranks,
        )


if __name__ == "__main__":
    main()
