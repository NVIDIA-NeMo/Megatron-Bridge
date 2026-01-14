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
Demonstration script that shows how to stream Canonical LoRA adapter weights from a
Hugging Face model by using the AutoBridge conversion APIs.

The example follows three steps:

1. Load a Hugging Face pretrained model (default: meta-llama/Llama-3.2-1B) with
   AutoBridge to obtain a Megatron provider.
2. Register a Canonical LoRA adapter as a pre-wrap hook so that every targeted
   linear layer is wrapped with LoRA modules when the Megatron model is materialized.
3. Stream the adapter weights with `AutoBridge.export_adapter_weights` and save
   them to a safetensors file without touching the base weights.

Run the example:

    uv run python examples/conversion/stream_adapter_weights.py \
        --output ./adapters/demo.safetensors

Multi-GPU launch (torchrun) with tensor/pipeline/expert parallelism:

    uv run python -m torch.distributed.run --nproc_per_node=4 examples/conversion/stream_adapter_weights.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --output ./adapters/demo_tp2_pp2.safetensors
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from rich.console import Console
from rich.table import Table
from safetensors.torch import save_file

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.peft.canonical_lora import CanonicalLoRA
from megatron.bridge.utils.common_utils import print_rank_0


HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the example."""

    parser = argparse.ArgumentParser(
        description="Stream Canonical LoRA adapter weights from a Megatron model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adapter_weights/demo_lora.safetensors"),
        help="Destination path for the streamed adapter tensors (safetensors file).",
    )
    parser.add_argument(
        "--adapter-dim",
        type=int,
        default=8,
        help="LoRA rank / bottleneck dimension used for the Canonical LoRA adapters.",
    )
    parser.add_argument(
        "--adapter-alpha",
        type=int,
        default=16,
        help="Scaling factor applied to the Canonical LoRA adapters.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Tensor model parallel degree.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Pipeline model parallel degree.",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        help="Expert model parallel degree.",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        help="Expert tensor parallel degree.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the textual progress bar while streaming the adapter weights.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip correctness verification between merged export and adapter merge-back.",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=HF_MODEL_ID,
        help="Hugging Face model ID to convert and attach adapters to.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the Hugging Face repository.",
    )
    return parser.parse_args()


def configure_device(device_index: int = 0) -> torch.device:
    """Return the CUDA device for model initialization (NCCL only)."""

    if not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requested but CUDA devices are not available.")
    # Use LOCAL_RANK when launched via torchrun; fallback to explicit device_index.
    device_index = int(os.environ.get("LOCAL_RANK", device_index))
    device_count = torch.cuda.device_count()
    if device_index < 0 or device_index >= device_count:
        raise ValueError(f"device_index={device_index} is invalid for {device_count} CUDA devices.")
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def calculate_required_world_size(args: argparse.Namespace) -> int:
    """Compute the model-parallel product used to validate distributed setup."""

    return (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.expert_model_parallel_size
        * args.expert_tensor_parallel_size
    )


@contextmanager
def distributed_context(
    required_world_size: int,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
):
    """Initialize torch.distributed (and Megatron model parallel) for the test run."""

    if dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size != required_world_size:
            raise RuntimeError(
                f"Requested world_size={required_world_size} from model-parallel settings "
                f"(tp={tp}, pp={pp}, ep={ep}, etp={etp}), but initialized world_size={world_size}. "
                "Launch with torchrun --nproc_per_node equal to the product."
            )
        yield world_size
        return

    # For multi-rank tests the script should be launched via torchrun so that
    # WORLD_SIZE/RANK/MASTER_* are present. Fall back to a single-process
    # initialization when those are absent.
    if required_world_size > 1 and "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Distributed world size is greater than 1 but WORLD_SIZE is not set. "
            "Launch with torchrun --nproc_per_node equal to the requested world size."
        )

    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        init_method = None
    else:
        # Dynamically allocate a port for the single-process case.
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            addr, port = s.getsockname()
        init_method = f"tcp://{addr}:{port}"

    world_size_env = int(os.environ.get("WORLD_SIZE", required_world_size))
    rank_env = int(os.environ.get("RANK", 0))
    dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size_env, rank=rank_env)
    try:
        world_size = dist.get_world_size()
        if world_size != required_world_size:
            raise RuntimeError(
                f"Requested world_size={required_world_size} from model-parallel settings "
                f"(tp={tp}, pp={pp}, ep={ep}, etp={etp}), but initialized world_size={world_size}. "
                "Launch with torchrun --nproc_per_node equal to the product."
            )
        yield world_size
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def register_canonical_lora_adapter(provider, adapter_dim: int, adapter_alpha: int) -> CanonicalLoRA:
    """Register a Canonical LoRA pre-wrap hook on the given provider."""

    canonical_lora = CanonicalLoRA(
        target_modules=[
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ],
        dim=adapter_dim,
        alpha=adapter_alpha,
        dropout=0.0,
    )

    def apply_lora(model_chunks):
        # Apply LoRA in-place and return the adapted model chunks so that the provider
        # can continue with the standard wrapping process.
        return canonical_lora(model_chunks, training=True)

    provider.register_pre_wrap_hook(apply_lora)
    return canonical_lora


def stream_and_collect_adapters(
    bridge: AutoBridge,
    megatron_model,
    show_progress: bool,
) -> dict[str, torch.Tensor]:
    """Iterate through adapter tensors produced by export_adapter_weights."""

    adapter_state: dict[str, torch.Tensor] = {}
    generator: Iterable[HFWeightTuple] = bridge.export_adapter_weights(
        megatron_model,
        cpu=True,
        show_progress=show_progress,
    )

    for weight_name, tensor in generator:
        adapter_state[weight_name] = tensor
        print_rank_0(f"Collected adapter tensor: {weight_name} with shape {tuple(tensor.shape)}")

    if not adapter_state:
        raise RuntimeError("No adapter tensors were found on the model.")

    return adapter_state


def _normalize_base_weight_name(param_name: str) -> str:
    """Remove the 'base_layer' suffix emitted when merge_adapter_weights=False."""

    if param_name.endswith("base_layer.weight"):
        return param_name[: -len("base_layer.weight")] + "weight"
    return param_name


def collect_hf_state_dict(
    bridge: AutoBridge,
    megatron_model,
    *,
    merge_adapters: bool,
    show_progress: bool,
) -> dict[str, torch.Tensor]:
    """Export HF-format weights and return them as a name->tensor dictionary."""

    state: dict[str, torch.Tensor] = {}
    for name, tensor in bridge.export_hf_weights(
        megatron_model,
        cpu=True,
        show_progress=show_progress,
        merge_adapter_weights=merge_adapters,
    ):
        normalized_name = _normalize_base_weight_name(name) if not merge_adapters else name
        state[normalized_name] = tensor

    return state


def merge_hf_lora_adapters(
    base_state: dict[str, torch.Tensor],
    adapter_state: dict[str, torch.Tensor],
    *,
    alpha: int,
    dim: int,
) -> dict[str, torch.Tensor]:
    """Apply HF-format LoRA adapters onto a base state dict and return the merged copy."""

    merged = dict(base_state)
    grouped: dict[str, dict[str, torch.Tensor]] = {}

    for name, tensor in adapter_state.items():
        if name.endswith(".lora_A.weight"):
            base_name = name[: -len(".lora_A.weight")] + ".weight"
            grouped.setdefault(base_name, {})["A"] = tensor
        elif name.endswith(".lora_B.weight"):
            base_name = name[: -len(".lora_B.weight")] + ".weight"
            grouped.setdefault(base_name, {})["B"] = tensor

    scale = alpha / float(dim)
    for base_name, parts in grouped.items():
        base_weight = base_state.get(base_name)
        lora_A = parts.get("A")
        lora_B = parts.get("B")

        if base_weight is None or lora_A is None or lora_B is None:
            # Skip incomplete pairs; verification will flag missing entries.
            continue

        lora_A = lora_A.to(base_weight.dtype)
        lora_B = lora_B.to(base_weight.dtype)
        delta = torch.matmul(lora_B, lora_A) * scale

        if delta.shape != base_weight.shape:
            raise ValueError(
                f"LoRA delta for {base_name} has shape {tuple(delta.shape)} "
                f"but base weight shape is {tuple(base_weight.shape)}."
            )

        merged[base_name] = base_weight + delta

    return merged


def compare_state_dicts(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[list[str], list[str]]:
    """Return (mismatched_keys, extra_keys) between two state dicts."""

    mismatches: list[str] = []
    for name, ref_tensor in reference.items():
        cand_tensor = candidate.get(name)
        if cand_tensor is None:
            mismatches.append(f"{name} (missing)")
            continue

        if not torch.allclose(ref_tensor, cand_tensor, rtol=rtol, atol=atol):
            diff = (ref_tensor - cand_tensor).abs()
            mismatches.append(f"{name} (max diff={diff.max().item():.3e})")

    extra_keys = [name for name in candidate.keys() if name not in reference]
    return mismatches, extra_keys


def main() -> None:
    """Create a model, attach Canonical LoRA, and stream adapter weights to disk."""

    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = configure_device()
    use_gpu = True
    required_world_size = calculate_required_world_size(args)

    print_rank_0(
        f"🧮 Model-parallel settings: tp={args.tensor_model_parallel_size}, "
        f"pp={args.pipeline_model_parallel_size}, "
        f"ep={args.expert_model_parallel_size}, etp={args.expert_tensor_parallel_size}. "
        f"Expected world_size={required_world_size}."
    )

    print_rank_0(f"🔧 Loading Hugging Face model {args.hf_model_id} with bfloat16 weights...")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_id,
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_id,
        ),
        torch_dtype=torch.bfloat16,
    )
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    provider.pipeline_dtype = torch.bfloat16
    provider.params_dtype = torch.bfloat16
    provider.expert_model_parallel_size = args.expert_model_parallel_size
    provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    provider.finalize()

    print_rank_0("🧩 Registering Canonical LoRA adapters...")
    register_canonical_lora_adapter(provider, adapter_dim=args.adapter_dim, adapter_alpha=args.adapter_alpha)

    print_rank_0("⚙️  Materializing Megatron model inside a temporary distributed context (backend=NCCL)...")
    with distributed_context(
        required_world_size=required_world_size,
        tp=args.tensor_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        ep=args.expert_model_parallel_size,
        etp=args.expert_tensor_parallel_size,
    ):
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=not use_gpu,
            init_model_with_meta_device=not use_gpu,
        )
        if use_gpu:
            megatron_model = [chunk.to(device) for chunk in megatron_model]

        print_rank_0("📤 Streaming adapter tensors only (base weights remain untouched)...")
        adapter_state = stream_and_collect_adapters(
            bridge,
            megatron_model,
            show_progress=not args.no_progress,
        )

        if not args.no_verify:
            is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            table = None
            if is_rank0:
                table = Table(title="Adapter Merge Verification")
                table.add_column("Weight Name", style="cyan")
                table.add_column("Shape")
                table.add_column("DType")
                table.add_column("Device")
                table.add_column("Matches Merged", justify="center")

            print_rank_0("🔍 Verifying adapter merge-back matches merged HF export...")
            merged_reference = collect_hf_state_dict(
                bridge,
                megatron_model,
                merge_adapters=True,
                show_progress=not args.no_progress,
            )
            base_state = collect_hf_state_dict(
                bridge,
                megatron_model,
                merge_adapters=False,
                show_progress=False,
            )
            merged_from_adapter = merge_hf_lora_adapters(
                base_state,
                adapter_state,
                alpha=args.adapter_alpha,
                dim=args.adapter_dim,
            )
            mismatches: list[str] = []
            for name, ref_tensor in merged_reference.items():
                cand_tensor = merged_from_adapter.get(name)
                if cand_tensor is None:
                    mismatches.append(f"{name} (missing)")
                    match = False
                else:
                    cand_tensor = cand_tensor.to(ref_tensor.device)
                    match = torch.allclose(ref_tensor, cand_tensor, rtol=1e-5, atol=1e-6)
                    if not match:
                        diff = (ref_tensor - cand_tensor).abs()
                        mismatches.append(f"{name} (max diff={diff.max().item():.3e})")

                if table:
                    table.add_row(
                        name,
                        str(tuple(ref_tensor.shape)),
                        str(ref_tensor.dtype).replace("torch.", ""),
                        str(ref_tensor.device),
                        "✅" if match else "❌",
                    )

            extra_keys = [name for name in merged_from_adapter.keys() if name not in merged_reference]
            if mismatches or extra_keys:
                mismatch_summary = "; ".join(mismatches[:5])
                extras_summary = ", ".join(extra_keys[:5])
                raise RuntimeError(
                    "Adapter merge verification failed: "
                    f"{len(mismatches)} mismatched tensors "
                    f"({mismatch_summary}) "
                    f"and {len(extra_keys)} unexpected tensors ({extras_summary})."
                )
            if table:
                console.print(table)
            print_rank_0(f"✅ Verification passed: {len(merged_reference)} tensors match.")
    print_rank_0(f"💾 Saving {len(adapter_state)} adapter tensors to {args.output} ...")
    save_file(adapter_state, str(args.output))
    print_rank_0("✅ Done! You can now load the adapters independently of the base model.")


if __name__ == "__main__":
    main()
