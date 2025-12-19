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
randomly initialized Megatron model by using the AutoBridge conversion APIs.

The example follows three steps:

1. Build a tiny Llama configuration and create an AutoBridge instance from it
   (no pretrained weights are downloaded).
2. Register a Canonical LoRA adapter as a pre-wrap hook so that every targeted
   linear layer is wrapped with LoRA modules when the Megatron model is materialized.
3. Stream the adapter weights with `AutoBridge.export_adapter_weights` and save
   them to a safetensors file without touching the base weights.

Run the example (GPU execution shown; omit the backend flag to auto-detect):

    uv run python examples/conversion/stream_adapter_weights.py \
        --backend nccl --device-index 0 --output ./adapters/demo.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
from megatron.bridge.peft.canonical_lora import CanonicalLoRA
from megatron.bridge.training.model_load_save import temporary_distributed_context


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
        "--backend",
        choices=["auto", "gloo", "nccl"],
        default="auto",
        help="Distributed backend to use when building the Megatron model. "
        "Set to 'nccl' to force GPU execution or 'gloo' to force CPU execution.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index to use when '--backend' resolves to 'nccl'. Ignored otherwise.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the textual progress bar while streaming the adapter weights.",
    )
    return parser.parse_args()


def resolve_backend(requested_backend: str) -> str:
    """Resolve backend selection, defaulting to NCCL when GPUs are available."""

    if requested_backend != "auto":
        return requested_backend
    return "nccl" if torch.cuda.is_available() else "gloo"


def configure_device(backend: str, device_index: int) -> torch.device:
    """Return the torch.device that should be used for model initialization."""

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requested but CUDA devices are not available.")
        device_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= device_count:
            raise ValueError(f"device_index={device_index} is invalid for {device_count} CUDA devices.")
        torch.cuda.set_device(device_index)
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def build_tiny_llama_config() -> LlamaConfig:
    """Create a small Llama configuration that is fast to instantiate."""

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1536,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=8,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
    )
    # Ensure the configuration advertises a supported architecture suffix.
    config.architectures = ["LlamaForCausalLM"]
    return config


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
        print(f"Collected adapter tensor: {weight_name} with shape {tuple(tensor.shape)}")

    if not adapter_state:
        raise RuntimeError("No adapter tensors were found on the model.")

    return adapter_state


def main() -> None:
    """Create a model, attach Canonical LoRA, and stream adapter weights to disk."""

    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend = resolve_backend(args.backend)
    device = configure_device(backend, args.device_index)
    use_gpu = backend == "nccl"

    print("🔧 Building tiny Llama configuration with random weights...")
    config = build_tiny_llama_config()
    bridge = AutoBridge.from_hf_config(config)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.finalize()

    print("🧩 Registering Canonical LoRA adapters...")
    register_canonical_lora_adapter(provider, adapter_dim=args.adapter_dim, adapter_alpha=args.adapter_alpha)

    backend_display = backend.upper()
    print(f"⚙️  Materializing Megatron model inside a temporary distributed context (backend={backend_display})...")
    with temporary_distributed_context(backend=backend):
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=not use_gpu,
            init_model_with_meta_device=not use_gpu,
        )
        if use_gpu:
            megatron_model = [chunk.to(device) for chunk in megatron_model]

        print("📤 Streaming adapter tensors only (base weights remain untouched)...")
        adapter_state = stream_and_collect_adapters(
            bridge,
            megatron_model,
            show_progress=not args.no_progress,
        )
    print(f"💾 Saving {len(adapter_state)} adapter tensors to {args.output} ...")
    save_file(adapter_state, str(args.output))
    print("✅ Done! You can now load the adapters independently of the base model.")


if __name__ == "__main__":
    main()
