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

"""Run a single-rank Qwen3-Omni thinker smoke forward on one real image+audio sample."""

import argparse
import datetime
import io
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from PIL import Image
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from megatron.bridge import AutoBridge


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Qwen3-Omni smoke forward example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model-path",
        type=Path,
        required=True,
        help="Path to the local Hugging Face Qwen3-Omni thinker checkpoint.",
    )
    parser.add_argument(
        "--sample-parquet",
        type=Path,
        required=True,
        help="Path to an OmniBench-style parquet file containing image/audio samples.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Row index inside the parquet file to use for smoke validation.",
    )
    parser.add_argument(
        "--megatron-model-path",
        type=Path,
        default=None,
        help="Optional Megatron checkpoint path. If omitted, the script runs the HF thinker path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is likely to happen next?",
        help="Prompt text paired with the local image+audio sample.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for Megatron loading.")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size for Megatron loading.")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size for Megatron loading.")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size for Megatron loading.")
    return parser.parse_args()


def load_real_sample_inputs(model_path: Path, parquet_path: Path, sample_index: int, prompt: str) -> dict[str, torch.Tensor]:
    """Build one real image+audio input batch with the checkpoint's own processor."""
    row = pd.read_parquet(parquet_path).iloc[sample_index]
    image = Image.open(io.BytesIO(row["images"][0]["bytes"])).convert("RGB")
    audio = np.asarray(row["audios"][0], dtype=np.float32)

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "audio"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
    return processor(text=text, images=[image], audio=[audio], return_tensors="pt")


def move_inputs_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Move a processor batch to the target device and cast float multimodal tensors."""
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if dtype is not None and key in {"pixel_values", "pixel_values_videos", "input_features"}:
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def find_free_port() -> str:
    """Reserve and return an ephemeral localhost port for single-process distributed init."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def init_single_rank_dist() -> None:
    """Initialize a single-rank distributed process group for Megatron loading."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", find_free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=1,
        rank=0,
        timeout=datetime.timedelta(minutes=30),
    )


def init_model_parallel() -> None:
    """Initialize 1-rank Megatron model-parallel state for smoke execution."""
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model_parallel_cuda_manual_seed(123)
    else:
        torch.manual_seed(123)


def cleanup_distributed() -> None:
    """Tear down Megatron and torch.distributed state after smoke execution."""
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def run_hf_smoke(args: argparse.Namespace) -> None:
    """Run the HF thinker path on one real image+audio sample."""
    inputs = load_real_sample_inputs(args.hf_model_path, args.sample_parquet, args.sample_index, args.prompt)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.hf_model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    prepared = move_inputs_to_device(inputs, device, dtype=model.dtype)

    with torch.no_grad():
        outputs = model.thinker(**prepared)

    print(f"mode=hf logits_shape={tuple(outputs.logits.shape)} dtype={outputs.logits.dtype}")


def run_megatron_smoke(args: argparse.Namespace) -> None:
    """Run the Megatron thinker path on one real image+audio sample."""
    if args.megatron_model_path is None:
        raise ValueError("--megatron-model-path is required for Megatron smoke mode.")

    init_single_rank_dist()
    init_model_parallel()
    try:
        inputs = load_real_sample_inputs(args.hf_model_path, args.sample_parquet, args.sample_index, args.prompt)
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, dtype=torch.bfloat16)
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": args.tp,
                "pipeline_model_parallel_size": args.pp,
                "expert_model_parallel_size": args.ep,
                "expert_tensor_parallel_size": args.etp,
                "pipeline_dtype": torch.bfloat16,
            },
            wrap_with_ddp=False,
        )
        if isinstance(model, list):
            model = model[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        prepared = move_inputs_to_device(
            inputs,
            device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        prepared["labels"] = prepared["input_ids"].clone()

        with torch.no_grad():
            outputs = model(**prepared)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        print(f"mode=megatron logits_shape={tuple(logits.shape)} dtype={logits.dtype}")
    finally:
        cleanup_distributed()


def main() -> None:
    """Dispatch to HF or Megatron thinker smoke execution."""
    args = parse_args()
    if args.megatron_model_path is None:
        run_hf_smoke(args)
    else:
        run_megatron_smoke(args)


if __name__ == "__main__":
    main()
