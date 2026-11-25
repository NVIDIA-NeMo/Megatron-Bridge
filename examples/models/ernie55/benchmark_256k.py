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

"""Prepare, run and summarize the ERNIE 5.5 256K benchmark. See README-256k.md."""

import argparse
import atexit
import datetime
import hashlib
import json
import logging
import os
import random
import re
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path


logger = logging.getLogger(__name__)
LENGTHS = {
    "single": (262144,),
    "uniform": (32768,) * 8,
    "skew": (131200, 65408, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 117),
}


def prepare(args: argparse.Namespace) -> None:
    """Write deterministic tokenized documents. Bridge packs them into one row."""
    rng = random.Random(args.seed)
    lengths = LENGTHS[args.layout]
    assert sum((n + 31) // 32 * 32 for n in lengths) == 262144
    src = args.source.open(encoding="utf-8") if args.source else nullcontext(None)
    with src as source, args.output.open("x", encoding="utf-8") as stream:
        for n in lengths:
            if source is None:
                tokens = [rng.randrange(256, 8192) for _ in range(n)]
            else:
                row = source.readline()
                if not row:
                    raise ValueError("One source document per requested output document is required.")
                tokens = json.loads(row)["input_ids"][:n]
                if len(tokens) != n:
                    raise ValueError(f"Expected at least {n} input IDs.")
            labels = tokens.copy()
            labels[:8] = [-100] * 8
            stream.write(json.dumps({"input_ids": tokens, "labels": labels}, separators=(",", ":")) + "\n")
    logger.info("Output SHA-256 %s", hashlib.sha256(args.output.read_bytes()).hexdigest())
    logger.info("Wrote %d documents and %d unpadded token positions", len(lengths), sum(lengths))


def architecture(args: argparse.Namespace) -> None:
    """Write metadata for the 256K training run, leaving the official checkpoint untouched."""
    path = args.source / "config.json"
    cfg = json.loads(path.read_text())
    if cfg.get("model_type") != "ernie5_5_moe" or cfg.get("hca_rope_type") != "yarn":
        raise ValueError("Expected the official ERNIE-5.5-Flash-Base HF architecture.")
    cfg.update(
        fp8=None,
        max_position_embeddings=262144,
        model_type="ernie5_5",
        dsa_indexer_loss_coeff=0.0,
        mqa_latent_rope_adjacent_pairing=True,
        rotary_scaling_factor=cfg["rope_scaling"]["factor"],
        original_max_position_embeddings=cfg["rope_scaling"]["original_max_position_embeddings"],
    )
    del cfg["quantization_config"]
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def convert(args: argparse.Namespace) -> None:
    """Convert the official checkpoint on four GPUs using Bridge's registered FP8 importer."""
    import torch
    import torch.distributed as dist

    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.ernie import ernie55_sft_64gpu_gb300_bf16_config

    if args.output.exists():
        raise ValueError(f"Provide a new checkpoint path: {args.output}")
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise ValueError("Run the EP4 conversion with four GPU processes.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    cfg = ernie55_sft_64gpu_gb300_bf16_config(hf_path=str(args.source))
    model = cfg.model
    model.context_parallel_size = 1
    model.expert_model_parallel_size = 4
    model.seq_length = 4096
    model.recompute_granularity = model.recompute_method = model.recompute_num_layers = None
    cfg.mixed_precision.setup(model)
    bridge = AutoBridge.from_hf_pretrained(str(args.source), torch_dtype=torch.bfloat16)
    stages = bridge.get_model(model, wrap_with_ddp=False)
    bridge.save_megatron_model(stages, str(args.output), hf_tokenizer_path=str(args.source))
    dist.barrier()
    dist.destroy_process_group()


def train(args: list[str]) -> None:
    """Launch the ordinary Bridge training CLI and record the local allocator peaks at exit."""
    import torch

    if dest := os.getenv("ERNIE55_COMPARE_HEADS"):
        from head_parity_256k import install

        install(Path(dest))
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 64:
        raise ValueError("This benchmark expects 64 GPUs.")
    # torchrun exposes every local GPU. Save the rank's device explicitly for the exit report.
    local_rank = int(os.environ["LOCAL_RANK"])

    def memory() -> None:
        if torch.cuda.is_initialized():
            allocated = torch.cuda.max_memory_allocated(local_rank)
            reserved = torch.cuda.max_memory_reserved(local_rank)
            logger.warning("ERNIE256K_PEAK rank=%d allocated=%d reserved=%d", rank, allocated, reserved)

    atexit.register(memory)
    script = Path(__file__).resolve().parents[3] / "scripts/training/run_recipe.py"
    sys.path.insert(0, str(script.parent))
    sys.argv = [
        str(script),
        "--recipe",
        "ernie55_pretrain_256k_64gpu_gb300_bf16_config",
        "--mode",
        "pretrain",
        "--step_func",
        "ernie55_step",
        *args,
    ]
    runpy.run_path(str(script), run_name="__main__")


def summarize(args: argparse.Namespace) -> None:
    """Require one complete time window and all 64 allocator reports for a single arm."""
    steps = {}
    peaks = {}
    for file in args.logs:
        with file.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                m = re.search(r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)", line)
                if m:
                    step, ms = int(m[1]), float(m[2])
                    if step in steps and steps[step] != ms:
                        raise ValueError(f"Mixed reports for step {step}; use a single run.")
                    steps[step] = ms
                m = re.search(r"ERNIE256K_PEAK rank=(\d+) allocated=(\d+) reserved=(\d+)", line)
                if m:
                    rank, allocated, reserved = map(int, m.groups())
                    if rank in peaks and peaks[rank] != (allocated, reserved):
                        raise ValueError(f"Mixed memory reports for rank {rank}.")
                    peaks[rank] = allocated, reserved
    if set(peaks) != set(range(64)):
        raise ValueError(f"Incomplete run: found peaks for {len(peaks)}/64 ranks.")
    window = range(args.first_step, args.last_step + 1)
    if not window or any(i not in steps or steps[i] <= 0 for i in window):
        raise ValueError("The specified interval must contain every requested optimizer update.")
    wall = sum(steps[i] for i in window) / 1000
    out = {
        "first_step": window.start,
        "last_step": window.stop - 1,
        "step_ms": {i: steps[i] for i in window},
        "physical_tokens_per_second_per_gpu": len(window) * 32 * 262144 / (64 * wall),
        "allocated_gib_max": max(p[0] for p in peaks.values()) / 2**30,
        "reserved_gib_max": max(p[1] for p in peaks.values()) / 2**30,
    }
    sys.stdout.write(json.dumps(out, indent=2) + "\n")


def main() -> None:
    """Dispatch the documented benchmark commands."""
    logging.basicConfig(level=logging.INFO)
    if sys.argv[1:2] == ["train"]:
        train(sys.argv[2:])
        return
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("layout", choices=LENGTHS)
    make.add_argument("output", type=Path)
    make.add_argument("--seed", type=int, default=42)
    make.add_argument(
        "--source", type=Path, help="JSONL source with one sufficiently long input_ids array per document."
    )
    arch = sub.add_parser("architecture")
    arch.add_argument("source", type=Path)
    arch.add_argument("output", type=Path)
    ckpt = sub.add_parser("convert")
    ckpt.add_argument("source", type=Path)
    ckpt.add_argument("output", type=Path)
    score = sub.add_parser("summarize")
    score.add_argument("logs", type=Path, nargs="+")
    score.add_argument("--first-step", type=int, default=4)
    score.add_argument("--last-step", type=int, default=12)
    args = p.parse_args()
    {"architecture": architecture, "prepare": prepare, "convert": convert, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
