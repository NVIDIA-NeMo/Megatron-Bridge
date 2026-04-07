"""Benchmark generation latency for NemotronDiffusion models.

Usage examples:

# Benchmark ministral_3b in dLLM mode (load from HF weights):
python benchmark_latency.py \
    --hf-model-id mistralai/Ministral-3B-Instruct-2410 \
    --mode dllm \
    --prompt-len 128 \
    --gen-len 128 \
    --block-length 32 \
    --num-warmup 2 \
    --num-runs 5

# Benchmark ministral_8b in AR mode (load from HF weights):
python benchmark_latency.py \
    --hf-model-id mistralai/Ministral-8B-Instruct-2410 \
    --mode ar \
    --prompt-len 128 \
    --gen-len 128 \
    --num-warmup 2 \
    --num-runs 5

# Benchmark ministral_8b from a Megatron checkpoint in dLLM mode:
python benchmark_latency.py \
    --hf-model-id mistralai/Ministral-8B-Instruct-2410 \
    --megatron-load-path /checkpoints/nemotron_diffusion_8b \
    --mode dllm \
    --prompt-len 128 \
    --gen-len 128 \
    --block-length 32 \
    --diffusion-steps 128 \
    --num-warmup 2 \
    --num-runs 5 \
    --output-json results_8b.json

# Benchmark ministral_cascade (two-model cascade) in dLLM mode:
#   Format: ckpt1|n_steps1|hf_id1|ckpt2|n_steps2|hf_id2
#   n_steps must sum to steps_per_block = diffusion-steps // (gen-len // block-length)
python benchmark_latency.py \
    --hf-model-id mistralai/Ministral-8B-Instruct-2410 \
    --megatron-load-path /checkpoints/nemotron_diffusion_8b \
    --mode dllm \
    --prompt-len 128 \
    --gen-len 128 \
    --block-length 32 \
    --diffusion-steps 128 \
    --cascade-schedule "/ckpt/draft|24|mistralai/Ministral-3B-Instruct-2410|/ckpt/verifier|8|mistralai/Ministral-8B-Instruct-2410" \
    --num-warmup 2 \
    --num-runs 5
"""

import argparse
import json
import math
import time
from typing import Optional, List, Tuple

import torch

# Register NemotronDiffusion bridge so AutoBridge can find it
import megatron.bridge.diffusion.conversion.nemotron_diffusion.nemotron_diffusion_bridge  # noqa: F401

from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import build_and_load_model

from diffusion.recipes.nemotron_diffusion.inference_nemotron_diffusion import (
    generate_ar,
    generate_dllm,
    _model_forward,
    _set_inference_mode,
    _set_inference_params,
    _clear_kv_cache,
)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model(
    hf_model_id: str,
    megatron_load_path: Optional[str],
    max_sequence_length: int,
) -> torch.nn.Module:
    """Load a single NemotronDiffusion model, either from HF weights or a Megatron checkpoint."""
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model_provider = bridge.to_megatron_provider(load_weights=(megatron_load_path is None))
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.seq_length = max_sequence_length
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    if megatron_load_path is not None:
        megatron_models = build_and_load_model(
            checkpoint_path=megatron_load_path,
            model_cfg=model_provider,
            skip_temp_dist_context=True,
            dist_ckpt_strictness="ignore_all",
        )
    else:
        megatron_models = model_provider.provide_distributed_model(wrap_with_ddp=False)

    if isinstance(megatron_models, list):
        model = megatron_models[0]
    else:
        model = megatron_models

    return model.cuda().eval()


def parse_cascade_schedule(
    cascade_str: str,
    max_sequence_length: int,
) -> List[Tuple[torch.nn.Module, int]]:
    """Parse '--cascade-schedule' string into a list of (model, n_steps) tuples.

    Format: ckpt1|n_steps1|hf_id1|ckpt2|n_steps2|hf_id2|...
    Each triplet defines one stage: checkpoint path (or empty string for HF),
    number of denoising steps, and HF model ID.
    """
    parts = cascade_str.split("|")
    if len(parts) % 3 != 0:
        raise ValueError(
            "--cascade-schedule must have groups of 3 fields: ckpt|n_steps|hf_id"
        )
    schedule = []
    for i in range(0, len(parts), 3):
        ckpt = parts[i].strip() or None
        n_steps = int(parts[i + 1].strip())
        hf_id = parts[i + 2].strip()
        print(f"  Loading cascade stage: hf_id={hf_id}, ckpt={ckpt}, n_steps={n_steps}")
        stage_model = load_model(hf_id, ckpt, max_sequence_length)
        schedule.append((stage_model, n_steps))
    return schedule


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(values: List[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _fmt(s: dict, scale: float = 1.0, unit: str = "") -> str:
    return (
        f"{s['mean'] * scale:.3f} ± {s['std'] * scale:.3f} "
        f"[min={s['min'] * scale:.3f}, max={s['max'] * scale:.3f}]{unit}"
    )


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_ar(
    model: torch.nn.Module,
    prompt_len: int,
    gen_len: int,
    temperature: float,
    num_warmup: int,
    num_runs: int,
    batch_size: int = 1,
) -> dict:
    """Benchmark AR generation, including TTFT measurement."""
    results = {
        "total_time_s": [],
        "time_per_token_ms": [],
        "tokens_per_second": [],
        "ttft_ms": [],
    }

    for run_idx in range(num_warmup + num_runs):
        prompt = torch.randint(1, 1000, (batch_size, prompt_len), dtype=torch.long, device="cuda")

        # --- Measure TTFT (prefill only) ---
        _set_inference_mode(model, True)
        _set_inference_params(model, causal=True, cache_enabled=True)
        _clear_kv_cache(model)

        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        _ = _model_forward(model, prompt)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t_prefill_start

        _set_inference_mode(model, False)

        # --- Measure full generation (prefill + decode) ---
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        _ = generate_ar(
            model,
            prompt,
            max_new_tokens=gen_len,
            temperature=temperature,
            eos_token_id=None,
        )
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t_start

        if run_idx >= num_warmup:
            results["total_time_s"].append(total_time)
            results["time_per_token_ms"].append(total_time / (batch_size * gen_len) * 1000.0)
            results["tokens_per_second"].append(batch_size * gen_len / total_time)
            results["ttft_ms"].append(ttft * 1000.0)

    return results


@torch.no_grad()
def benchmark_dllm(
    model: torch.nn.Module,
    prompt_len: int,
    gen_len: int,
    block_length: int,
    diffusion_steps: int,
    temperature: float,
    mask_token_id: int,
    num_warmup: int,
    num_runs: int,
    model_schedule: Optional[List[Tuple[torch.nn.Module, int]]] = None,
    batch_size: int = 1,
) -> dict:
    """Benchmark dLLM (block diffusion) generation."""
    results = {
        "total_time_s": [],
        "time_per_token_ms": [],
        "tokens_per_second": [],
        "nfe": [],
    }

    for run_idx in range(num_warmup + num_runs):
        prompt = torch.randint(1, 1000, (batch_size, prompt_len), dtype=torch.long, device="cuda")

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        _, nfe = generate_dllm(
            model,
            prompt,
            gen_length=gen_len,
            block_length=block_length,
            steps=diffusion_steps,
            temperature=temperature,
            remasking="low_confidence",
            mask_id=mask_token_id,
            threshold=None,
            shift_logits=True,
            neg_entropy=True,
            model_schedule=model_schedule,
        )
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t_start

        if run_idx >= num_warmup:
            results["total_time_s"].append(total_time)
            results["time_per_token_ms"].append(total_time / (batch_size * gen_len) * 1000.0)
            results["tokens_per_second"].append(batch_size * gen_len / total_time)
            results["nfe"].append(float(nfe))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(mode: str, args: argparse.Namespace, raw: dict) -> None:
    sep = "=" * 72
    print(sep)
    print(f"  NemotronDiffusion Latency Benchmark  |  mode={mode.upper()}")
    print(f"  HF model : {args.hf_model_id}")
    if args.megatron_load_path:
        print(f"  Megatron : {args.megatron_load_path}")
    print(f"  prompt_len={args.prompt_len}  gen_len={args.gen_len}  batch_size={args.batch_size}  "
          f"temperature={args.temperature}")
    if mode == "dllm":
        print(f"  block_length={args.block_length}  diffusion_steps={args.diffusion_steps}")
        if args.cascade_schedule:
            print(f"  cascade_schedule={args.cascade_schedule}")
    print(f"  warmup={args.num_warmup}  runs={args.num_runs}")
    print(sep)

    total = _stats(raw["total_time_s"])
    tpt = _stats(raw["time_per_token_ms"])
    tps = _stats(raw["tokens_per_second"])

    print(f"  Total time (s)       : {_fmt(total)}")
    print(f"  Time per token (ms, incl. prefill) : {_fmt(tpt)}")
    print(f"  Tokens per second    : {_fmt(tps)}")

    if mode == "ar" and "ttft_ms" in raw:
        ttft = _stats(raw["ttft_ms"])
        print(f"  TTFT (ms)            : {_fmt(ttft)}")

    if mode == "dllm" and "nfe" in raw:
        nfe = _stats(raw["nfe"])
        print(f"  NFE                  : {_fmt(nfe)}")

    print(sep)


def build_json_output(mode: str, args: argparse.Namespace, raw: dict) -> dict:
    out = {
        "mode": mode,
        "hf_model_id": args.hf_model_id,
        "megatron_load_path": args.megatron_load_path,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "temperature": args.temperature,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "metrics": {},
    }
    if mode == "dllm":
        out["block_length"] = args.block_length
        out["diffusion_steps"] = args.diffusion_steps
        out["cascade_schedule"] = args.cascade_schedule

    for key, values in raw.items():
        out["metrics"][key] = _stats(values)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NemotronDiffusion generation latency (single GPU, TP=1, PP=1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-model-id",
        required=True,
        help="HuggingFace model ID or local path for the primary model.",
    )
    parser.add_argument(
        "--megatron-load-path",
        default=None,
        help="Optional path to a Megatron checkpoint. If not set, loads HF weights directly.",
    )
    parser.add_argument(
        "--mode",
        choices=["ar", "dllm"],
        default="dllm",
        help="Generation mode: autoregressive ('ar') or block diffusion ('dllm').",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Prompt length in tokens (random tokens in [1, 1000)).",
    )
    parser.add_argument(
        "--gen-len",
        type=int,
        default=128,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="dLLM block length (only used in dllm mode).",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Total denoising steps (only used in dllm mode). Defaults to gen-len.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=100,
        help="Mask token ID used in dLLM mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (excluded from statistics).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4096,
        help="Maximum sequence length for the model.",
    )
    parser.add_argument(
        "--cascade-schedule",
        default=None,
        help=(
            "Optional cascade model schedule (dllm mode only). "
            "Format: ckpt1|n_steps1|hf_id1|ckpt2|n_steps2|hf_id2|... "
            "The n_steps values must sum to steps_per_block = diffusion_steps // (gen_len // block_length). "
            "Use an empty string for ckpt to load from HF weights."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional file path to write benchmark results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve default diffusion steps
    if args.diffusion_steps is None:
        args.diffusion_steps = args.gen_len

    print(f"Loading primary model: {args.hf_model_id}")
    model = load_model(
        hf_model_id=args.hf_model_id,
        megatron_load_path=args.megatron_load_path,
        max_sequence_length=args.max_sequence_length,
    )
    print("Primary model loaded.")

    model_schedule = None
    if args.mode == "dllm" and args.cascade_schedule:
        print("Parsing cascade schedule...")
        model_schedule = parse_cascade_schedule(args.cascade_schedule, args.max_sequence_length)
        print(f"Cascade schedule: {[(type(m).__name__, s) for m, s in model_schedule]}")

    if args.mode == "ar":
        print(f"\nRunning AR benchmark ({args.num_warmup} warmup + {args.num_runs} runs)...")
        raw = benchmark_ar(
            model=model,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            temperature=args.temperature,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            batch_size=args.batch_size,
        )
    else:
        print(f"\nRunning dLLM benchmark ({args.num_warmup} warmup + {args.num_runs} runs)...")
        raw = benchmark_dllm(
            model=model,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            block_length=args.block_length,
            diffusion_steps=args.diffusion_steps,
            temperature=args.temperature,
            mask_token_id=args.mask_token_id,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            model_schedule=model_schedule,
            batch_size=args.batch_size,
        )

    print_table(args.mode, args, raw)

    if args.output_json:
        out = build_json_output(args.mode, args, raw)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
