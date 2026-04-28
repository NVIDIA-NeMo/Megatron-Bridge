"""
Full-model import/roundtrip test for DeepSeek-V4-Flash bridge.

Runs on 1 node × 8 H100 GPUs with TP=8 PP=1.
Phases:
  1. Init distributed + Megatron parallel state
  2. Build full provider from real HF config
  3. Instantiate the full Megatron model
  4. load_hf_weights() from real checkpoint
  5. Spot-check a few non-FP8 parameter values vs original HF tensors
  6. export_hf_weights() and compare shapes + values for the same sample
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist


MODEL_PATH = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/nemo_home/models/deepseek/DeepSeek-V4-Flash"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m-\033[0m"


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def is_rank0():
    return rank() == 0


def section(title):
    if is_rank0():
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}", flush=True)


def ok(msg):
    if is_rank0():
        print(f"  {PASS} {msg}", flush=True)


def fail(msg, exc=None):
    if is_rank0():
        print(f"  {FAIL} {msg}", flush=True)
        if exc:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Phase 1: Distributed init
# ---------------------------------------------------------------------------
section("Phase 1 — Distributed init (TP=8 PP=1)")
try:
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank_id = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank_id,
    )
    ok(f"dist.init_process_group: world_size={world_size} rank={rank_id}")

    from megatron.core import parallel_state, tensor_parallel

    tp_size = int(os.environ.get("TP_SIZE", "8"))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    ok(f"Megatron parallel state: TP={tp_size} PP=1 EP=1")
except Exception as e:
    fail("Distributed init failed", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 2: Provider config from real checkpoint
# ---------------------------------------------------------------------------
section("Phase 2 — Provider config (full model)")
try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    bridge = AutoBridge.from_hf_pretrained(MODEL_PATH)
    hf_pretrained = PreTrainedCausalLM(MODEL_PATH)
    provider = bridge._model_bridge.provider_bridge(hf_pretrained)

    provider.tensor_model_parallel_size = tp_size
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.finalize()

    ok(f"model_type: {bridge.hf_pretrained.config.model_type}")
    ok(f"num_layers: {provider.num_layers}")
    ok(f"mtp_num_layers: {provider.mtp_num_layers}")
    ok(f"csa_compress_ratios[:6]: {provider.csa_compress_ratios[:6]}")
    ok(f"TP={provider.tensor_model_parallel_size}")
except Exception as e:
    fail("Provider config failed", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 3: Instantiate full model
# ---------------------------------------------------------------------------
section("Phase 3 — Model init (full, TP=8)")
model = None
try:
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    num_params = sum(p.numel() for p in model[0].parameters())
    ok(f"Model instantiated: {type(model[0]).__name__}")
    ok(f"Parameters on this rank: {num_params:,}")
except Exception as e:
    fail("Model init failed", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 4: Load HF weights
# ---------------------------------------------------------------------------
section("Phase 4 — load_hf_weights() from real checkpoint")
try:
    # Pre-load a small sample of original HF weights on rank 0 for later comparison
    orig_sample = {}
    if is_rank0():
        from safetensors import safe_open

        shard0 = os.path.join(MODEL_PATH, "model-00001-of-00046.safetensors")
        with safe_open(shard0, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # Grab a few non-FP8 tensors for roundtrip check
            non_fp8_keys = [k for k in keys if f.get_tensor(k).dtype != torch.float8_e4m3fn][:5]
            for k in non_fp8_keys:
                orig_sample[k] = f.get_tensor(k).clone()
        ok(f"Pre-loaded {len(orig_sample)} non-FP8 reference tensors from shard0")
        for k, v in orig_sample.items():
            ok(f"  orig {k!r}: shape={tuple(v.shape)} dtype={v.dtype}")

    bridge.load_hf_weights(model)
    ok("load_hf_weights() completed")

    # Spot-check a few values are non-zero
    for name, param in list(model[0].named_parameters())[:5]:
        std = param.float().std().item()
        ok(f"  {name}: std={std:.4f}  shape={tuple(param.shape)}")
except Exception as e:
    fail("load_hf_weights() failed", e)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 5+6: Export + roundtrip check
# ---------------------------------------------------------------------------
section("Phase 5+6 — export_hf_weights() + roundtrip check")

# DIAGNOSTIC: count expert-param tasks per TP rank before export
try:
    _diag_tasks = bridge._model_bridge.build_conversion_tasks(bridge.hf_pretrained, model)
    _expert_names = [
        t.global_param_name
        for t in _diag_tasks
        if t and hasattr(t, "mapping") and getattr(t.mapping, "is_expert", False)
    ]
    _expert_count = len(_expert_names)
    print(
        f"[rank {dist.get_rank()}] expert tasks: {_expert_count}  last5={_expert_names[-5:] if _expert_names else []}",
        flush=True,
    )
    _count_t = torch.tensor([_expert_count], dtype=torch.int64, device="cuda")
    _all_c = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(dist.get_world_size())]
    dist.all_gather(_all_c, _count_t)
    if is_rank0():
        counts = [c.item() for c in _all_c]
        print(f"[diag] expert_task_counts={counts}", flush=True)
        if len(set(counts)) > 1:
            print("[diag] WARNING: asymmetric task lists!", flush=True)
except Exception as _de:
    print(f"[rank {dist.get_rank()}] diag error: {_de}", flush=True)
dist.barrier()

# Free any unused GPU memory before the export to avoid NCCL CUDA OOM
torch.cuda.empty_cache()

try:
    exported = {}
    for hf_key, tensor in bridge.export_hf_weights(model, cpu=True):
        exported[hf_key] = tensor
    ok(f"export_hf_weights() produced {len(exported)} tensors")

    if is_rank0() and orig_sample:
        shape_ok = shape_fail = 0
        for hf_key, orig in orig_sample.items():
            if hf_key not in exported:
                fail(f"  MISSING in export: {hf_key!r}")
                shape_fail += 1
                continue
            exp = exported[hf_key]
            if exp.shape == orig.shape:
                shape_ok += 1
            else:
                fail(f"  Shape mismatch {hf_key!r}: orig={tuple(orig.shape)} exp={tuple(exp.shape)}")
                shape_fail += 1

        ok(f"Shape checks: {shape_ok} passed, {shape_fail} failed")

        # Value roundtrip
        for hf_key, orig in orig_sample.items():
            if hf_key not in exported:
                continue
            orig_f = orig.float()
            exp_f = exported[hf_key].float()
            max_diff = (orig_f - exp_f).abs().max().item()
            mean_diff = (orig_f - exp_f).abs().mean().item()
            ok(f"  Roundtrip {hf_key!r}: max_diff={max_diff:.3e}  mean_diff={mean_diff:.3e}")
except Exception as e:
    fail("Export/roundtrip failed", e)
    traceback.print_exc()
    # Hard abort: prevents rank 0 from hanging in a collective
    # if this rank failed mid-export (e.g. NCCL OOM).
    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(1)

section("Done")
dist.destroy_process_group()
