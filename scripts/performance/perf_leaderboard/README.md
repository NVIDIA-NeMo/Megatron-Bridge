# Nemotron 3 Ultra deterministic perf leaderboard

A tiny harness for "what does turning on bit-exact determinism cost?" on the
Nemotron 3 Ultra (550B-A55B) GB200 recipe.

## Shared artifacts (world-readable)

All nsys-rep / sqlite / CSV / leaderboard / training-log files for the
24-node (bit-exact) and 48-node (bit-wise mismatch) runs are mirrored to:

```
/lustre/share/coreai_dlalgo_llm/zhiyul/nemotron-3-ultra-nsys-compare/
├── README.md         ← layout + headline summary
├── 24n-baseline/{processed,raw/{det,nondet,det-bitwise}}/
└── 48n-mismatch/{processed,raw/{det,nondet,det-bitwise}}/
```

Total ≈ 1.2 GB. Use this path when sharing results outside the
`coreai_dlalgo_llm` group — the source `/lustre/fsw/coreai_dlalgo_llm/zhiyul/`
parent has `--S---` perms and is unreadable to non-group users.

## Files

| File | What it is |
|---|---|
| [`../launch_nemotron_3_ultra_deterministic.sh`](../launch_nemotron_3_ultra_deterministic.sh) | Single-run launcher; the bit-exact-reproduced recipe (jobs 2074557 / 2074641 / 2074651 / 2076499 / 2076503 / 2102770 / 2103151). |
| [`../launch_nemotron_3_ultra_nsys_compare.sh`](../launch_nemotron_3_ultra_nsys_compare.sh) | End-to-end harness: submits one det + one non-det run with nsys, waits, then renders the leaderboard. Recipe block is byte-aligned with the deterministic launcher. |
| `print_nsys_leaderboard.py` | Forward / backward / op buckets, ranked by `abs(det − nondet)`. Copy of upstream PR #5041. |
| `extract_nsys_csv.py` | SQLite → `nsys stats -r nvtx_sum --format csv` fallback when `nsys` isn't on PATH. |

## Quick start

```bash
# Env (same set as launch_nemotron_3_ultra_deterministic.sh)
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export ACCOUNT=coreai_dlalgo_llm
export PARTITION=gb200
export CONTAINER_IMAGE=/path/to/nemo-26.04.01.squashfs
export REPO_ROOT="$PWD"
export HF_CACHE=/lustre/.../hf_cache

# Optional: override if `python` doesn't have nemo_run
export PYTHON=/path/to/venv/bin/python
export OUT_DIR=./nsys-compare-$(date +%s)

bash scripts/performance/launch_nemotron_3_ultra_nsys_compare.sh
```

`$OUT_DIR/leaderboard.txt` is written when both 24-node jobs finish (~15 min
each + queue wait).

## What gets toggled between the two runs

The compare script only changes the **6 determinism knobs** — everything else
(alltoall dispatcher, DDP overlap ON, attention=fused, parallelism, batch sizes,
recompute, manual_gc) is identical to `launch_nemotron_3_ultra_deterministic.sh`.

| Knob | det | non-det |
|---|---|---|
| `NCCL_ALGO` | `Ring` | unset (default Tree) |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | unset (default 1) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | unset |
| `MAMBA_DETERMINISTIC` | `1` | unset |
| `model.deterministic_mode` | `true` | `false` |
| `model.cross_entropy_loss_fusion` | `false` | `true` |

## Output layout

```
$OUT_DIR/
├── submit-det.log      submit-nondet.log     (nemo_run setup_experiment output)
├── jobid-det.txt       jobid-nondet.txt      (Slurm job ID for each side)
├── wdj-det.txt         wdj-nondet.txt        (wandb job names)
├── nsys-det.csv        nsys-nondet.csv       (nvtx_sum CSVs, rank 0)
└── leaderboard.txt     (top-20 by abs(det − nondet) in 3 buckets: forward / backward / op)
```

## Bit-wise determinism check (optional, separate)

Submit a 2nd det run (no nsys) and diff its iter-50 lm-loss / mtp / grad-norm
against any earlier paired det run (e.g. 2102770 or 2103151):

```bash
# Use launch_nemotron_3_ultra_deterministic.sh with a unique WANDB_JOB_NAME.
# Then compare:
strip='s/^ \[[^]]+\] //; s/elapsed time per iteration \(ms\): [0-9.]+ \| //; s/throughput per GPU \(TFLOP\/s\/GPU\): [0-9.]+ \| //'
diff <(grep " iteration " run_a.log | sed -E "$strip") \
     <(grep " iteration " run_b.log | sed -E "$strip")
# Empty diff → bit-exact reproducible.
```

## Reference

- Upstream tool: [NVIDIA/Megatron-LM PR #5041 — `run_nsys_breakdown.sh`](https://github.com/NVIDIA/Megatron-LM/pull/5041)
