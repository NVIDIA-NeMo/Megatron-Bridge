# TP vs DP Numerical Equivalence Analysis

Analysis of numerical differences between Tensor Parallel (TP) and Data Parallel (DP)
configurations in Megatron-Core transformer blocks.

**Setup**: 28-layer GPT transformer, hidden_size=1024, 16 attention heads, seq_len=4096,
batch_size=2, unfused attention, deterministic mode, H100 (sm90).

## 1. Weight Initialization: Bit-Identical

With `use_cpu_initialization=True` and same `torch.manual_seed(12345)`:

- `_initialize_affine_weight_cpu` creates the **full** master weight on CPU, calls
  `init_method`, then slices for each TP rank
- TE layers receive `init_method=lambda w: None` (no-op), then Megatron overwrites
  via `_initialize_affine_weight_cpu`
- LayerNorm params are deterministic (gamma=1, beta=0)
- Biases are explicitly zeroed

**Verified**: all 26 parameters across TP=1, TP=2, TP=4 have **max diff = 0.0** after
gathering TP shards.

## 2. Source of Forward Pass Differences

Each transformer layer has two types of TP-sharded linear layers:

- **Column-parallel** (QKV projection, MLP fc1): splits the **output** dimension.
  Each rank computes different output elements independently. **No all-reduce.**
- **Row-parallel** (attention output projection, MLP fc2): splits the **reduction**
  dimension. Each rank computes partial sums of the **same** output elements,
  then combines via `all_reduce(SUM)`.

Only row-parallel introduces divergence because the all-reduce changes the
floating-point summation order:

- **TP=1**: `Y = X @ W` — single matmul, accumulates over full K=4096
- **TP=4**: `Y = allreduce(X_0 @ W_0 + X_1 @ W_1 + X_2 @ W_2 + X_3 @ W_3)` —
  four partial matmuls over K=1024 each, then summed

Floating-point addition is **non-associative** — different groupings produce different
rounding. Column-parallel is **bit-exact** because each output element is computed
by exactly one rank with the same sub-matrix.

### 2.1 Isolated Proof: Column-Parallel vs Row-Parallel

**Column-parallel matmul** (TP=1 vs TP=4, bf16):
```
Max abs diff:  0.0         ← bit-exact, no all-reduce
Mean abs diff: 0.0
```

**Single row-parallel matmul** (TP=1 vs TP=4, bf16, input=[4096,2,4096], weight=[1024,4096]):
```
Max abs diff:    0.031250
Mean abs diff:   0.00195487
Median abs diff: 0.00006104
Output mean:     0.000639   (outputs are O(1) scale)
Output std:      0.992402

% > 1e-4:  50.0%
% > 1e-3:  41.6%
% > 5e-3:   8.1%
% > 1e-2:   1.3%
```

A single all-reduce introduces up to 0.03125 (1 bf16 ULP at scale ~1.0) max
difference, with roughly half the elements differing by more than 1e-4.

### 2.2 Error Compounding Across Stacked MLP Layers

Stacking column→row pairs (no LayerNorm, no attention — pure MLP) shows the
all-reduce error compounding. Each layer is: `h = (h @ W_col.t()) @ W_row.t()`
where column-parallel is exact and row-parallel introduces the error.

```
Layer |        Max |         Mean |       Median |   %>1e-4 |   %>1e-3 |   %>1e-2
------+------------+--------------+--------------+----------+----------+---------
    0 |   0.031250 |   0.00194148 |   0.00007629 |   49.99% |   41.61% |    1.25%
    1 |   0.031250 |   0.00382236 |   0.00390625 |   69.86% |   66.34% |    4.46%
    2 |   0.039062 |   0.00497189 |   0.00390625 |   75.80% |   73.12% |   10.66%
    3 |   0.046875 |   0.00585712 |   0.00488281 |   79.12% |   76.84% |   16.50%
    4 |   0.046875 |   0.00659324 |   0.00585938 |   81.36% |   79.32% |   21.47%
    5 |   0.062500 |   0.00719865 |   0.00781250 |   83.01% |   81.11% |   25.40%
    9 |   0.062500 |   0.00895941 |   0.00781250 |   86.81% |   85.19% |   35.57%
   13 |   0.072266 |   0.01010600 |   0.00781250 |   88.87% |   87.35% |   41.16%
   15 |   0.078125 |   0.01049999 |   0.00781250 |   89.61% |   88.09% |   42.92%
```

After 16 MLP layers: **mean diff grows 5.4x**, max diff grows 2.5x.
The median stabilizes around 1 bf16 ULP (0.0078) after a few layers.

See ``test_row_parallel_allreduce_divergence.py`` for the runnable proof.

## 3. Per-Layer Error Growth: bf16

DP (TP=1) vs TP=4, bf16:

```
Layer | Max Abs Diff | Mean Abs Diff |   %>1e-3 |   %>1e-2
------+--------------+---------------+----------+---------
    0 |     0.031250 |    0.00014072 |    2.97% |    0.04%
    1 |     0.031250 |    0.00031284 |    6.80% |    0.11%
    2 |     0.031250 |    0.00049002 |   10.83% |    0.21%
    3 |     0.031250 |    0.00066876 |   14.85% |    0.33%
    4 |     0.062500 |    0.00084779 |   18.73% |    0.48%
    9 |     0.062500 |    0.00173278 |   34.92% |    1.72%
   13 |     0.078125 |    0.00241730 |   44.41% |    3.24%
   19 |     0.093750 |    0.00340257 |   54.82% |    6.32%
   27 |     0.109375 |    0.00463839 |   64.14% |   11.39%
------+--------------+---------------+----------+---------
final |     0.093750 |    0.00447368 |   64.03% |   10.51%
```

Mean abs diff grows ~linearly at ~0.00017/layer. Max abs diff steps up at bf16
quantization boundaries (0.03125, 0.0625, 0.078125, 0.09375, 0.109375).

## 4. Per-Layer Error Growth: fp32

DP (TP=1) vs TP=4, fp32 (same model, same input):

```
Layer | Max Abs Diff | Mean Abs Diff |   %>1e-3 |   %>1e-2
------+--------------+---------------+----------+---------
    0 |     0.000044 |    0.00000046 |    0.00% |    0.00%
    1 |     0.000068 |    0.00000383 |    0.00% |    0.00%
    4 |     0.000143 |    0.00001659 |    0.00% |    0.00%
    9 |     0.000262 |    0.00003254 |    0.00% |    0.00%
   13 |     0.000328 |    0.00004308 |    0.00% |    0.00%
   19 |     0.000369 |    0.00005704 |    0.00% |    0.00%
   27 |     0.000547 |    0.00007358 |    0.00% |    0.00%
------+--------------+---------------+----------+---------
final |     0.000495 |    0.00006948 |    0.00% |    0.00%
```

fp32 max diff after 28 layers: **0.000547** (well within atol=1e-3).
Zero elements exceed 0.001 threshold.

## 5. bf16 vs fp32 Comparison

| Metric (after 28 layers)      | bf16         | fp32         | Ratio    |
|-------------------------------|--------------|--------------|----------|
| Max absolute diff             | 0.1094       | 0.0005       | ~200x    |
| Mean absolute diff            | 0.0046       | 0.00007      | ~66x     |
| % elements > 0.001            | 64.14%       | 0.00%        | —        |
| % elements > 0.01             | 11.39%       | 0.00%        | —        |

The ~200x ratio in max diff matches the precision ratio: bf16 mantissa is 7 bits
(~3 decimal digits), fp32 mantissa is 23 bits (~7 decimal digits) → ~10^4 precision
difference, compounded non-linearly over 28 layers.

## 6. Typical Output Values: TP=1 vs TP=4

**bf16** (pos 0, batch 0, first 5 hidden dims):
```
TP=1: [ 0.300781, -0.812500, -0.496094, -0.192383, -1.460938]
TP=4: [ 0.296875, -0.816406, -0.503906, -0.199219, -1.468750]
diff: [ 0.003906,  0.003906,  0.007812,  0.006836,  0.007812]
```

**fp32** (pos 0, batch 0, first 5 hidden dims):
```
TP=1: [ 0.296995, -0.813083, -0.494352, -0.191834, -1.419001]
TP=4: [ 0.297103, -0.813083, -0.494522, -0.191901, -1.419005]
diff: [-0.000108, -0.000001,  0.000170,  0.000067,  0.000004]
```

Output values are O(1) (mean≈0, std≈1 after LayerNorm). bf16 per-element differences
are ~0.004-0.008 (last 1-2 significant digits). fp32 differences are ~0.0001.

## 7. Test Tolerances

| Test                              | dtype  | Layers | atol   | rtol   | Status |
|-----------------------------------|--------|--------|--------|--------|--------|
| `test_tp_vs_dp_fp32_equivalence`  | fp32   | 1      | 1e-3   | 1e-3   | PASS   |
| `test_tp_vs_dp_equivalence`       | bf16   | 28     | 0.15   | 0.0    | PASS   |

The bf16 test uses `atol=0.15` to accommodate the ~0.11 max diff observed across
28 layers. This is not a bug — it is the expected precision limit of bf16 tensor
parallelism.

## 8. Environment Notes

`Utils.initialize_distributed()` and `Utils.destroy_model_parallel()` in the test
utilities **pop** `NVTE_FLASH_ATTN`, `NVTE_FUSED_ATTN`, and `NVTE_UNFUSED_ATTN`
from `os.environ`. These must be re-set **after** calling those functions.

On Hopper (sm90), TE's default backend selection prefers FusedAttention even when
`NVTE_FUSED_ATTN=0` is set via shell — it must be set in-process after the pop.
Use `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1` to verify the selected backend.
