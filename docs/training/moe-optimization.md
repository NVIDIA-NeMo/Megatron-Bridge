# MoE Training Optimization

This page covers the optimization framework, techniques, and best practices for
training Mixture-of-Experts models with Megatron-Core. It is based on
[Scalable Training of MoE Models with Megatron Core](https://arxiv.org/abs/2603.07685).

For tuning knobs, hardware-specific configs, and benchmark-oriented guidance, see:

- [skills/perf-techniques/moe-optimization-workflow/SKILL.md](../skills/perf-techniques/moe-optimization-workflow/SKILL.md)
- [skills/perf-techniques/moe-dispatcher-selection/SKILL.md](../skills/perf-techniques/moe-dispatcher-selection/SKILL.md)
- [skills/perf-techniques/moe-long-context/SKILL.md](../skills/perf-techniques/moe-long-context/SKILL.md)
- [skills/perf-techniques/moe-hardware-configs/SKILL.md](../skills/perf-techniques/moe-hardware-configs/SKILL.md)
- [skills/perf-techniques/moe-vlm-training/SKILL.md](../skills/perf-techniques/moe-vlm-training/SKILL.md)
- [skills/perf-techniques/moe-comm-overlap/SKILL.md](../skills/perf-techniques/moe-comm-overlap/SKILL.md)

## The Three Walls

MoE training is constrained by three tightly coupled barriers:

| Wall | Root Cause | Metric |
|---|---|---|
| **Memory** | All E experts stored but only K active per token | GB per GPU |
| **Communication** | EP all-to-all dispatches tokens across GPUs | % of step time |
| **Compute Efficiency** | Small expert GEMMs + host overhead | GPU SM utilization |

These walls interact: solving one often exposes another. FP8 reduces memory and
can improve GEMM throughput, but it can also shift more of the remaining cost
to quantization kernels and host overhead. Communication overlap can hide
latency, but it may add scheduling and buffer constraints. Effective
optimization treats all three as a unified system.

## The Optimization Workflow

### Phase 1: Establish Memory-Feasible Parallelism

Memory is the hard gate — if it doesn't fit, nothing runs.

| Strategy | Activation | Weight | Optimizer | Comm |
|---|---|---|---|---|
| TP | 1/d (with SP) | 1/d | 1/d | High |
| EP | ~1 (load-dependent) | 1/d (MoE only) | 1/d | Medium |
| PP | 1 (>1 with VPP) | 1/d | 1/d | Medium |
| CP | 1/d | 1 | 1/d† | Medium |
| DP | 1 | 1 | 1/d† | Low |

†Requires `--use-distributed-optimizer`.

Quick test: use `--fake-init-process-group` to emulate distributed training on a
single GPU for rapid parallelism iteration before spending cluster time.

### Phase 2: Select Optimal Parallelism

Five guidelines, in order of priority:

1. **Minimize model parallelism, maximize DP.** Model parallelism adds
   communication; use distributed optimizer to free memory for larger DP.

2. **Keep EP × TP within the fast interconnect domain.** EP and TP are
   communication-intensive, so keeping the hot path on the fastest available
   links usually matters more than theoretical FLOPs.

3. **Use PP for multi-node scaling.** PP's point-to-point comms scale
   better across nodes than TP/EP. Enable VPP to reduce bubbles.

4. **Prefer EP over TP for expert layers.** EP gives better GEMM
   efficiency, lower communication, and eliminates local permutation
   when EP = num_experts. Use Parallel Folding to decouple attention TP
   from expert EP.

5. **Enable CP once sequence length makes attention memory dominant.** In
   practice that often starts around the 8K-class regime, but the exact point
   depends on model size and hardware. Use hierarchical CP (`a2a+p2p`) on
   NVL72-class systems when appropriate.

### Phase 3: Profile and Optimize Bottlenecks

Profile the training run and identify which wall dominates:

**Memory bottleneck** — forced into full recompute or excessive parallelism:

| Optimization | Overhead | Flag |
|---|---|---|
| FP8 training | Low | `--fp8-format --fp8-recipe` |
| Selective recompute | Low | `--recompute-granularity --recompute-modules` |
| Precision-aware optimizer | Low | `--use-precision-aware-optimizer` |
| Activation offloading | Medium | `--fine-grained-activation-offloading` |
| Optimizer offloading | Medium | `--offload-optimizer-states` |

**Communication bottleneck** — profiling shows time in collectives:

| Comm Type | Fix |
|---|---|
| DP grad/param | `--overlap-grad-reduce --overlap-param-gather` |
| TP | `--tp-comm-overlap` |
| EP dispatcher | `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend {deepep\|hybridep}` |
| EP all-to-all | `--overlap-moe-expert-parallel-comm` |
| PP send/recv | `--pipeline-model-parallel-layout` (flexible VPP) |

**CPU overhead** — gaps between GPU kernels in Nsight traces:

| Fix | Flag |
|---|---|
| Disable Python GC | `--manual-gc --manual-gc-interval 10` |
| CUDA Graphs | `--cuda-graph-impl transformer_engine` |
| Reduce kernel launches | Decrease TP or increase MBS |

**Compute inefficiency** — low SM utilization despite no comm/CPU issues:

| Fix | Flag |
|---|---|
| Grouped GEMM | `--moe-grouped-gemm` |
| Kernel fusions | `--moe-router-fusion --moe-permute-fusion` |
| FP8 precision | `--fp8-format --fp8-recipe` |

This process is **iterative**: fitting the model, choosing parallelism, and
profiling the dominant wall usually matter more than any single micro-optimization.

## Parallel Folding

Attention and MoE layers have conflicting optimal parallelisms.
Parallel Folding decouples their configurations.

```text
Attention layers: TP × CP × DP × PP
MoE layers:      ETP × EP × EDP × PP  (PP must match)
```

Key benefits:
- **Breaks EP ≤ DP constraint**: EP can "fold" across TP×CP groups
- **Independent optimization**: Attention uses high TP; MoE uses ETP=1
- **Fewer GPUs needed**: CP=8 and EP=8 share the same 8 GPUs
- **NVLink-local comms**: Both CP and EP stay in high-bandwidth domain

Example: 256 GPUs with attention TP=4, CP=2, DP=8, PP=4.
Traditional: EP ≤ DP = 8. With folding: EP=64, ETP=1, EDP=1.

## Memory Optimization Stack

Ordered by overhead (lowest first):

1. **Memory-efficient permutation** (zero overhead): Absorbs routing
   weights into activations before FC2, eliminating saved tensors for
   router backward.

   Standard: `y = Σ p_i · W2_i · φ(W1_i · x)`
   Memory-efficient: `y = Σ W2_i · (p_i · φ(W1_i · x))`

   Mathematically equivalent when experts have no bias. Eliminates saving
   each expert output for router backward — activation is recomputed from
   already-saved inputs.

2. **FP8/FP4 activations**: Store linear-layer inputs in lower precision than
   BF16. This usually gives a modest but useful activation-memory reduction.

3. **Fine-grained recompute**: Recompute only cheap operations such as
   LayerNorm, activation functions, or model-specific up-projection modules.
   This often recovers much of the needed memory while keeping overhead much
   lower than full-layer recompute.

4. **Fine-grained offloading**: Module-level D2H/H2D with stream overlap. This
   can free a meaningful amount of memory at a small throughput cost and may
   allow a better parallelism layout that more than repays the offload overhead.

5. **Optimizer state offloading**: Move optimizer states to CPU between steps.
   This is especially attractive on GB200-class systems, where the host-device
   path is strong enough to make the trade practical.

6. **FSDP for MoE**: Dual DeviceMesh — primary mesh for attention,
   expert mesh for MoE. AllGather/ReduceScatter stay within small
   EDP groups. Zero-copy comms via NCCL User Buffer Registration.

## FP8 Recipe Selection

| Recipe | Platform | Granularity | Recommended |
|---|---|---|---|
| Per-tensor FP8 | Hopper/Blackwell | 1 scale/tensor | Starting point |
| Blockwise FP8 | Hopper | 128×128 blocks | **Production on Hopper** |
| MXFP8 | Blackwell | 1×32 elements | **Default on Blackwell** |
| NVFP4 | Blackwell | 16 elements, 2-level | Maximum throughput |

Key rules:
- Router stays in FP32 always
- Embeddings, output layer, gradients, optimizer stay in original precision
- Expert GEMMs are the primary quantization target
- MXFP8 on Blackwell communicates params in BF16 (can't save on AllGather)
- NVFP4 requires Random Hadamard Transforms, 2D scaling, stochastic rounding

## CUDA Graphs for MoE

Two modes, different trade-offs:

| Mode | What's Captured | When to Use |
|---|---|---|
| Full CUDA Graphs | Entire fwd+bwd | Drop-and-pad MoE only |
| Partial (layer-wise) | attn + router + moe_preprocess | **Dropless MoE (default)** |

Partial CUDA graphs capture static components while leaving dynamic expert
computation outside the graph, which is why they are the safer default for
dropless MoE.

For full CUDA Graphs on dropless MoE, three techniques are needed:

- **Device-initiated Grouped GEMM**: Reads shapes from GPU memory.
  cuBLASLt (CUDA 13.1+) or cuteDSL with fused activation/quantization.

- **ECHO** (Elastic Cloning for Hot Experts): Clones hot experts to
  underutilized ranks via bin-packing. Reduces load variance so
  worst-case buffer sizing is closer to actual.

- **Paged Stashing**: Single worst-case tmp buffer shared across layers
  for computation; paged stashing buffer stores only actual tokens.
  Reduces memory from O(layers × worst_case) to O(worst_case + actual).
  64 tokens per page, free list via circular buffer.

## Flexible Asymmetric VPP

PP layout string controls per-stage layer distribution:

```bash
--pipeline-model-parallel-layout "Et*3|(tt|)*29m|L"
```

- `E` = embedding, `t` = transformer, `m` = MTP, `L` = loss, `|` = stage boundary
- Balance workload: embedding + N dense layers ≈ fewer MoE layers
- Place MTP and loss on dedicated stages for memory isolation

## EP Communication Overlap

For exact overlap constraints and verification guidance, see
[docs/training/communication-overlap.md](communication-overlap.md) and the
related MoE overlap skills.

Two overlap patterns for 1F1B:

1. **Merged FWD-FWD / BWD-BWD**: Same-type passes from two microbatches
   run in parallel. Costs 2× activation memory. Less overlap (fwd compute
   is half of bwd).

2. **Merged FWD-BWD** (preferred): Forward of microbatch i+1 overlaps
   with backward of microbatch i. No extra memory. Matches DualPipe
   design. Limited: first FWD and last BWD can't be hidden.

Key optimization: **W/D split**. Split backward MLP work into weight-gradient
and data-gradient pieces so the weight-gradient portion can overlap with forward
compute when forward MLP alone is too short to hide dispatch cost.

## MoE Token Dispatchers

MoE models route tokens to experts via all-to-all communication. The
dispatcher backend controls how this communication is implemented:

| Dispatcher | Backend | Mechanism |
|---|---|---|
| `alltoall` | Standard MoE | Torch-native all-to-all collectives |
| `flex` + DeepEP | DeepEP library | Low-latency SM-based dispatch with GPU-side routing |
| `flex` + HybridEP | HybridEP library | Fused intra-node NVLink + inter-node IB dispatch |

### Hardware affinity

| Hardware | Recommended Dispatcher | Rationale |
|---|---|---|
| H100 / B200 (NVL8) | DeepEP | Optimized for node-based topologies |
| GB200 / GB300 (NVL72) | HybridEP | Exploits NVLink domain for lower latency |

HybridEP advantage usually grows with EP degree because it fuses intra-node
NVLink transfers with inter-node IB work, avoiding much of the two-phase
overhead of standard all-to-all at large EP sizes.

## Long-Context MoE Training

At long sequences (64K+), SDPA dominates FLOPs. Context parallelism (CP)
is the primary mechanism for scaling sequence length.

CP sizing rules of thumb:

1. **Start with CP ≈ seq_len / 4096**: then round to a practical layout.

2. **Keep DP ≥ 1**: CP × EP × TP × PP must not exceed total GPUs.

3. **Prefer selective recompute over full**: Recompute `up_proj, norm, moe,
   mlp` rather than full recompute for better throughput.

4. **TP can sometimes substitute for some CP on NVLink systems**: on NVL72
   systems, higher TP can be competitive with a more CP-heavy plan.

5. **Optimizer CPU offload is often critical** at long context because
   activation pressure consumes so much of the memory budget.

Long-context recommendations:
- Keep sub-sequence length ~4096–8192 per CP/TP shard
- **Don't recompute SDPA at long context**: SDPA recompute adds significant
  compute overhead while saving relatively little memory. Recompute
  non-SDPA modules instead.
- TP preferred within node (fast comms, reduces param memory)
- P2P CP preferred across nodes (natural overlap with attention)
- a2a CP + TP within node when ring exchange is undesirable

## Dynamic Context Parallelism

For variable-length training (RL, SFT):
- Per-microbatch CP sizing instead of static CP for all
- Pre-constructs multiple CP groups during init (powers of 2)
- Scheduler selects effective cp_size per microbatch
- Works with packed sequences (THD format)

## MoE VLM Training

MoE vision-language models combine a vision encoder with a MoE language
decoder. Training requires choosing between two strategies:

| Approach | Mechanism | When to Use |
|---|---|---|
| FSDP | Shards params, grads, and optimizer across all GPUs | Simpler setup and a better first bring-up path |
| 3D Parallel | TP + PP + EP + DP | Higher throughput ceiling once the multimodal path is already stable |

Key principles:
- **Always benchmark with real vision data** — image-free mock runs can
  significantly overestimate throughput.
- **Freezing vision encoder** saves compute when fine-tuning only the decoder.
- **MBS is critical for 3D-parallel VLM** — larger micro-batch sizes often
  matter more than they do for text-only MoE.
- **FSDP is simpler and often competitive** for initial bring-up.

## Production Features Summary

| Feature | Purpose |
|---|---|
| Force-balance routing | Even token distribution; best for benchmarking |
| Aux-loss-free balancing | Learnable expert bias; adapts over time |
| Shared expert overlap | Hides shared expert latency behind dispatch/combine |
| LatentMoE | Reduces comms and per-expert params by compression ratio α |
| Distributed checkpoint | Parallelism-agnostic save/load with automatic resharding |
| Upcycling | Convert dense checkpoint to MoE without retraining |
| MTP | Multi-token prediction with flexible VPP placement |
| Muon optimizer | Matrix-aware updates; fewer steps than AdamW |
