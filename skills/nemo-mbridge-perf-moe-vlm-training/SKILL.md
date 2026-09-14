---
name: nemo-mbridge-perf-moe-vlm-training
description: Practical guidance for training MoE VLMs in Megatron Bridge. Compares FSDP and 3D-parallel approaches, using rounded lessons from Qwen3-VL, Qwen3-Next, and other multimodal experiments.
license: Apache-2.0
when_to_use: Training MoE VLMs, or investigating a commit that caused MoE VLM training failure or OOM; 'MoE VLM', 'multimodal MoE', 'Qwen3-VL training', 'FSDP vs 3D-parallel for VLM', 'MoE vision language model'.
---

# MoE VLM Training

Stable docs: @docs/training/moe-optimization.md
Card: @skills/nemo-mbridge-perf-moe-vlm-training/card.yaml

## FSDP vs 3D Parallel

| Approach | Strength | Best fit |
|---|---|---|
| FSDP | Simplest path to a working multimodal run | first bring-up, memory-first tuning, awkward PP boundaries |
| 3D parallel | Higher ceiling after tuning | stable models with a clean PP layout and time for deeper sweeps |

For MoE VLMs, the practical workflow is usually:

1. get the first reliable run with FSDP
2. stabilize real-data input, recompute, and memory behavior
3. move to 3D parallel only if the throughput headroom is worth the extra work

## Rounded Findings From Recent VLM Runs

### Qwen3-VL class models

The main patterns were consistent across the tracker:

- FSDP on GB200-class systems can already reach healthy high-teens utilization
  with a comparatively simple setup
- B200 FSDP runs are viable, but more sensitive to recompute choice and frozen
  vision settings
- 3D parallel can recover to a similar or better operating point, but only after
  tuning MBS, recompute, and the real vision path together

### Real data vs mock data

Mock-data VLM runs are not trustworthy performance proxies. In the experiments,
image-free mock runs looked closer to "roughly twice as fast" than "slightly
optimistic" when compared with real multimodal input.

Use real or realistic image payloads before drawing any conclusion about VLM
throughput.

### Smaller multimodal MoE runs

The smaller Qwen3.5-style multimodal experiments reinforce the same lessons:

- HybridEP is a solid default on GB200
- TE-scoped CUDA graphs help once the training loop is stable
- larger MBS can pay off, but only if the vision encoder does not become the
  next bottleneck

## Decision Guide

### Choose FSDP when

- you are bringing up a new VLM for the first time
- the model has awkward stage boundaries across embedding, vision, and decoder
- memory fit matters more than absolute throughput
- you may freeze the vision stack during decoder-focused tuning

### Choose 3D parallel when

- the model is already stable under FSDP
- the PP layout is clear and repeatable
- you can sweep MBS, recompute, and CUDA-graph scope together
- the goal is best steady-state throughput, not easiest bring-up

## Key Tuning Knobs

1. **Keep TP as small as memory permits**: prefer activation recompute over
   increasing TP when activations, rather than model weights, are the reason the
   run does not fit. Lower TP keeps GEMMs larger and avoids unnecessary TP
   communication. TP may still be required for dense weights, the vocabulary
   head, or a workable PP boundary; verify the actual memory limiter before
   removing it.

2. **Make EP large enough to shard expert weights effectively**: for an MoE
   model whose expert count is divisible by the rank count, start with EP8 on an
   8-GPU allocation and EP32 on a 32-GPU allocation. Treat these as starting
   points, not unconditional rules: EP does not shard dense, vision, embedding,
   or output-layer weights, and the final choice must respect PP/DP rank layout
   and the hardware topology.

3. **Prefer selective activation recompute after fit is established**: use full
   recompute to get a memory-constrained configuration running, then replace it
   with the narrowest supported selective modules. This usually recovers more
   throughput than adding TP solely for activation memory.

   Qwen3.5/Qwen3.6 requires special care. Megatron-Core supports the targeted
   `gdn_norm_out` recompute module, but that does not checkpoint the complete GDN
   recurrence. If `core_attn`, `gdn_norm_out`, and `moe_act` still do not provide
   enough headroom—commonly with smaller EP such as EP8—retain full recompute
   rather than claiming unsupported GDN selective coverage. Larger EP, such as
   EP32, can reduce expert-weight pressure enough to retry the selective policy.

4. **Prefer HybridEP, then measure against AllToAll**: HybridEP is the default
   candidate when the topology supports it. Within one 8-GPU NVLink domain in
   BF16, HybridEP and conventional AllToAll can be close; benchmark both with
   identical work instead of assuming HybridEP must win.

5. **Freeze the vision stack when appropriate**: if the work is decoder-focused,
   freezing the vision side often gives a small but real throughput gain and
   reduces memory pressure.

6. **Sweep MBS aggressively**: VLMs are more MBS-sensitive than text-only MoE
   runs because the vision path changes the compute-to-overhead balance.

7. **Match CUDA-graph scope to the workload**: `attn moe_router moe_preprocess`
   is the safer MoE default, while narrower scopes can still be useful for
   controlled experiments.

8. **Use ETP only when EP alone is insufficient**: it can unlock a layout, but
   it also introduces more communication and more tuning surface.

## Fit-First Tuning Order

Use this order so each measurement answers one question:

1. Select the largest topology-compatible EP that usefully shards the experts.
2. Minimize TP while retaining enough memory for non-expert weights.
3. Start with full recompute if needed to obtain a stable real-data run.
4. Replace full recompute with supported selective modules and remeasure memory.
5. Inspect per-rank peak allocated and reserved memory in the experiment logger.
6. If there is headroom, increase MBS; if TP was used for activation pressure,
   lower TP and remeasure. Change one of these two controls at a time.
7. Compare HybridEP with AllToAll on the same topology and precision.
8. Add communication overlap and CUDA graphs only after the memory and batch
   operating point is stable.

At every step, keep the dataset, sequence/image shapes, global batch, precision,
trainable parameters, and useful-work accounting fixed. A configuration that
fits is the baseline; it is not automatically the performance recipe.

## Representative Config Families

### FSDP-first GB200 path

```text
TP=1  CP=1  PP=1
EP sized to the expert topology, often large
Dispatcher: HybridEP on GB200-class systems
Recompute: start with full, then relax toward selective recompute
```

### 3D-parallel GB200 path

```text
TP=1  CP=1  PP=1 or modest PP
EP and ETP sized to the expert topology
Dispatcher: HybridEP
CUDA Graph: start narrow, then widen only after the real-data path is stable
```

## Compatibility

| Feature | FSDP | 3D parallel |
|---|---|---|
| HybridEP on GB200 | strong default | strong default once topology is stable |
| CUDA graphs | useful after bring-up | useful, but more scope-sensitive |
| Freeze vision | natural fit | possible, but less often used as the headline perf path |
| Selective recompute | recommended | recommended |

## Pitfalls

1. **Mock multimodal data is misleading**: it can make the decoder look much
   healthier than the real end-to-end VLM path.

2. **The vision encoder can dominate unexpectedly**: profile encoder, projector,
   and decoder separately before attributing everything to the dispatcher.

3. **Do not compare FSDP and 3D-parallel runs with different effective work**:
   normalize by useful tokens and workload shape, not only by step time.

4. **ETP is not free**: use it as a fit or topology tool, not as the default.

5. **Recompute and CUDA-graph choices are coupled**: the setting that gets the
   model to fit is often not the setting that gives the best steady-state speed.

6. **Qwen3.5 GDN recompute coverage is easy to overstate**: `gdn_norm_out`
   checkpoints the normalization/output portion, not the full GDN recurrence.
   Confirm the supported module names in the pinned Megatron-Core before
   replacing full recompute.

7. **High EP does not solve every memory problem**: it reduces expert-weight
   pressure but does not shard the vision encoder, embeddings, dense attention,
   or output head.
