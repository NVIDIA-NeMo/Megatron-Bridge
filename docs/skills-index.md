# Public Agent Skills Reference

Public operational guides and validated knowledge cards for Megatron Bridge.

Each public skill contains enablement snippets, code anchors, constraints,
pitfalls, and verification steps.

Contributor-facing repo workflow skills live under
`.agents/contributor-skills/` and are intentionally outside the public catalog
sync path.

```{toctree}
:caption: Getting Started
:maxdepth: 1

skills/mbridge-mlm-bridge-training/SKILL
skills/mbridge-recipe-recommender/SKILL
```

```{toctree}
:caption: Performance Techniques
:maxdepth: 1

skills/mbridge-perf-parallelism-strategies/SKILL
skills/mbridge-perf-cuda-graphs/SKILL
skills/mbridge-perf-tp-dp-comm-overlap/SKILL
skills/mbridge-perf-megatron-fsdp/SKILL
skills/mbridge-perf-sequence-packing/SKILL
skills/mbridge-perf-hierarchical-context-parallel/SKILL
skills/mbridge-perf-expert-parallel-overlap/SKILL
skills/mbridge-perf-moe-comm-overlap/SKILL
skills/mbridge-perf-activation-recompute/SKILL
skills/mbridge-perf-memory-tuning/SKILL
skills/mbridge-perf-moe-dispatcher-selection/SKILL
skills/mbridge-perf-moe-hardware-configs/SKILL
skills/mbridge-perf-moe-long-context/SKILL
skills/mbridge-perf-moe-optimization-workflow/SKILL
skills/mbridge-perf-moe-vlm-training/SKILL
skills/mbridge-perf-cpu-offloading/SKILL
```

```{toctree}
:caption: Cluster & Debugging
:maxdepth: 1

skills/mbridge-multi-node-slurm/SKILL
```

```{toctree}
:caption: Resiliency
:maxdepth: 1

skills/mbridge-resiliency/SKILL
```
