# Skills

Public product-use skills for AI coding agents working with Megatron Bridge.

These skills are under the public catalog sync path. Contributor-facing repo
workflow skills live in [.agents/contributor-skills](../.agents/contributor-skills/README.md)
and are not synced externally.

## Public Catalog Skills

| Skill | Description |
|---|---|
| `mbridge-mlm-bridge-training` | Run Megatron-LM and Megatron Bridge training with mock or real data |
| `mbridge-multi-node-slurm` | Convert scripts to multi-node Slurm jobs and debug multi-node failures |
| `mbridge-perf-activation-recompute` | Enable selective and full activation recompute |
| `mbridge-perf-cpu-offloading` | Enable activation and optimizer CPU offloading |
| `mbridge-perf-cuda-graphs` | Use CUDA graph capture in Megatron Bridge |
| `mbridge-perf-expert-parallel-overlap` | Enable MoE expert-parallel communication overlap |
| `mbridge-perf-hierarchical-context-parallel` | Configure hierarchical context parallelism |
| `mbridge-perf-megatron-fsdp` | Enable Megatron FSDP |
| `mbridge-perf-memory-tuning` | Reduce peak GPU memory and debug OOMs |
| `mbridge-perf-moe-comm-overlap` | Tune MoE dispatch/combine communication overlap |
| `mbridge-perf-moe-dispatcher-selection` | Choose alltoall, DeepEP, or HybridEP dispatchers |
| `mbridge-perf-moe-hardware-configs` | Pick representative MoE hardware configurations |
| `mbridge-perf-moe-long-context` | Tune long-context MoE training |
| `mbridge-perf-moe-optimization-workflow` | Follow a systematic MoE optimization workflow |
| `mbridge-perf-moe-vlm-training` | Train MoE VLMs with Megatron Bridge |
| `mbridge-perf-parallelism-strategies` | Choose and combine parallelism strategies |
| `mbridge-perf-sequence-packing` | Use packed sequences and long-context training constraints |
| `mbridge-perf-tp-dp-comm-overlap` | Enable TP, DP, and PP communication overlap |
| `mbridge-recipe-recommender` | Recommend and customize Megatron Bridge recipes |
| `mbridge-resiliency` | Configure fault tolerance, straggler detection, and restart behavior |
