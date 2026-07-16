# GLM-4.7 / GLM-4.7-Flash Examples

Scripts for the GLM-4.7 family: [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) (`zai-org/GLM-4.7`) and [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) (`zai-org/GLM-4.7-Flash`).

The two models share the GLM-4.7 family but use different architectures:

| Model | HF ID | Architecture | Bridge | Layers | Params | Active Params |
|---|---|---|---|---|---|---|
| GLM-4.7 | `zai-org/GLM-4.7` | MoE (160 routed experts, top-8, 1 shared) | `GLM45Bridge` (`Glm4MoeForCausalLM`) | 92 | ~358B | ~32B |
| GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` | MLA + MoE (64 routed experts, top-4, 1 shared) | `GLM47FlashBridge` (`Glm4MoeLiteForCausalLM`) | 47 | ~30B | ~3B |

GLM-4.7 uses the existing `glm4_moe` architecture already covered by `GLM45Bridge`; its config has 92 transformer layers with the first 3 dense and the rest MoE. GLM-4.7-Flash uses `glm4_moe_lite` with Multi-Latent Attention (MLA, `q_lora_rank=768`, `kv_lora_rank=512`), 47 transformer layers, and the first layer dense.

**Requirements:** `transformers >= 5.0.0rc0` (for `Glm4MoeLiteForCausalLM` / `Glm4MoeForCausalLM`).

## Validation Status

- [x] GLM-4.7-Flash round-trip on 8 GPUs with `TP=1, PP=1, EP=8`: 9,701 weights matched, 0 failures.
- [x] GLM-4.7-Flash inference on 8 GPUs with `TP=1, PP=1, EP=8`: coherent output on the default prompt.
- [x] GLM-4.7 inference on 32 GPUs with `TP=1, PP=1, EP=32`: coherent output on the default prompt.

## Hardware Requirements

| Model | Min GPUs | Recommended parallelism |
|---|---|---|
| GLM-4.7-Flash | 8 (1 node x H100/H200 80 GB) | `TP=1, EP=8, PP=1` |
| GLM-4.7 | 32 (4 nodes x 8 GPUs) | `TP=1, EP=32, PP=1` |

EP must divide the number of routed experts (64 for Flash, 160 for full). TP does **not** reduce expert memory; scale EP first.

## Inference (Megatron)

### GLM-4.7-Flash (single node)

[inference.sh](inference.sh) runs text generation directly with `torch.distributed.run`:

```bash
bash examples/models/glm47/inference.sh
```

### GLM-4.7 (multi-node via Slurm)

[slurm_inference.sh](slurm_inference.sh) loads the HF checkpoint, converts to Megatron in-memory, and runs greedy text generation across 4 nodes (32 GPUs) with `TP=1, EP=32`.

```bash
sbatch examples/models/glm47/slurm_inference.sh
```

### Expected output (GLM-4.7-Flash, `TP=1 EP=8`, prompt: "What is artificial intelligence?")

```
======== GENERATED TEXT OUTPUT ========
Prompt: What is artificial intelligence?
Generated: What is artificial intelligence? Artificial intelligence (AI) is the
simulation of human intelligence processes by computer systems. These processes
include learning (the acquisition of information and rules for using the
information), reasoning (using rules to reach approximate or definite
conclusions), and self-correction.

Artificial intelligence is a branch of computer science that aims to create
intelligent machines. It is an interdisciplinary field that combines computer
science, mathematics, and statistics to build systems that can perform tasks
that typically require ...
=======================================
```

### Expected output (GLM-4.7, `TP=1 PP=1 EP=32`, prompt: "What is artificial intelligence?")

```
======== GENERATED TEXT OUTPUT ========
Prompt: What is artificial intelligence?
Generated: What is artificial intelligence? How is artificial intelligence going
to change our lives?
"Alice in Wonderland" author Lewis Carroll wasn't thinking about artificial
intelligence when he wrote about the Red Queen's race. But the scene where Alice
runs as fast as she can just to stay in the same place is a perfect metaphor for
the current state of AI.
AI is evolving at a breakneck pace. It's moving so fast that it's hard to keep
up with the latest developments, let alone understand their implications.
=======================================
```

## Checkpoint Conversion (Round-Trip)

### GLM-4.7-Flash (single node)

[conversion.sh](conversion.sh) runs HF -> Megatron -> HF round-trip with `TP=1, PP=1, EP=8` on 8 GPUs, then imports / exports a Megatron checkpoint.

```bash
bash examples/models/glm47/conversion.sh
```

### GLM-4.7 (multi-node via Slurm)

[slurm_conversion.sh](slurm_conversion.sh) runs HF -> Megatron -> HF round-trip with `TP=1, PP=1, EP=32` on 4 nodes (32 GPUs).

```bash
export CONTAINER_IMAGE=/path/to/container.sqsh
export SLURM_ACCOUNT=your_account
bash examples/models/glm47/slurm_conversion.sh
```

The wrapper uses `convert.sh roundtrip --executor slurm`, submits from the login
node, and waits for completion by default. The job validates weights in memory
without producing a new checkpoint. The model, resources, and parallelism
layout are written directly in the script so the example is self-contained.

## Slurm Script Configuration

Set the following before running a Slurm wrapper:

| Variable | Description |
|---|---|
| `CONTAINER_IMAGE` | Path to Singularity / SquashFS container image |
| `SLURM_ACCOUNT` | Slurm account used for the submitted job |
| `SLURM_PARTITION` | Slurm partition; defaults to `batch` |
| `CONTAINER_MOUNTS` | Optional comma-separated bind mounts for data or caches; the current checkout is mounted automatically at `/opt/Megatron-Bridge` |
| `HF_HOME` | HuggingFace cache directory containing the downloaded checkpoint |
| `HF_TOKEN` | HuggingFace access token (for gated model access) |
| `PROMPT`, `MAX_NEW_TOKENS` | Optional inference prompt and generation length overrides |

Cluster-specific `srun` flags are not hardcoded. Pass them after the conversion
wrapper, for example `--srun-arg=--mpi=pmix`.
