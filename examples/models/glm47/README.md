# GLM-4.7 / GLM-4.7-Flash Examples

Scripts for the GLM-4.7 family — [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) (`zai-org/GLM-4.7`) and [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) (`zai-org/GLM-4.7-Flash`).

The two models share the GLM-4.7 family but use different architectures:

| Model | HF ID | Architecture | Bridge | Params | Active Params |
|---|---|---|---|---|---|
| GLM-4.7 | `zai-org/GLM-4.7` | MoE (160 experts, top-8, 1 shared) | `GLM47Bridge` (via `Glm4MoeForCausalLM`) | ~358B | ~32B |
| GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` | MLA + MoE (64 experts, top-4, 1 shared) | `GLM47FlashBridge` | ~30B | ~3B |

GLM-4.7 uses standard multi-head attention; GLM-4.7-Flash adopts Multi-Latent Attention (MLA, `q_lora_rank=768`, `kv_lora_rank=512`) inherited from DeepSeek-V3. Both have 47 transformer layers with the first dense and the rest MoE.

**Requirements:** `transformers >= 5.0.0rc0` (for `Glm4MoeLiteForCausalLM` / `Glm4MoeForCausalLM`).

## Hardware Requirements

| Model | Min GPUs | Recommended parallelism |
|---|---|---|
| GLM-4.7-Flash | 8 (1 node × H100/H200 80 GB) | `TP=1, EP=8, PP=1` |
| GLM-4.7 | 32 (4 nodes × 8 GPUs) | `TP=1, EP=32, PP=1` |

EP must divide the number of routed experts (64 for Flash, 160 for full). TP does **not** reduce expert memory — scale EP first.

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

## Checkpoint Conversion (Round-Trip)

### GLM-4.7-Flash (single node)

[conversion.sh](conversion.sh) runs HF → Megatron → HF round-trip with `EP=8` and `TP=2 EP=4` on 8 GPUs, then imports / exports a Megatron checkpoint.

```bash
bash examples/models/glm47/conversion.sh
```

### GLM-4.7 (multi-node via Slurm)

[slurm_conversion.sh](slurm_conversion.sh) sweeps multiple parallelism configs (`TP,PP,EP`) to verify round-trip conversion. Each config runs sequentially.

```bash
sbatch examples/models/glm47/slurm_conversion.sh
```

Default sweep: `1,1,32`, `2,1,16`, `1,2,16` on 4 nodes (32 GPUs).

## Slurm Script Configuration

Set the following before `sbatch`:

| Variable | Description |
|---|---|
| `CONTAINER_IMAGE` | Path to Singularity / SquashFS container image |
| `CONTAINER_MOUNTS` | Bind mounts (must include the Bridge checkout as `/opt/Megatron-Bridge`) |
| `HF_HOME` | HuggingFace cache directory containing the downloaded checkpoint |
| `HF_TOKEN` | HuggingFace access token (for gated model access) |
| `MODEL_NAME` | `GLM-4.7` or `GLM-4.7-Flash` |
