---
name: nemo-mbridge-recipe-recommender
license: Apache-2.0
description: Recommend and customize Megatron Bridge library and benchmark recipes for a user's model, GPU count, hardware, sequence length, and pretrain/SFT/PEFT goal. Use when selecting a starting recipe, comparing library and benchmark configs, resizing parallelism for a GPU allocation, or distinguishing convergence changes, semantics-preserving execution tuning, and benchmark-only shortcuts.
---

# Auto Recipe — Recipe Index & Recommendation

This skill indexes every shipped recipe and helps users pick the right starting
config, adjust parallelism, and avoid common pitfalls.

## How to Use This Skill

1. Ask the user for: **model name/size**, **GPU count & type**, **training goal**
   (pretrain / SFT / PEFT), and **sequence length** (if non-default).
2. Look up the best-match recipe in the index below.
3. Recommend the recipe function name + entry-point command.
4. Provide adjustment advice (parallelism resizing, batch tuning, pitfalls).

## First Answer Checklist

When recommending recipes, always include these distinctions before the long
index details:

1. **Library recipes** under `src/megatron/bridge/recipes/` are for functional
   training and use `scripts/training/run_recipe.py`.
2. **Benchmark recipes** under `src/megatron/bridge/perf_recipes/` are for
   upper-bound throughput benchmarks. They own their canonical benchmark data
   and settings and should not be presented as production training recipes.
3. For a first-time Bridge smoke test, recommend `llama3_8b_pretrain_config`
   with mock data via `--dataset mock`.
4. For normal SFT recommendations, select a finetuning preset such as
   `--dataset squad` or `--dataset tulu3`; for pretrain and mock validation
   recommendations, use `--dataset mock`. Do not pair the pretraining-only
   `mock` preset with an SFT or PEFT mode.
5. After the recipe and dataset, give the required resizing rules: TP must
   divide `num_key_value_heads`, keep TP within one node unless using
   NVL72-class interconnect, enable SP when TP > 1, configure CP for long
   context, DP is implicit, and reduce `micro_batch_size` first on OOM.
6. State whether each proposed override changes the convergence contract or
   only the execution/performance mapping. Do not trade convergence semantics
   for throughput without calling it a new experiment.

## Configuration Layers and Change Control

Separate training semantics from their hardware mapping before recommending or
tuning a recipe.

**Convergence configuration** includes the starting checkpoint and trainable
parameters; dataset/revision/split/order/seeds; tokenizer, masking, truncation,
and packing; sequence length; global batch and token budget; objective and loss
coefficients; natural or forced MoE routing and token-dropping policy;
optimizer, LR, schedule, warmup, betas, epsilon, weight decay, clipping, and
dropout; arithmetic and optimizer-state precision; and PEFT adapter settings.
Changing one of these creates a new convergence experiment.

**Execution/performance configuration** includes hardware count and topology;
TP/PP/VP/CP/EP/ETP/DP/SP; recompute and offload; distributed optimizer/FSDP;
communication overlap; fusions and attention backends; CUDA graphs and
compilation; checkpoint I/O; and MoE transport through all-to-all, DeepEP, or
HybridEP when the routing policy is unchanged. These settings should preserve
the objective and effective updates, although floating-point reduction order
can produce small numerical drift that still needs validation.

Treat micro batch size and gradient accumulation as execution fingerprints.
Tune them only with fixed global batch size, global batch membership/order,
normalization, optimizer boundaries, and token budget, and validate fresh loss
sentinels for each layout. Packing, precision, forced MoE load balancing, token
dropping/capacity, and router/auxiliary loss changes are never performance-only
knobs.

Treat mock data, forced balancing, disabled correctness checks, and timing-only
schedules as benchmark-only shortcuts. They may be appropriate in
`perf_recipes`, but their losses and checkpoints are not convergence evidence.

For comparable model-verification recipes, choose a cohort-wide convergence
contract before tuning performance. Keep the same bounded data selection,
preprocessing, sequence length, global batch, optimizer/schedule, precision,
seeds, routing policy, optimizer-step horizon, and processed-token checkpoints
where the architectures permit. Record any necessary model-specific deviation
and do not present that result as apples-to-apples convergence evidence.
Absolute losses from different architectures or tokenizers are not directly
rankable; compare stability and trend at equal token counts.

When a recipe's batch disagrees with the chosen convergence contract, modify
and validate the library recipe separately. A declared bounded-verification
protocol may explicitly apply the same LR, schedule, sequence, and data
overrides across a cohort, but do not make one-off convergence changes merely
to improve throughput. Conversely, first try TP/PP/CP/EP, recompute/offload,
dispatcher transport, overlap, fusion, and CUDA graphs when optimizing fit or
throughput.

---

## Entry Points

### Library recipes (functional training)

```bash
# Pretrain with mock data
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe <recipe_function_name> \
    --dataset mock

# SFT with SQuAD
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe <recipe_function_name> \
    --dataset squad

# Override any field via CLI
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe llama3_8b_pretrain_config \
    --dataset mock \
    'model.tensor_model_parallel_size=2' \
    'train.global_batch_size=64'
```

### Benchmark recipes (throughput benchmarks)

```bash
./scripts/training/train.sh \
    --nodes 2 --gpus-per-node 8 \
    --account ACCOUNT --partition PARTITION --container-image IMAGE \
    --recipe qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config \
    --mode pretrain
```

The total GPU allocation must match the count encoded in the recipe name. The
user selects the node shape, and the selected partition must provide the
requested hardware. The launcher does not inject benchmark offline defaults or
cluster-specific launch policy. Use `--env NAME` for exported offline or NCCL
fabric settings and repeated `--srun-arg=ARG` options for `srun`.
Configure CPU/NUMA wrappers and Slurm segment sizing through the target cluster
integration, or use `scripts/performance/setup_experiment.py` when its compatibility
policies are required. The unified
launcher supports exact exported text pretraining, text SFT/PEFT, Qwen-VL
pretraining, and Wan pretraining recipes and infers their forward step. Text
SFT/PEFT text benchmark recipes retain the flat runner's mock-data default;
Qwen-VL and Wan retain their model-specific datasets. Exported benchmark PEFT
recipes are fixed LoRA configs; use a configurable library recipe for DoRA.
Trailing `KEY=VALUE` overrides are accepted, but an overridden benchmark
recipe no longer represents its canonical benchmark configuration. Use
`scripts/performance/setup_experiment.py` for selector-based invocation,
dataset replacement, topology resizing, and specialized benchmark controls.

See the Benchmark Recipe Index for important caveats before using these for
anything beyond throughput benchmarking.

---

## Benchmark Recipe Layout

Benchmark recipes use the same **Python function** format as library recipes,
but live in a dedicated namespace for throughput benchmarking:

- Benchmark recipes live in `src/megatron/bridge/perf_recipes/<family>/<hardware>/<model>.py`
- Each benchmark recipe is a **self-contained Python function** (e.g. `llama3_8b_pretrain_8gpu_h100_bf16_config()`)
- Recipe names encode model, task, GPU count, hardware, precision, and optional variant
- `scripts/performance/utils/utils.py` derives compatibility `WorkloadBaseConfig` views from the flat recipe itself
- Shared helpers: `_benchmark_common()` (50 iters, timing, TE RNG), `_perf_precision()` (bf16 / fp8_cs / fp8_mx / nvfp4)

**Why Python, not YAML?** Previous YAML-based approaches had problems:
recipe logic was split across multiple indirection layers, configs were not
self-contained, and the two-level pipeline made maintenance and debugging
difficult. Python functions are explicit, greppable, and composable.

The training launcher discovers library and benchmark recipes from the
complete exported function name. Five legacy duplicate names select the
benchmark definition; use the corresponding generic alias for those functional
workloads. New recipe names should be unique across both packages.

---

## Library Recipe Index

All recipes live under `src/megatron/bridge/recipes/`. Each function returns a
`ConfigContainer` with model, training, optimizer, and data settings.

### Llama

| Recipe | Mode | TP | PP | CP | SP | GPUs (min) | Seq Len |
|--------|------|----|----|----|----|------------|---------|
| `llama2_7b_pretrain_config` | Pretrain | 2 | 1 | — | — | 2 | 4K |
| `llama3_8b_pretrain_config` | Pretrain | 2 | 1 | — | ✓ | 2 | 8K |
| `llama3_8b_16k_pretrain_config` | Pretrain | 2 | 1 | 2 | ✓ | 4 | 16K |
| `llama3_8b_64k_pretrain_config` | Pretrain | 2 | 1 | 4 | ✓ | 8 | 64K |
| `llama3_8b_128k_pretrain_config` | Pretrain | 2 | 1 | 8 | ✓ | 16 | 128K |
| `llama3_70b_pretrain_config` | Pretrain | 8 | 4 | — | ✓ | 32 | 8K |
| `llama3_70b_16k_pretrain_config` | Pretrain | 8 | 4 | 2 | ✓ | 64 | 16K |
| `llama3_70b_64k_pretrain_config` | Pretrain | 8 | 4 | 4 | ✓ | 128 | 64K |
| `llama31_405b_pretrain_config` | Pretrain | 8 | 16 | — | ✓ | 128 | 8K |
| `llama3_8b_sft_config` | SFT | 2 | 1 | — | ✓ | 2 | 8K |
| `llama3_70b_sft_config` | SFT | 4 | 4 | — | ✓ | 16 | 8K |
| `llama31_405b_sft_config` | SFT | 8 | 8 | — | ✓ | 64 | 8K |
| `llama3_8b_peft_config` | PEFT | 1 | 1 | — | — | 1 | 8K |
| `llama3_70b_peft_config` | PEFT | 2 | 4 | — | ✓ | 8 | 8K |
| `llama31_405b_peft_config` | PEFT | 4 | 8 | — | ✓ | 32 | 8K |

### Qwen2 / Qwen2.5

| Recipe | Mode | TP | PP | Sizes |
|--------|------|----|----|-------|
| `qwen2_*_{pretrain,sft,peft}_config` | All | 1–8 | 1–4 | 500M, 1.5B, 7B, 14B, 32B, 72B |
| `qwen25_*_{pretrain,sft,peft}_config` | All | 1–8 | 1–4 | 500M, 1.5B, 3B, 7B, 14B, 32B, 72B |

### Qwen3 (Dense)

| Recipe | Mode | TP | PP | CP | GPUs | Sizes / notes |
|--------|------|----|----|----|------|---------------|
| `qwen3_8b_pretrain_config` | Pretrain | 1 | 1 | — | 16 | Bounded convergence cohort |
| `qwen3_8b_sft_config` | SFT | 4 | 1 | — | 4 | 2K bounded convergence cohort |
| `qwen3_8b_sft_32k_config` | SFT | 4 | 1 | 2 | 8 | Separate 32K long-context cohort |
| `qwen3_8b_peft_config` | PEFT | 1 | 1 | — | 1 | Bounded LoRA/DoRA cohort |
| `qwen3_*_{pretrain,sft,peft}_config` | All | 1–8 | 1–2 | — | varies | Other dense sizes, 600M–32B |
| `qwen3_600m_sft_128k_config` | SFT | 1 | 1 | 8 | 8 | 600M, 128K sequence |

### Qwen3 MoE

| Recipe | Mode | TP | PP | EP | CP | GPUs |
|--------|------|----|----|----|----|------|
| `qwen3_30b_a3b_pretrain_config` | Pretrain | 1 | 1 | 16 | — | 16 |
| `qwen3_30b_a3b_sft_config` | SFT | 1 | 1 | 16 | — | 16 |
| `qwen3_30b_a3b_peft_config` | PEFT | 4 | 1 | 4 | — | 4 |
| `qwen3_235b_a22b_pretrain_config` | Pretrain | 4 | 16 | 8 | 2 | 256 |
| `qwen3_235b_a22b_sft_config` | SFT | 4 | 16 | 4 | — | 64 |
| `qwen3_235b_a22b_peft_config` | PEFT | 1 | 4 | 4 | — | 16 |

### Qwen3-Next

| Recipe | Mode | TP | PP | EP |
|--------|------|----|----|-----|
| `qwen3_next_80b_a3b_pretrain_config` | Pretrain | 1 | 4 | 8 |
| `qwen3_next_80b_a3b_sft_config` | SFT | 1 | 2 | 8 |
| `qwen3_next_80b_a3b_peft_config` | PEFT | 1 | 1 | 4 |

### DeepSeek

| Recipe | Mode | TP | PP | EP | GPUs |
|--------|------|----|----|-----|------|
| `deepseek_v2_lite_pretrain_config` | Pretrain | 1 | 1 | 8 | 8 |
| `deepseek_v2_pretrain_config` | Pretrain | 1 | 4 | 32 | 128 |
| `deepseek_v3_pretrain_config` | Pretrain | 2 | 16 | 64 | 2048 |
| `deepseek_v3_pretrain_config_32nodes` | Pretrain | 2 | 8 | 32 | 256 |

### GLM-4.5

| Recipe | Mode | TP | PP | EP | GPUs |
|--------|------|----|----|-----|------|
| `glm45_355b_pretrain_config` | Pretrain | 2 | 8 | 16 | 256 |
| `glm45_air_106b_pretrain_config` | Pretrain | 1 | 4 | 8 | 32 |
| `glm45_355b_sft_config` | SFT | 2 | 8 | 16 | 256 |
| `glm45_air_106b_sft_config` | SFT | 1 | 4 | 8 | 32 |
| `glm45_355b_peft_config` | PEFT | 2 | 4 | 4 | 32 |
| `glm45_air_106b_peft_config` | PEFT | 1 | 2 | 4 | 8 |

### Gemma

| Recipe | Mode | TP | PP | Sizes |
|--------|------|----|----|-------|
| `gemma2_*_{pretrain,sft,peft}_config` | All | 2–8 | 1–2 | 2B, 9B, 27B |
| `gemma3_1b_{pretrain,sft,peft}_config` | All | 1 | 1 | 1B (32K seq) |

### NemotronH / Nemotron

| Recipe | Mode | TP | PP | EP | Notes |
|--------|------|----|----|-----|-------|
| `nemotronh_{4b,8b,47b,56b}_*_config` | P/S/PEFT | 1–8 | 1–4 | — | Dense SSM-hybrid |
| `nemotron_3_nano_*_config` | P/S/PEFT | varies | 1 | 8 | MoE + Mamba |
| `nemotron_3_super_*_config` | P/S/PEFT | 4 | 1 | 8 | MoE + Mamba, ~40% CUDA graph gain |
| `nemotron_nano_{9b,12b}_v2_*_config` | P/S/PEFT | varies | 1 | — | Dense |

### Other Models

| Recipe | Mode | Notes |
|--------|------|-------|
| `moonlight_16b_pretrain_config` | Pretrain | 16 GPUs, TP1/PP1/EP8 with HybridEP; GBS/MBS 1024/2 bounded convergence cohort |
| `moonlight_16b_sft_config` | SFT | 8 GPUs, TP1/PP1/EP8; 8K offline packing, GBS/MBS 8/1, 65,536 tokens/update |
| `moonlight_16b_peft_config` | PEFT | 4 GPUs, TP1/PP1/EP4; bounded LoRA/DoRA cohort |
| `moonlight_16b_sft_8k_config` | SFT | 8 GPUs, TP2/PP1/CP2/EP8; separate 8K cohort |
| `olmoe_7b_{pretrain,sft,peft}_config` | All | MoE EP=8 |
| `ministral3_{3b,8b,14b}_{sft,peft}_config` | SFT/PEFT | Dense |
| `gpt_oss_20b_*_config` | All | MoE + FP8/MXFP8 variants |
| `gpt_oss_120b_*_config` | All | MoE |
| `vanilla_gpt_pretrain_config` | Pretrain | MLM/Bridge parity baseline |
| `gpt3_175b_pretrain_config` | Pretrain | TP=4, PP=8, VP=6 |
| `kimi_k2_pretrain_config` | Pretrain | 1T MoE, TP=2 PP=16 EP=32 |

### VLM Recipes

| Recipe | Mode | TP | PP | EP | GPUs |
|--------|------|----|----|-----|------|
| `gemma3_vl_{4b,12b,27b}_{sft,peft}_config` | SFT/PEFT | 1–8 | 1–2 | — | 1–16 |
| `qwen25_vl_{3b,7b,32b,72b}_{sft,peft}_config` | SFT/PEFT | 1–8 | 1–4 | — | 1–32 |
| `qwen3_vl_{8b,30b_a3b,235b_a22b}_{sft,peft}_config` | SFT/PEFT | 1–4 | 1–8 | 1–32 | 1–512 |
| `qwen35_vl_*_{sft,peft}_config` | SFT/PEFT | varies | varies | varies | varies |
| `glm_45v_{sft,peft}_config` | SFT/PEFT | 1 | 8 | 4–16 | 64–512 |
| `nemotron_nano_v2_vl_12b_{sft,peft}_config` | SFT/PEFT | 2–4 | 1 | — | 8 |

### Diffusion Recipes

| Recipe | Mode | TP | CP |
|--------|------|----|----|
| `wan_1_3B_{pretrain,sft}_config` | P/SFT | 1 | 8 |
| `wan_14B_{pretrain,sft}_config` | P/SFT | 2 | 4 |
| `flux_12b_{pretrain,sft}_config` | P/SFT | 2 | 1 |

---

## Benchmark Recipe Index

Benchmark recipe source lives under `src/megatron/bridge/perf_recipes/`. The
compatibility launcher in `scripts/performance/` resolves those flat recipe
names and derives compatibility workload views from the selected flat recipe
when legacy helper paths still need them.

> **Important:** Benchmark recipes are designed for **upper-bound throughput
> measurements**, not production training. Text benchmarks use mock data by
> default; Qwen-VL and Wan retain their model-specific datasets. Throughput
> numbers are aspirational targets, not validated convergence configs.

### Llama 3 / 3.1

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Llama 3 8B | 8 | H100, B200, B300, GB200, GB300, R100 | CUDA graphs (local), FSDP on GB variants |
| Llama 3 70B | 64 | H100, B200, B300, GB200, GB300 | TP comm overlap (userbuffers), FSDP, CUDA graphs |
| Llama 3.1 405B | 128–1024 | H100, B200, B300, GB200, GB300 | TP+CP comm overlap (userbuffers), FSDP, heavy PP/VP |

SFT/LoRA variants also exist (e.g. 8B SFT with packed sequences, 70B SFT on 32 GPUs).

### DeepSeek V3

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| DeepSeek V3 (671B MoE) | 256–1024 | H100, B200, B300, GB200, GB300 | HybridEP dispatcher, MLA recompute, CUDA graphs (TE scoped) |

### Qwen3 MoE

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Qwen3 30B-A3B | 8–16 | H100, B200, B300, GB200, GB300 | MoE alltoall/flex dispatcher |
| Qwen3 235B-A22B | 64–256 | H100, B200, B300, GB200, GB300 | TP comm overlap, CUDA graphs, MoE a2a overlap |
| Qwen3-Next 80B-A3B | 64–128 | H100, B200, B300, GB200, GB300 | EP 64–128 |

### Qwen3-VL

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Qwen3-VL 30B-A3B | 8–16 | H100, B200, B300, GB200, GB300 | VLM + MoE |
| Qwen3-VL 235B-A22B | 64–256 | H100, B200, B300, GB200, GB300 | VLM + MoE, TP comm overlap |

### Kimi K2

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Kimi K2 (1T MoE) | 256–1024 | H100, B200, B300, GB200, GB300 | Muon/Adam optimizer, HybridEP, pipeline layout helpers |

### NemotronH

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Nemotron 3 Nano (30B MoE+Mamba) | 8–16 | H100, B200, B300, GB200, GB300 | TE CUDA graphs (attn+mamba+moe), HybridEP |
| Nemotron 3 Super | 64 | H100, B200, B300, GB200, GB300 | TE CUDA graphs, EP=64 |
| NemotronH 56B | 64 | H100, B200, B300 | TP=2–8, TE graphs (mamba+attn) |

### GPT-OSS

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| GPT-OSS 120B | 64 | H100, B200, GB200 | EP=64, HybridEP on GB200 |

---

## Recommendation Decision Tree

```text
User wants to train a model
│
├─ Know the model name?
│   ├─ Yes → Look up in Library Recipe Index above
│   │   ├─ Has a recipe for their size + mode? → Use it directly
│   │   └─ No exact match? → Use closest size, adjust parallelism
│   └─ No → Ask for model name, size, and HF model ID
│
├─ What's the training goal?
│   ├─ Pretrain → Use *_pretrain_config
│   ├─ SFT (full fine-tune) → Use *_sft_config
│   └─ PEFT (LoRA/DoRA) → Use *_peft_config (lowest GPU requirement)
│
├─ How many GPUs?
│   ├─ 1 GPU → Only PEFT recipes work (TP=1, PP=1)
│   ├─ 8 GPUs (1 node) → Most 8B–16B models, small MoE (EP=8)
│   ├─ 16–64 GPUs → 70B dense, medium MoE
│   └─ 128+ GPUs → 405B+, large MoE (DeepSeek V3, Kimi K2)
│
├─ Want throughput benchmarks?
│   ├─ Yes → Use benchmark recipes (src/megatron/bridge/perf_recipes/)
│   │   ├─ Exact exported recipe → scripts/training/train.sh --recipe <exact function name>
│   │   └─ Selector/specialized workflow → scripts/performance/setup_experiment.py
│   └─ No → Use library recipes (scripts/training/run_recipe.py)
│
└─ Long context?
    ├─ > 8K → Need CP (context parallelism), check *_16k / *_64k / *_128k variants
    └─ ≤ 8K → Default recipes work
```

---

## Adjustment Advice (When Recommending)

### Parallelism Resizing Rules

When the user's GPU count differs from the recipe default:

1. **TP must divide `num_key_value_heads`** (GQA constraint). E.g. if
   `num_key_value_heads=8`, valid TP = {1, 2, 4, 8}.
2. **TP should stay within a single node** (NVLink). TP > 8 requires
   inter-node NVLink (e.g., GB200 NVL72).
3. **PP adds pipeline bubbles.** Minimize PP; only increase when TP alone can't
   fit the model. Use VP (virtual pipeline) to mitigate bubble overhead.
4. **EP doesn't reduce dense-layer memory.** Only expert parameters shard with
   EP. Shared attention/embeddings are replicated. For "OOM with MoE", increase
   EP first, not TP.
5. **SP should be True whenever TP > 1.** It eliminates redundant activation
   copies and is essentially free.
6. **CP requires all-to-all or ring attention.** Check `cp_comm_type`. For
   GQA models, `a2a+p2p` hierarchical CP allows CP > num_kv_heads.
7. **Dense and expert meshes overlap.** Do not multiply TP and EP together.
   The minimum MoE world size is `PP × max(TP × CP, EP × ETP)`. Dense DP is
   `world_size / (TP × PP × CP)` and expert EDP is
   `world_size / (PP × EP × ETP)`; both quotients must be integral, and the
   expert count must be divisible by EP.

### Batch Size Tuning

- Start with the recipe's `micro_batch_size`. If OOM, reduce to 1.
- `global_batch_size` determines learning dynamics. Scale with DP:
  `GBS = micro_batch_size × DP × gradient_accumulation_steps`.
- For MoE, `micro_batch_size=1` is typical at scale.

### Common Pitfalls to Warn About

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| TP > num_kv_heads | Crash: "TP must divide num_query_groups" | Reduce TP to a divisor of num_kv_heads |
| PP without VP | Poor throughput (large bubble) | Set `virtual_pipeline_model_parallel_size` |
| EP too low for large MoE | OOM on expert params | Increase EP; each expert lives on EP/num_experts ranks |
| CUDA graphs + packed sequences | Assert: "CUDA graph accepts only Tensor inputs" | Disable packing or use `local` full-iteration graphs |
| CUDA graphs + full recompute | Assert: "full recompute only with full iteration CUDA graph" | Disable recompute or switch to `local` impl |
| `use_te_rng_tracker` not set | Assert on provider init when CUDA graphs enabled | Set `cfg.model.use_te_rng_tracker = True` and `cfg.rng.te_rng_tracker = True` |
| FSDP + TP > 1 on H100 | Possible comm bottleneck | Prefer FSDP with TP=1 or TP=2 on H100; FSDP shines on GB/B-series |
| Long context without CP | OOM on activations | Add CP=2/4/8; use `*_16k`, `*_64k`, or `*_128k` recipe variants |
| MoE `overlap_grad_reduce` on H100 | May hurt throughput (False in many H100 presets) | Set `overlap_grad_reduce=False` for MoE on H100 |
| VLM SFT missing image data | Runs but produces garbage | Provide actual multimodal dataset or use mock VLM data |
| Qwen35-VL MoE FSDP | Tested on Blackwell only | May not work on H100; validate first |

### Recipe Override Examples

```bash
# Scale Llama3 8B from 2 GPUs to 8 GPUs (increase DP)
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe llama3_8b_pretrain_config \
    --dataset mock

# Run the native 4-GPU Qwen3-MoE 30B PEFT topology
uv run python -m torch.distributed.run --nproc_per_node=4 scripts/training/run_recipe.py \
    --recipe qwen3_30b_a3b_peft_config \
    --dataset tulu3

# Add long context to an existing recipe
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe llama3_8b_pretrain_config \
    --dataset mock \
    'model.seq_length=32768' \
    'model.context_parallel_size=4'

# Enable CUDA graphs on any recipe
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe qwen3_30b_a3b_pretrain_config \
    --dataset mock \
    'model.cuda_graph_impl=transformer_engine' \
    'model.cuda_graph_scope=[attn,moe_router,moe_preprocess]' \
    'model.use_te_rng_tracker=True' \
    'rng.te_rng_tracker=True'
```

---

## Quick Reference: Which Recipe for My Situation?

| I want to... | Start with | GPUs needed |
|---|---|---|
| Try Bridge for the first time | `llama3_8b_pretrain_config` + mock data | 2 |
| Fine-tune a 7-8B model | `llama3_8b_sft_config` or `qwen3_8b_sft_config` | 2–4 |
| LoRA on 1 GPU | `llama3_8b_peft_config` or `qwen3_8b_peft_config` | 1 |
| Pretrain a dense 70B | `llama3_70b_pretrain_config` | 32–64 |
| Train a small MoE | `qwen3_30b_a3b_pretrain_config` | 16 |
| Train a large MoE (235B+) | `qwen3_235b_a22b_pretrain_config` | 256–512 |
| Benchmark text-pretrain throughput | Benchmark recipe via `train.sh --recipe <exact name>` | Exact encoded count |
| Long-context training | `llama3_8b_128k_pretrain_config` or add CP override | 16+ |
| VLM fine-tuning | `qwen3_vl_8b_sft_config` or `gemma3_vl_*_sft_config` | 4–8 |
| Diffusion training | `wan_1_3B_pretrain_config` or `flux_12b_pretrain_config` | 8 |

---

## Code Anchors

| What | Path |
|------|------|
| Library recipes root | `src/megatron/bridge/recipes/` |
| Recipe `__init__.py` (all exports) | `src/megatron/bridge/recipes/__init__.py` |
| Common recipe helpers | `src/megatron/bridge/recipes/common.py` |
| Training entry point | `scripts/training/run_recipe.py` |
| Training Slurm launcher | `scripts/training/train.sh` |
| Benchmark recipes root | `src/megatron/bridge/perf_recipes/` |
| Benchmark compatibility launcher | `scripts/performance/setup_experiment.py` |
| Benchmark recipe helpers | `scripts/performance/utils/utils.py` |
| Benchmark overrides | `scripts/performance/utils/overrides.py` |
