# Beam

Megatron Bridge supports checkpoint conversion, inference, pretraining, full SFT and LoRA PEFT for Beam, a 52-layer
sparse mixture-of-experts language model with about 501B parameters. Three dense layers are followed by 49 MoE layers
with 256 routed experts each (top-8). Attention alternates five sliding-window layers (window 512, RoPE) with one
full-attention NoPE layer, and the last layer uses full attention. See the
[Beam verification card](../../model_verification_cards/beam/card.yaml) for the exact commands, revisions, metrics and
current verification status.

## Supported Variants

| Variant | Hugging Face model | Published weight format |
| --- | --- | --- |
| Beam-1.0-501B-A23B | `reflection/Beam-1.0-501B-A23B` | BF16 |

The commands below pin the public release revision `06641076793dad51b8907f7eb2a14f4e08fd3ec8`.
The verification card retains the exact revisions used for historical runs; its original H200 revision is no longer
publicly resolvable. Those historical measurements do not establish identical results for the public release weights.

## Modeling

Beam uses the builder-backed API. `BeamBridge` translates the Hugging Face configuration directly into
`BeamModelConfig`, and `BeamModelBuilder` constructs a native Megatron-Core `HybridModel`: each Beam layer is an
attention/MLP (or attention/MoE) pair in the hybrid layer pattern. `BeamModelProvider` remains only as a compatibility
path for provider-based callers.

The Beam stack spec preserves the published decoder math: GQA (48 query heads, 8 KV heads) with Q/K RMSNorm, a
full-width sigmoid attention output gate with a per-head bias, sandwich RMSNorms after attention and after the MLP,
depth-scaled residual connections, SSMax query scaling on the NoPE layers, grouped sigmoid routing (8 groups, top-4
groups, normalized top-8 weights, routed scaling 2.5) with an expert bias and FP32 router parameters, and a tanh
soft cap of 30 on the output logits. The bridge maps the expert-bias update rate to zero, so fine-tuning and the
bounded pretraining recipe keep the published router bias fixed and add no auxiliary load-balancing loss.

## Builder-backed configuration

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "reflection/Beam-1.0-501B-A23B", revision="06641076793dad51b8907f7eb2a14f4e08fd3ec8"  # pragma: allowlist secret
)
model_config = bridge.get_model_config()

# Apply deployment-specific parallelism before construction.
model_config.tensor_model_parallel_size = 2
model_config.expert_model_parallel_size = 16
models = bridge.get_model(model_config)
```

## Runtime Requirements

Beam runs on the repository's default Megatron-Core pin. The bridge adds Beam's SSMax scaling, attention-gate bias,
sandwich norms, depth-scaled residuals and FP32 router parameters on top of an unmodified Megatron-Core
`HybridModel`, so no Megatron-LM branch switch is needed. Bridge can load the released checkpoint metadata through
its configuration fallback; Hugging Face reference inference requires a Transformers build that implements Beam.
The historical H200 verification runs used the `nvcr.io/nvidia/pytorch:26.08-py3` base container; see the
[repository installation instructions](../../../README.md#-installation).

The Slurm launchers require an account, partition, and container image. Set them before running any
`--executor slurm` conversion or `train.sh` command:

```bash
export SLURM_ACCOUNT=your-account
export SLURM_PARTITION=your-partition
export CONTAINER_IMAGE=your-container-image
```

The conversion and Megatron inference examples below adapt the historical verification commands to the public release
revision and shorter paths. The historical runs used H200 (CPU conversion on a B300 host). The training commands are shortened from the
card's commands, which add the pinned dataset revision, seeds, NaN checks and logging options.

## Conversion

Bridge includes a configuration-only fallback for Transformers builds without native Beam registration.
It supports Bridge model construction and checkpoint conversion; HF model inference still requires a
Transformers build that implements `BeamForCausalLM`. Existing native Beam registrations take precedence.

The BF16 checkpoint is about 1 TB. GPU import and export on H200 use two nodes at EP16, which places about 69 GiB of
weights on each GPU; EP8 on one node would need about 127 GiB per GPU and leaves too little headroom.

```bash
# HF -> Megatron (2 x 8 H200)
./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 2 --gpus-per-node 8 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 06641076793dad51b8907f7eb2a14f4e08fd3ec8 \
  --megatron-path work/models/beam --torch-dtype bfloat16 \
  --tp 1 --pp 1 --ep 16 --etp 1 --distributed-timeout-minutes 110 --low-memory-save

# Megatron -> HF (2 x 8 H200)
./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 2 --gpus-per-node 8 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 06641076793dad51b8907f7eb2a14f4e08fd3ec8 \
  --megatron-path work/models/beam/iter_0000000 \
  --hf-path work/models/beam-hf-export --torch-dtype bfloat16 \
  --tp 1 --pp 1 --ep 16 --etp 1 --distributed-timeout-minutes 110 \
  --distributed-save --save-every-n-ranks 1 --no-progress
```

The historical GPU round trip reproduced all 38,366 source tensors exactly; `config.json` differs only in `transformers_version`.

CPU import and export run in a single process on a host with well over 1 TB of memory and need one runtime GPU
(`--gpus-per-node 1`): Beam's attention layers are Transformer Engine modules, and Transformer Engine refuses to
construct them without CUDA even though the weights stay on CPU. The GPU holds only the CUDA context.

```bash
./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --gpus-per-node 1 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 06641076793dad51b8907f7eb2a14f4e08fd3ec8 \
  --megatron-path work/models/beam-cpu --torch-dtype bfloat16

./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --gpus-per-node 1 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 06641076793dad51b8907f7eb2a14f4e08fd3ec8 \
  --megatron-path work/models/beam-cpu/iter_0000000 \
  --hf-path work/models/beam-cpu-hf-export --torch-dtype bfloat16
```

## Inference

Megatron generation uses the non-cached legacy full-prefix path on 2 x 8 H200:

```bash
./scripts/inference/infer.sh --nodes 2 --gpus-per-node 8 --task legacy-full-prefix-generation \
  --hf_model_path reflection/Beam-1.0-501B-A23B \
  --hf-revision 06641076793dad51b8907f7eb2a14f4e08fd3ec8 \
  --megatron_model_path work/models/beam/iter_0000000 \
  --tp 1 --pp 1 --ep 16 --etp 1 \
  --prompt 'Explain why the sky appears blue in one sentence.' \
  --max_new_tokens 32 --apply-chat-template --legacy-full-prefix
```

Greedy generation is deterministic across runs. Beam's chat template opens each response in the analysis channel,
so short token budgets end before the final channel.

To compare the imported checkpoint with Hugging Face, use `--task model-comparison` with reference logits from a BF16
Hugging Face forward of the same revision (`--hf-logits-path`). On the prompt "The capital of France is the city of"
the historical comparison predicted " Paris" in both implementations with cosine similarity 0.9997.

For Hugging Face inference on a single 8 x H200 node, cap each GPU at 110 GiB so Transformers offloads the remaining
layers to CPU. Uncapped `--device-map auto` placement runs out of memory while it stacks the per-expert weights, and
a 125 GiB cap leaves only about 1.5 GiB of headroom when the loader stacks two layers at once:

```bash
./scripts/inference/infer.sh --task hf-inference --nodes 1 --gpus-per-node 8 \
  --hf-model work/models/beam-hf-export \
  --prompt 'Write one concise sentence explaining why deterministic tests matter:' \
  --max-new-tokens 16 --chat-template --device-map auto --max-gpu-memory-gib 110
```

## Recipes

Hardware-qualified library recipes are defined under [`recipes/beam`](../../../src/megatron/bridge/recipes/beam):

| Recipe | Layout | Purpose |
| --- | --- | --- |
| `beam_pretrain_128gpu_h200_bf16_config` | 16 nodes, TP2 / EP128, SP | Bounded pretraining; BF16 gradients and Adam moments with FP32 master weights |
| `beam_sft_128gpu_h200_bf16_config` | 16 nodes, TP8 / PP2 / EP64, SP | Full SFT on 8K packed sequences, GBS 8, DP 8 |
| `beam_peft_16gpu_h200_bf16_config` | 2 nodes, TP2 / EP16, SP | LoRA (rank 8, alpha 16) on `linear_qkv` and `linear_proj`, 8K packed sequences, GBS 8 |

These H200 recipes resolve the model and tokenizer from the pinned Hub revision; pass `hf_path` to use a local
checkpoint instead. They use the all-to-all MoE dispatcher and full recompute. Use expert parallelism first: the
experts hold about 493B of the 501B parameters, so pipeline parallelism does not lower the memory floor. All-to-all
context parallelism needs the KV heads per tensor-parallel rank to be divisible by the context-parallel size, so CP8
runs at TP1. With context parallelism, SFT switches to per-token loss normalization, which is why the 8K SFT
layout reaches DP8 without it.

### GB200 library recipes

The GB200 variants use four GPUs per node within one NVLink domain. Pretraining, full SFT and LoRA have each
completed 100-step validation with the public checkpoint.

| Recipe | GPUs | TP / PP / CP / EP / ETP | Sequence length | GBS / MBS |
| --- | --- | --- | --- | --- |
| `beam_pretrain_64gpu_gb200_bf16_config` | 64 | 4 / 1 / 1 / 64 / 1 | 4096 | 256 / 1 |
| `beam_sft_64gpu_gb200_bf16_config` | 64 | 8 / 1 / 1 / 64 / 1 | 8192 | 8 / 1 |
| `beam_peft_32gpu_gb200_bf16_config` | 32 | 4 / 1 / 1 / 32 / 1 | 8192 | 8 / 1 |

These variants use HybridEP with natural routing, fixed expert biases, BF16 computation, and FP32 main gradients,
master parameters and Adam moments. Pretraining captures attention and router CUDA graphs and selectively
recomputes only expert activations to provide memory headroom. It does not recompute the expert projections or
communication. Full SFT and LoRA use neither activation recompute nor CUDA graphs. Unlike H200 pretraining, the GB200
pretraining recipe does not use BF16 gradients or optimizer moments; performance comparisons must account for this
precision difference. Full SFT and LoRA retain the H200 learning rates, schedules, dataset selection and adapter scope.

All three recipes collect Python garbage every 10 updates, disable activation fusion, and set
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. Matched 20-step finetuning comparisons retain the same checkpoint, data packs,
learning-rate schedule, natural routing and FP32 optimizer state. GC every 5 updates did not improve throughput;
SM16 with automatic garbage collection did not outperform the selected GC10/SM32 configuration. The combination
of GC10 and SM16 was not tested for finetuning.

All three final runs completed 100 finite updates with no skipped or NaN updates in NeMo 26.10 RC5.
Checkpoint writes and validation were disabled for these measurements. Measurements cover updates 51–100, including five garbage-collection cycles and intervening logging. Each rate
uses the slowest rank's synchronized elapsed time and Bridge's hybrid-layer model-FLOP estimate; token rates count
physical processed tokens, including padding.

| Recipe | Seconds/update | Physical tokens/s/GPU | Model TFLOP/s/GPU | Minimum observed free memory | First / last 10-update mean loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretraining, 64 GPUs | 7.7679 | 2109.18 | 291.94 | 11.36 GiB | 1.1828 / 1.0398 |
| Full SFT, 64 GPUs | 1.8025 | 565.84 | 77.72 | 9.28 GiB | 1.0766 / 0.7391 |
| LoRA, 32 GPUs | 1.4390 | 1419.52 | 194.95 (full-training equivalent) | 87.90 GiB | 1.0778 / 0.8389 |

The LoRA model-work estimate uses the full-training FLOP formula. It is not actual executed LoRA compute or MFU.
These runs validate training behavior; they do not establish checkpoint save, adapter export or resume behavior.

Matched pretraining screens selected attention/router graphs with expert-activation recompute: graphs alone
left less than 4 GiB of observed free memory. SM16 and SM32 were effectively tied, so SM32 was retained without a
performance-superiority claim. TP2/PP2/EP32, activation fusion and TP overlap were rejected after their respective
memory or numerical comparison failures. Variable baseline timing limits repeatable-speedup claims.

Offline packing aligns individual sequences to 8 tokens for SFT and 4 for LoRA. The LoRA alignment differs from the
H200 recipe's alignment of 2 and can change pack membership and order. Prepare a separate dataset output for each
alignment. Matched performance comparisons must use the same prepared packs and starting checkpoint.

The pretraining input is `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, training split. Each nonblank row becomes one document in source order,
tokenized with the pinned Beam tokenizer and an appended EOD token: 1,165,029 documents and 125,410,048 tokens.
Finetuning uses the recipes' pinned Tulu3 revision `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`, split
`train[:10000]`, dataset seed 1234 and model seed 5678. The separate SFT and LoRA packing outputs contain 503 and
500 packs, respectively. Pretraining uses seed 1234 for both the dataset and model.

Fine-grained expert communication overlap is disabled because its layer schedule does not execute Beam's sandwich
norms and depth-scaled residuals. Enabling that schedule would change the model's computation.

#### Launching the GB200 recipes

Use the imported public checkpoint and keep all GPUs within one NVLink domain. On a Slurm cluster with segment
placement, request `segment=16` for pretraining/SFT and `segment=8` for LoRA, and verify the allocated nodes share
one domain. The examples below use four GPUs per node, 16 CPU cores per rank, and 900 GiB of host memory per node.
Set the account, partition and container image as described above, and mount the checkpoint and dataset paths.

These 100-step training checks disable checkpoint writes and evaluation. They retain the recipes' performance
settings; pretraining uses a shortened 100-step cosine schedule with 40 warmup updates. Finetuning uses each recipe's
pinned Tulu3 configuration and packing alignment. Keep its dataset configuration by omitting `--dataset`.

```bash
# Pretraining: 16 nodes x 4 GB200; Beam-tokenized indexed input.
./scripts/training/train.sh --nodes 16 --gpus-per-node 4 \
  --additional-slurm-params 'segment=16;cpus-per-task=16;mem=900G' \
  --recipe beam_pretrain_64gpu_gb200_bf16_config --mode pretrain \
  --dataset megatron-indexed --max_steps 100 --warmup_iters 40 \
  'dataset.blend=[["work/data/wikitext103/wikitext103_beam_text_document"],null]' \
  'dataset.split="100,0,0"' dataset.num_workers=8 scheduler.lr_decay_iters=100 \
  --pretrained_checkpoint work/models/beam/iter_0000000 \
  checkpoint.load=null checkpoint.save=null checkpoint.finetune=true \
  checkpoint.load_optim=false checkpoint.load_rng=false \
  validation.eval_iters=0 validation.eval_interval=1000 \
  logger.log_interval=1 dist.distributed_timeout_minutes=120

# Full SFT: 16 nodes x 4 GB200; separate packs with alignment 8.
./scripts/training/train.sh --nodes 16 --gpus-per-node 4 \
  --additional-slurm-params 'segment=16;cpus-per-task=16;mem=900G' \
  --recipe beam_sft_64gpu_gb200_bf16_config --mode sft \
  dataset.hf_output_root=work/data/beam-tulu3-8k-pad8 \
  --pretrained_checkpoint work/models/beam/iter_0000000 \
  checkpoint.load=null checkpoint.save=null checkpoint.finetune=true \
  checkpoint.load_optim=false checkpoint.load_rng=false \
  validation.eval_iters=0 validation.eval_interval=1000 \
  logger.log_interval=1 dist.distributed_timeout_minutes=120

# LoRA: 8 nodes x 4 GB200; separate packs with alignment 4.
./scripts/training/train.sh --nodes 8 --gpus-per-node 4 \
  --additional-slurm-params 'segment=8;cpus-per-task=16;mem=900G' \
  --recipe beam_peft_32gpu_gb200_bf16_config --mode lora \
  dataset.hf_output_root=work/data/beam-tulu3-8k-pad4 \
  --pretrained_checkpoint work/models/beam/iter_0000000 \
  checkpoint.load=null checkpoint.save=null checkpoint.finetune=true \
  checkpoint.load_optim=false checkpoint.load_rng=false \
  validation.eval_iters=0 validation.eval_interval=1000 \
  logger.log_interval=1 dist.distributed_timeout_minutes=120
```

### H200 training examples

The H200 commands below start from the imported Megatron checkpoint, keep Beam's routing natural
(`model.transformer.moe_router_force_load_balancing=false`), and raise the collective timeout because rank 0 packs
the dataset offline while the other ranks wait.

### Supervised Fine-Tuning

```bash
./scripts/training/train.sh --nodes 16 --gpus-per-node 8 \
  --recipe beam_sft_128gpu_h200_bf16_config --mode sft --dataset tulu3 \
  --pretrained_checkpoint work/models/beam/iter_0000000 \
  --max_steps 100 --seq_length 8192 --lr 5e-6 --min_lr 0 --warmup_iters 10 \
  'dataset.hf_dataset.split="train[:10000]"' \
  dataset.enable_offline_packing=true dataset.hf_output_root=work/data/beam-tulu3-8k \
  +dataset.offline_packing_specs.pad_seq_to_mult=8 scheduler.lr_decay_iters=100 \
  model.transformer.moe_router_force_load_balancing=false \
  checkpoint.save_optim=false checkpoint.save_rng=false \
  --save_dir work/results/beam-sft --save_interval 100 \
  dist.distributed_timeout_minutes=120 \
  '++tokenizer.hf_tokenizer_kwargs.revision="06641076793dad51b8907f7eb2a14f4e08fd3ec8"'  # pragma: allowlist secret
```

For 64K sequences, keep the SFT recipe and switch to all-to-all context parallelism of 8 at TP1:
`--seq_length 65536 model.transformer.context_parallel_size=8 model.transformer.tensor_model_parallel_size=1
model.transformer.sequence_parallel=false model.transformer.pipeline_model_parallel_size=2
model.transformer.expert_model_parallel_size=64 +dataset.offline_packing_specs.pad_seq_to_mult=16`.

### Parameter-Efficient Fine-Tuning

```bash
./scripts/training/train.sh --nodes 2 --gpus-per-node 8 \
  --recipe beam_peft_16gpu_h200_bf16_config --mode lora --dataset tulu3 \
  --pretrained_checkpoint work/models/beam/iter_0000000 \
  --max_steps 100 --seq_length 8192 --lr 1e-4 --min_lr 0 --warmup_iters 10 \
  'dataset.hf_dataset.split="train[:10000]"' \
  dataset.enable_offline_packing=true dataset.hf_output_root=work/data/beam-tulu3-8k-pad2 \
  +dataset.offline_packing_specs.pad_seq_to_mult=2 scheduler.lr_decay_iters=100 \
  model.transformer.moe_router_force_load_balancing=false \
  --save_dir work/results/beam-peft --save_interval 100 \
  '++tokenizer.hf_tokenizer_kwargs.revision="06641076793dad51b8907f7eb2a14f4e08fd3ec8"'  # pragma: allowlist secret
```

The adapter checkpoint holds 208 LoRA matrices (52 attention layers x 2 projections x 2 matrices).

### Pretraining

```bash
./scripts/training/train.sh --nodes 16 --gpus-per-node 8 \
  --recipe beam_pretrain_128gpu_h200_bf16_config --mode pretrain \
  --dataset megatron-indexed --seq_length 4096 --max_steps 100 --warmup_iters 40 \
  'dataset.blend=[["work/data/wikitext103/wikitext103_beam_text_document"],null]' \
  scheduler.lr_decay_iters=100 model.transformer.moe_router_force_load_balancing=false \
  --pretrained_checkpoint work/models/beam/iter_0000000 checkpoint.load=null \
  --save_dir work/results/beam-pretrain --save_interval 50 \
  dist.distributed_timeout_minutes=120
```

The dataset must be tokenized with Beam's tokenizer into Megatron's indexed format. To resume, pass
`--load_dir work/results/beam-pretrain` with `checkpoint.ckpt_step=<step>` and a new `--save_dir`.

## Storage

| Artifact | Approximate size |
| --- | --- |
| Hugging Face checkpoint | 1.0 TB |
| Imported BF16 Megatron checkpoint | 934 GiB |
| Each exported BF16 HF checkpoint | 1.0 TB |
| Each model-only SFT checkpoint (`checkpoint.save_optim=false`) | 1.0 TB |
| Each full-state pretraining checkpoint (weights, FP32 master weights, optimizer state) | about 5 TB |
| LoRA adapter checkpoint | 182 MB |

## Known Limitations

- The launcher's distributed CPU export requests no GPU resources and therefore fails for Beam with
  "TransformerEngine needs CUDA". Use the single-process CPU commands above with one runtime GPU.
- Megatron KV-cache inference is not verified; use the legacy full-prefix generation path.
- Only BF16 recipes are provided; no FP8 recipe is provided.
- Hugging Face reference inference requires a Transformers build that implements `BeamForCausalLM`. Bridge
  configuration loading and checkpoint conversion also work with its configuration-only fallback.
