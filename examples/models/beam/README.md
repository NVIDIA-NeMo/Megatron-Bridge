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

The commands below pin revision `3ea13c81833d9a523ed4bdc4152a33edd16bbd30`, the revision the verification card records.

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
    "reflection/Beam-1.0-501B-A23B", revision="3ea13c81833d9a523ed4bdc4152a33edd16bbd30"
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
`HybridModel`, so no Megatron-LM branch switch is needed. Loading the Hugging Face checkpoint needs Transformers
5.16.0 or later. The verification runs used the `nvcr.io/nvidia/pytorch:26.08-py3` base container; see the
[repository installation instructions](../../../README.md#-installation).

The Slurm launchers require an account, partition, and container image. Set them before running any
`--executor slurm` conversion or `train.sh` command:

```bash
export SLURM_ACCOUNT=your-account
export SLURM_PARTITION=your-partition
export CONTAINER_IMAGE=your-container-image
```

The conversion and Megatron inference commands below follow the verification card's commands with shorter paths; the
card's commands ran on H200 (CPU conversion on a B300 host) through the launchers' Python entry points. The training commands are shortened from the
card's commands, which add the pinned dataset revision, seeds, NaN checks and logging options.

## Conversion

The BF16 checkpoint is about 1 TB. GPU import and export on H200 use two nodes at EP16, which places about 69 GiB of
weights on each GPU; EP8 on one node would need about 127 GiB per GPU and leaves too little headroom.

```bash
# HF -> Megatron (2 x 8 H200)
./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 2 --gpus-per-node 8 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 3ea13c81833d9a523ed4bdc4152a33edd16bbd30 \
  --megatron-path work/models/beam --torch-dtype bfloat16 \
  --tp 1 --pp 1 --ep 16 --etp 1 --distributed-timeout-minutes 110 --low-memory-save

# Megatron -> HF (2 x 8 H200)
./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 2 --gpus-per-node 8 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 3ea13c81833d9a523ed4bdc4152a33edd16bbd30 \
  --megatron-path work/models/beam/iter_0000000 \
  --hf-path work/models/beam-hf-export --torch-dtype bfloat16 \
  --tp 1 --pp 1 --ep 16 --etp 1 --distributed-timeout-minutes 110 \
  --distributed-save --save-every-n-ranks 1 --no-progress
```

The GPU round trip reproduces all 38,366 source tensors exactly; `config.json` differs only in `transformers_version`.

CPU import and export run in a single process on a host with well over 1 TB of memory and need one runtime GPU
(`--gpus-per-node 1`): Beam's attention layers are Transformer Engine modules, and Transformer Engine refuses to
construct them without CUDA even though the weights stay on CPU. The GPU holds only the CUDA context.

```bash
./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --gpus-per-node 1 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 3ea13c81833d9a523ed4bdc4152a33edd16bbd30 \
  --megatron-path work/models/beam-cpu --torch-dtype bfloat16

./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --gpus-per-node 1 \
  --hf-model reflection/Beam-1.0-501B-A23B \
  --hf-revision 3ea13c81833d9a523ed4bdc4152a33edd16bbd30 \
  --megatron-path work/models/beam-cpu/iter_0000000 \
  --hf-path work/models/beam-cpu-hf-export --torch-dtype bfloat16
```

## Inference

Megatron generation uses the non-cached legacy full-prefix path on 2 x 8 H200:

```bash
./scripts/inference/infer.sh --nodes 2 --gpus-per-node 8 --task legacy-full-prefix-generation \
  --hf_model_path reflection/Beam-1.0-501B-A23B \
  --hf-revision 3ea13c81833d9a523ed4bdc4152a33edd16bbd30 \
  --megatron_model_path work/models/beam/iter_0000000 \
  --tp 1 --pp 1 --ep 16 --etp 1 \
  --prompt 'Explain why the sky appears blue in one sentence.' \
  --max_new_tokens 32 --apply-chat-template --legacy-full-prefix
```

Greedy generation is deterministic across runs. Beam's chat template opens each response in the analysis channel,
so short token budgets end before the final channel.

To compare the imported checkpoint with Hugging Face, use `--task model-comparison` with reference logits from a BF16
Hugging Face forward of the same revision (`--hf-logits-path`). On the prompt "The capital of France is the city of"
both predict " Paris" with cosine similarity 0.9997.

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

All three recipes resolve the model and tokenizer from the pinned Hub revision; pass `hf_path` to use a local
checkpoint instead. They use the all-to-all MoE dispatcher and full recompute. Use expert parallelism first: the
experts hold about 493B of the 501B parameters, so pipeline parallelism does not lower the memory floor. All-to-all
context parallelism needs the KV heads per tensor-parallel rank to be divisible by the context-parallel size, so CP8
runs at TP1. With context parallelism, SFT switches to per-token loss normalization, which is why the 8K SFT
layout reaches DP8 without it.

All training commands below start from the imported Megatron checkpoint, keep Beam's routing natural
(`model.transformer.moe_router_force_load_balancing=false`), and raise the collective timeout because rank 0 packs
the dataset offline while the other ranks wait.

### Supervised Fine-Tuning

```bash
./scripts/training/train.sh --nodes 16 --gpus-per-node 8 \
  --recipe beam_sft_128gpu_h200_bf16_config --mode sft --dataset tulu3 \
  --pretrained_checkpoint work/models/beam/iter_0000000 \
  --max_steps 100 --seq_length 8192 --lr 5e-6 --min_lr 0 --warmup_iters 10 \
  'dataset.hf_dataset.split="train[:10000]"' \
  '++tokenizer.hf_tokenizer_kwargs.revision="3ea13c81833d9a523ed4bdc4152a33edd16bbd30"' \
  dataset.enable_offline_packing=true dataset.hf_output_root=work/data/beam-tulu3-8k \
  +dataset.offline_packing_specs.pad_seq_to_mult=8 scheduler.lr_decay_iters=100 \
  model.transformer.moe_router_force_load_balancing=false \
  checkpoint.save_optim=false checkpoint.save_rng=false \
  --save_dir work/results/beam-sft --save_interval 100 \
  dist.distributed_timeout_minutes=120
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
  '++tokenizer.hf_tokenizer_kwargs.revision="3ea13c81833d9a523ed4bdc4152a33edd16bbd30"' \
  dataset.enable_offline_packing=true dataset.hf_output_root=work/data/beam-tulu3-8k-pad2 \
  +dataset.offline_packing_specs.pad_seq_to_mult=2 scheduler.lr_decay_iters=100 \
  model.transformer.moe_router_force_load_balancing=false \
  --save_dir work/results/beam-peft --save_interval 100
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
- Only BF16 recipes are provided; FP8 and Blackwell recipes are not part of this release.
