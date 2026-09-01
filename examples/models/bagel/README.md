# BAGEL Training

Megatron Bridge supports BAGEL-7B-MoT pretraining and fine-tuning through
Megatron MIMO. The integration covers the T2I, Editing, and VLM objectives,
native BAGEL checkpoint initialization, Megatron FSDP training, and loader
checkpoint restore.

## Requirements

- A Megatron-LM revision containing
  [NVIDIA/Megatron-LM#3635](https://github.com/NVIDIA/Megatron-LM/pull/3635).
- The NeMo container documented by the verification card, which provides the
  compatible Torch, TorchVision, and FlashAttention runtime.
- The `ByteDance-Seed/BAGEL-7B-MoT` model assets, including
  `ae.safetensors` and tokenizer files.
- BAGEL data converted and prepared as described in the
  [BAGEL data tutorial](../../../tutorials/data/bagel/README.md).

Install the optional Python dependencies and clone the official runtime source:

```bash
uv sync --extra bagel
git clone https://github.com/ByteDance-Seed/Bagel.git work/dependencies/Bagel
git -C work/dependencies/Bagel checkout a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f
```

The checkout remains a runtime dependency because Bridge calls its tokenizer,
image transforms, and flow-matching utilities directly. Imports are lazy, so
other Bridge models do not require the extra or the checkout. BAGEL commands
fail with an installation hint when either is unavailable.

## Launch Pretraining

The following example uses the 32-GPU H100 recipe and a previously initialized
Megatron checkpoint. Paths under `work/` are user-managed inputs:

```bash
./scripts/training/train.sh --nodes 4 --gpus-per-node 8 \
  --recipe bagel_7b_pretrain_32gpu_h100_bf16_config --mode pretrain \
  --max_steps 30 \
  model.bagel_repo=work/dependencies/Bagel \
  model.model_path=work/models/BAGEL-7B-MoT \
  model.vae_path=work/models/BAGEL-7B-MoT/ae.safetensors \
  dataset.dataset_root=work/data/bagel-wds \
  dataset.bagel_repo=work/dependencies/Bagel \
  dataset.tokenizer_model=work/models/BAGEL-7B-MoT \
  checkpoint.load=work/checkpoints/bagel-mcore-init \
  dataset.dataloader_load=work/checkpoints/bagel-mcore-init
```

Use `bagel_7b_pretrain_8gpu_h100_bf16_config` for the 8-GPU pretraining
configuration or `bagel_7b_finetune_8gpu_h100_bf16_config` for fine-tuning.
The lower-level `pretrain_bagel.py` example additionally supports initializing
from a native BAGEL checkpoint and recording loss-alignment traces.

The standard `throughput/tflops/device` logger uses official BAGEL's Qwen2
formula with the data-parallel-global packed sequence length and sum of squared
sample lengths. It reports theoretical training TFLOP/s per GPU. As in the
official implementation, this numerator does not separately count the vision
encoder, VAE, EMA, optimizer, or activation recomputation.

## Image-Understanding Inference

Run deterministic greedy inference from a converted Megatron checkpoint:

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  examples/models/bagel/inference_bagel.py \
  --bagel-repo work/dependencies/Bagel \
  --hf-model ByteDance-Seed/BAGEL-7B-MoT \
  --hf-revision 5019f57d168e5816e8f3f701b17cc816bb7cf24b \
  --checkpoint work/checkpoints/bagel-mcore-init \
  --image work/images/example.jpg \
  --prompt "Describe this image."
```

The example uses BAGEL's official tokenizer and image preprocessing. It
recomputes the short sequence for each generated token because the current
MCore BAGEL wrapper does not expose a maintained KV-cache interface.

## Current Limitations

- Tensor, pipeline, and context parallel sizes must all be one.
- The maintained recipes use Megatron FSDP with data parallelism.
- Image generation and Hugging Face export are not provided. The maintained
  understanding inference example currently supports TP=PP=CP=1 only.
- Official and Bridge throughput results are not directly comparable when
  recompute or EMA settings differ.
