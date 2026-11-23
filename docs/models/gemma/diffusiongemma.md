# DiffusionGemma

Megatron Bridge supports the public `google/diffusiongemma-26B-A4B-it`
checkpoint for block-diffusion SFT.

## What is implemented

- HF ↔ Megatron conversion and exact checkpoint tensor coverage.
- Text and image-conditioned encoder/decoder inference.
- Uniform-state corruption with one diffusion time per example.
- `base-sft`, `reweighted-ce`, `loo-ce`, and `reweighted-loo-ce` objectives.
- Detached two-pass self-conditioning.
- Optional clean-target encoder next-token loss. It runs a separate encoder
  pass so the denoising decoder never sees the target tokens as context.
- Tensor parallelism with sequence parallelism and expert parallelism.
- Distributed optimizer training and torch-dist checkpoint save/resume.

The reference SFT objective does not add a router balancing penalty. The
provider exposes `moe_aux_loss_coeff` for experiments, but its default is zero.

## Recipe

Use `diffusion_gemma_26b_sft_config()` from
`megatron.bridge.recipes.diffusion_gemma`. The production recipe uses TP=1,
EP=8, one pipeline stage, BF16 mixed precision, a distributed optimizer, and
full uniform activation recomputation. The recomputation setting keeps
large-image prompts with long text and prior-target context within H200
memory; callers can override it for shorter inputs.
Pass `train_path` and optional `validation_path`/`test_path` to train on
existing JSONL split files; the adapter never creates a new split. Without
`train_path`, the recipe uses synthetic smoke-test data. Each JSONL row has the
form:

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"images/example.png"},{"type":"text","text":"Describe the image."}]}],"completion":"A blue square above a red circle."}
```

Image paths are local and relative to the JSONL file. The caller is responsible
for supplying only instructions/context available at inference time and for
curating completion labels. Multi-block completions are split without
truncation; only prior clean target blocks are appended to the encoder context.

```python
from megatron.bridge.recipes.diffusion_gemma import diffusion_gemma_26b_sft_config
from megatron.bridge.training.diffusion_gemma_step import forward_step
from megatron.bridge.training.pretrain import pretrain

cfg = diffusion_gemma_26b_sft_config(
    train_path="data/train.jsonl", validation_path="data/validation.jsonl",
    test_path="data/test.jsonl",
)
pretrain(cfg, forward_step)
```

Run this entry point with `uv run python -m torch.distributed.run
--nproc_per_node=8 train.py`. Initial weights are loaded through the HF bridge
hook; `checkpoint.load` points to the resume directory, not the HF model ID.

The dataset must emit the following fields:

```text
input_ids, attention_mask, position_ids, mm_token_type_ids,
canvas_ids, canvas_mask,
pixel_values, image_position_ids  # optional for image examples
```

Prompts are left-padded; canvases are right-padded. When sequence parallelism
is enabled, the prompt length must be divisible by TP.

## Validation evidence

The tiny model has HF loss/gradient parity at roughly 1e-3 relative gradient
error and overfits a fixed batch. TP=2 and EP=2 checks pass, including a
prompt crossing the toy sliding-window boundary. A real 26B BF16
run on eight H200 GPUs completed three unfrozen optimizer steps with EP=8 and a
distributed optimizer; no updates were skipped or non-finite. After warmup,
the measured steps were about 4.6–6.1 seconds and peak rank-0 allocated memory
was about 74.6 GiB.
With the clean-target encoder objective enabled, the three-step smoke also
passed: warmed-up steps measured 6.1–8.4 seconds and peak rank-0 allocated
memory was about 83.6 GiB. These are small synthetic microbatches, not a
prediction of production throughput. No 26B optimizer checkpoint
was written during either smoke run.
On a synthetic long-context multimodal example with a large image, a long text
prompt, and a multi-block completion, the recipe completed three EP=8 steps
with recomputation enabled. Peak rank-0 allocation was 123.8 GiB and warmed-up
steps took about 3.4 seconds. The same example without recomputation exceeded
the 139.8 GiB H200 capacity during attention.

The two-GPU `pretrain()` checkpoint test saved at step three and resumed in
a fresh process. Steps four through six reproduced the uninterrupted losses,
gradient norms, and consumed-sample positions exactly.

## Known limitations

- Context parallelism and pipeline parallelism are not implemented by the
  diffusion training step.
- Context lengths beyond the validated sliding-window boundary still need
  production-shaped memory and quality evaluation.
- `MockDiffusionGemmaDatasetConfig` is for smoke tests only. Pass
  `train_path` to use `DiffusionGemmaDatasetConfig` with curated JSONL input files.
- A successful optimizer smoke test does not demonstrate task quality or
  generalization. Those require held-out task evaluation on a saved split.
