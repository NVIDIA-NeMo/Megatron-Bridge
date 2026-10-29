# NemotronLabsDiffusion

This directory contains recipes for training and running NemotronLabsDiffusion language models (dLLMs) based on Ministral-3 (3B, 8B, 14B). The full workflow is:

0. **Bridge (Checkpoint Conversion)** — convert a HuggingFace Ministral-3 checkpoint to Megatron-Bridge format.
1. **Continuous Pretraining (CPT)** — standard autoregressive pretraining on the base Ministral-3 model with additional data.
2. **AR-to-DLM** — converts the CPT checkpoint into a diffusion language model using the block diffusion paradigm.
3. **Inference** — run text generation from a trained checkpoint.

---


## Stage 1: Continuous Pretraining (CPT)

CPT fine-tunes a pretrained Ministral-3 model on new data using standard autoregressive cross-entropy loss. This stage adapts the model to the target domain before diffusion training.

**Example script:**
```bash
torchrun --nproc_per_node=8 examples/models/nemotron_labs_diffusion/continuous_pretraining.py \
    --model-size 3b \
    --hf-path mistralai/Ministral-3-3B-Base-2512 \
    --data-paths /path/to/dclm/merged_tokenized_text_document
```


---

## Stage 2: AR-to-DLM

This stage converts the CPT checkpoint into a diffusion LM. It replaces the standard attention with `NemotronLabsDiffusionAttention` and trains with a combined diffusion + AR loss.

**Key recipe:** `examples/models/nemotron_labs_diffusion/ar_to_dlm.py`

The model is built via `NemotronLabsDiffusionModelProvider`, which extends `Ministral3ModelProvider` with:
- `dlm_paradigm = "sbd_block_diff"` — attention with block masking
- `block_size = 64` — number of tokens per diffusion block
- `mask_token_id = 100` — token ID used for masking during diffusion
- `dlm_loss_weight = 0.3`, `ar_loss_weight = 1.0` — loss weighting between diffusion and AR objectives
- `NemotronLabsDiffusionAttention` replaces core attention to support block-causal masking

The CPT checkpoint from Stage 1 is passed via `checkpoint.pretrained_checkpoint`. Setting `checkpoint.finetune=true` skips loading the optimizer state from the CPT stage.

**Example launch:**
```bash
torchrun --nproc_per_node=8 examples/models/nemotron_labs_diffusion/ar_to_dlm.py \
    --model-size 3b \
    --hf-path mistralai/Ministral-3-3B-Base-2512 \
    checkpoint.finetune=true \
    checkpoint.pretrained_checkpoint=/path/to/cpt_checkpoint \
    --data-paths /path/to/dclm/merged_tokenized_text_document
```


---

## Inference

The script [`inference_nemotron_labs_diffusion.py`](inference_nemotron_labs_diffusion.py) runs text generation from a trained Megatron-format NemotronLabsDiffusion checkpoint. Both dLLM (block diffusion) and AR modes are supported.

### dLLM mode (default)

```bash
torchrun --nproc_per_node=4 examples/models/nemotron_labs_diffusion/inference_nemotron_labs_diffusion.py \
    --megatron-path /path/to/checkpoints/ar_to_dlm_3b/iter_xxxxxxx \
    --hf-model mistralai/Ministral-3-3B-Base-2512 \
    --prompts "The capital of France is" \
    --gen-length 256 --block-length 32 --steps-per-block 32
```

### AR mode

```bash
python examples/models/nemotron_labs_diffusion/inference_nemotron_labs_diffusion.py \
    --megatron-path /path/to/checkpoints/ar_to_dlm_3b/iter_xxxxxxx \
    --hf-model mistralai/Ministral-3-3B-Base-2512 \
    --mode ar \
    --prompts "Once upon a time" \
    --max-new-tokens 128
```

You can pass `--tp` argument, but it must match the tensor parallelism degree of the saved checkpoint (e.g. `--tp 4` for 8B checkpoints saved with TP=4). `--hf-model` is used for the tokenizer and model config only — weights are loaded from `--megatron-path`.

---


## Checkpoint Conversion (Bridge)

The `NemotronLabsDiffusionBridge` converts between HuggingFace `NemotronLabsDiffusionModel` and Megatron-Bridge distributed checkpoint format. It handles:

- **Language model weights** — mapped between HF (`encoder.*`) and Megatron (`language_model.decoder.*`) with proper QKV merging and tensor-parallel sharding.
- **Diffusion head** (`diffusion_head.weight`) — mapped to Megatron's `language_model.output_layer.weight`.

The conversion script is [`convert_checkpoints.py`](convert_checkpoints.py).

### Import: HuggingFace → Megatron

```bash
python examples/models/nemotron_labs_diffusion/convert_checkpoints.py import \
    --hf-model nvidia/Nemotron-Labs-Diffusion-3B \
    --megatron-path /path/to/checkpoints/hf_to_mb_3b \
    --torch-dtype bfloat16
```

The Megatron checkpoint is written under `--megatron-path` (e.g. `.../hf_to_mb_3b/iter_0000000/`). Use the parent directory for training with `checkpoint.load`.

### Export: Megatron → HuggingFace

Export a trained Megatron checkpoint back to HuggingFace format. A reference HF model is required to provide config and tokenizer artifacts:

```bash
python examples/models/nemotron_labs_diffusion/convert_checkpoints.py export \
    --hf-model nvidia/Nemotron-Labs-Diffusion-3B \
    --megatron-path /path/to/checkpoints/ar_to_dlm_3b \
    --hf-path /path/to/checkpoints/mb_to_hf_3b
```

The `--hf-model` argument is used as the reference for config, tokenizer, and any non-LM artifacts. The exported directory contains a self-contained HuggingFace model.

---
