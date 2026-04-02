# Nemotron Diffusion

This directory contains recipes for training and running Nemotron Diffusion language models (dLLMs) based on Ministral-3 (3B, 8B, 14B). The full workflow is:

0. **Bridge (Checkpoint Conversion)** — convert a HuggingFace Ministral-3 checkpoint to Megatron-Bridge format, and export trained checkpoints back to HuggingFace.
1. **Continuous Pretraining (CPT)** — standard autoregressive pretraining on the base Ministral-3 model with additional data.
2. **AR-to-DLM** — converts the CPT checkpoint into a diffusion language model using the block diffusion paradigm.
3. **Inference** — run text generation from a trained checkpoint.

---


## Stage 1: Continuous Pretraining (CPT)

CPT fine-tunes a pretrained Ministral-3 model on new data using standard autoregressive cross-entropy loss. This stage adapts the model to the target domain before diffusion training.

**Example script:**
```bash
torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/continuous_pretraining.py \
    --model-size 3b \
    --hf-path mistralai/Ministral-3-3B-Base-2512 \
    --data-paths /path/to/dclm/merged_tokenized_text_document \
    --config-file examples/diffusion/recipes/nemotron_diffusion/conf/cpt_3b.yaml
```

Available config files: [`conf/cpt_3b.yaml`](conf/cpt_3b.yaml), [`conf/cpt_8b.yaml`](conf/cpt_8b.yaml), [`conf/cpt_14b.yaml`](conf/cpt_14b.yaml).

---

## Stage 2: AR-to-DLM

This stage converts the CPT checkpoint into a diffusion LM. It replaces the standard attention with `NemotronDiffusionAttention` and trains with a combined diffusion + AR loss.

**Key recipe:** `examples/diffusion/recipes/nemotron_diffusion/ar_to_dlm.py`

The model is built via `NemotronDiffusionModelProvider`, which extends `Ministral3ModelProvider` with:
- `dlm_paradigm = "sbd_block_diff"` — semi-block diffusion with block masking
- `block_size = 64` — number of tokens per diffusion block
- `mask_token_id = 100` — token ID used for masking during diffusion
- `dlm_loss_weight = 0.3`, `ar_loss_weight = 1.0` — loss weighting between diffusion and AR objectives
- `NemotronDiffusionAttention` replaces core attention to support block-causal masking

The CPT checkpoint from Stage 1 is passed via `checkpoint.pretrained_checkpoint`. Setting `checkpoint.finetune=true` skips loading the optimizer state from the CPT stage.

**Example launch:**
```bash
torchrun --nproc_per_node=8 examples/diffusion/recipes/nemotron_diffusion/ar_to_dlm.py \
    --model-size 3b \
    --hf-path mistralai/Ministral-3-3B-Base-2512 \
    --data-paths /path/to/dclm/merged_tokenized_text_document \
    --config-file examples/diffusion/recipes/nemotron_diffusion/conf/ar_to_dlm_3b_dlm.yaml \
    checkpoint.finetune=true \
    checkpoint.pretrained_checkpoint=/path/to/cpt_checkpoint
```

Available config files: [`conf/ar_to_dlm_3b_dlm.yaml`](conf/ar_to_dlm_3b_dlm.yaml), [`conf/ar_to_dlm_8b_dlm.yaml`](conf/ar_to_dlm_8b_dlm.yaml).

---

## Inference

The script [`inference_nemotron.py`](inference_nemotron.py) runs text generation from a trained Megatron-format NemotronDiffusion checkpoint. Both dLLM (block diffusion) and AR modes are supported.

### dLLM mode (default)

```bash
torchrun --nproc_per_node=4 examples/diffusion/recipes/nemotron_diffusion/inference_nemotron.py \
    --megatron-path /path/to/checkpoints/ar_to_dlm_8b \
    --hf-model mistralai/Ministral-3-8B-Base-2512 \
    --prompts "The capital of France is" \
    --gen-length 256 --block-length 32 --diffusion-steps 256 \
    --tp 4
```

### AR mode

```bash
python examples/diffusion/recipes/nemotron_diffusion/inference_nemotron.py \
    --megatron-path /path/to/checkpoints/ar_to_dlm_3b \
    --hf-model mistralai/Ministral-3-3B-Base-2512 \
    --mode ar \
    --prompts "Once upon a time" \
    --max-new-tokens 128
```

The `--tp` argument must match the tensor parallelism degree of the saved checkpoint (e.g. `--tp 4` for 8B checkpoints saved with TP=4). `--hf-model` is used for the tokenizer and model config only — weights are loaded from `--megatron-path`.

---


## Checkpoint Conversion (Bridge)

The `NemotronDiffusionBridge` converts between HuggingFace `Mistral3ForConditionalGeneration` and Megatron-Bridge distributed checkpoint format. It handles:

- **Language model weights** — mapped between HF (`language_model.model.*`) and Megatron (`language_model.decoder.*`) with proper QKV merging and tensor-parallel sharding.
- **Vision encoder weights** (`vision_tower.**`) — replicated across tensor-parallel ranks (no sharding needed).
- **Multimodal projector weights** (`multi_modal_projector.**`) — replicated similarly.

The conversion script is [`convert_checkpoints.py`](convert_checkpoints.py).

### Import: HuggingFace → Megatron

```bash
python examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py import \
    --hf-model mistralai/Ministral-3-3B-Base-2512 \
    --megatron-path /path/to/checkpoints/hf_to_mb_3b \
    --torch-dtype bfloat16
```

The Megatron checkpoint is written under `--megatron-path` (e.g. `.../hf_to_mb_3b/iter_0000000/`). Use the parent directory for CPT training with `checkpoint.load`.

For the 8B model (TP=4):
```bash
python examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py import \
    --hf-model mistralai/Ministral-3-8B-Base-2512 \
    --megatron-path /path/to/checkpoints/hf_to_mb_8b \
    --torch-dtype bfloat16
```

### Export: Megatron → HuggingFace

Export a trained Megatron checkpoint back to HuggingFace format. A reference HF model is required to provide config and tokenizer artifacts:

```bash
python examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py export \
    --hf-model mistralai/Ministral-3-3B-Base-2512 \
    --megatron-path /path/to/checkpoints/ar_to_dlm_3b \
    --hf-path /path/to/checkpoints/mb_to_hf_3b
```

The `--hf-model` argument is used as the reference for config, tokenizer, and any non-LM artifacts. The exported directory contains a self-contained HuggingFace model.

**Note:** If the reference HF model does not include vision tower weights (e.g. an LM-only checkpoint), warnings of the form `Can't find vision_tower.* in hf_keys` are expected and benign — the LM weights are still exported correctly.

---