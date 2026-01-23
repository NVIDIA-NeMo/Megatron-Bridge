# Gemma 3 VL - Vision Language Model

This directory contains examples for Gemma 3 Vision Language Model, including checkpoint conversion, inference, and fine-tuning.

## Conversion with 🤗 Hugging Face

See the [conversion.sh](conversion.sh) script for example commands to:
- Importing Hugging Face checkpoints into Megatron format on CPU
- Exporting Megatron checkpoints back to Hugging Face on CPU
- Running a multi-GPU round-trip validation to automatically convert between formats and verify model weights for compatibility between Hugging Face and Megatron-LM checkpoints

```bash
# Import HF → Megatron
python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-3-4b-it \
    --megatron-path /models/gemma-3-4b-it

# Export Megatron → HF
python examples/conversion/convert_checkpoints.py export \
    --hf-model google/gemma-3-4b-it \
    --megatron-path /models/gemma-3-4b-it/iter_0000000 \
    --hf-path ./gemma3-vl-hf-export

# Round-trip validation
torchrun --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id google/gemma-3-4b-it --tp 2 --pp 2
```

You will see the following output when the commands succeed:

**Import command output:**
```
✅ Successfully imported model to: /models/gemma-3-4b-it
📁 Checkpoint structure:
   📄 latest_checkpointed_iteration.txt
   📂 iter_0000000/
   📄 latest_train_state.pt
```

**Export command output:**
```
Success: All tensors from the original checkpoint were written.
✅ Successfully exported model to: /tmp/gemma3-vl-hf-export
📁 Export structure:
...
🔍 You can now load this model with:
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained('/tmp/gemma3-vl-hf-export')
```

**Round-trip validation output:**
```
Success: All tensors from the original checkpoint were written.
```

## Run Inference

See the [inference.sh](inference.sh) script for example commands to:
- Run inference with Hugging Face checkpoints
- Run inference with Megatron checkpoints

```bash
# Inference with Hugging Face checkpoints
torchrun --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-3-4b-it \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 100 \
    --tp 2 \
    --pp 2

# Inference with Megatron checkpoints
torchrun --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-3-4b-it \
    --megatron_model_path /models/gemma-3-4b-it/iter_0000000 \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 100 \
    --tp 2 \
    --pp 2
```

You will see the following output when the commands succeed:

**Inference output:**
```
Describe this image.<end_of_turn>
<start_of_turn>model
Here's a description of the image you sent, breaking down the technical specifications of the H100 SXM and H100 NVL server cards:

**Overall:**

The image is a table comparing the technical specifications of two NVIDIA server cards: the H100 SXM and the H100 NVL. It's designed to highlight the performance differences between the two cards, particularly in terms of compute power and memory.

**Column Breakdown:**

*
=======================================
```

## Pretrain

Pretraining is not supported for this model.

## Finetune Recipes

See the [sft.sh](sft.sh) script for example commands to:
- Full finetuning with all model parameters
- Parameter-Efficient Finetuning (PEFT) with LoRA

```bash
# Finetune
torchrun --nproc_per_node=8 finetune_gemma3_vl.py \
    --recipe gemma3_vl_4b_finetune_config \
    --pretrained-checkpoint /models/gemma-3-4b-it \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=2 \
    model.context_parallel_size=1

# Lora
torchrun --nproc_per_node=8 finetune_gemma3_vl.py \
    --recipe gemma3_vl_4b_finetune_config \
    --pretrained-checkpoint /models/gemma-3-4b-it \
    --peft_scheme lora \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=2 \
    model.context_parallel_size=1
```

You will see the following output when the commands succeed:

**Fine-tuning output:**
```
----------------------------------------------------------------------------------------------------------
 validation loss at iteration 20 on test set | lm loss value: 0.000000E+00 | lm loss PPL: 1.000000E+00 |
----------------------------------------------------------------------------------------------------------
```