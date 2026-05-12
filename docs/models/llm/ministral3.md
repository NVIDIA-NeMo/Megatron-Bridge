# Ministral 3

The [Ministral 3](https://huggingface.co/collections/mistralai/ministral-3-67e0e02b00ad7e4f6c8e9a09)
family from **Mistral AI** is an edge-optimized line of compact models
combining a language model with a vision encoder for multimodal
capabilities. Ministral 3 is available in 3B, 8B, and 14B parameter
variants, ships in Base / Instruct / Reasoning forms, and supports a
context window of up to 256k tokens.

Ministral 3 inherits its core architecture from Mistral (grouped-query
attention, RMSNorm, SwiGLU MLP, rotary position embeddings) so it is
supported through the Bridge system as an extension of the Mistral
provider — checkpoint conversion, SFT, and PEFT all "just work" with the
same Bridge surface as the rest of the Mistral family.

## Model Architecture

| Variant | Layers | Hidden size | FFN hidden | Heads | Query groups (GQA) | Default seq length |
|---|---|---|---|---|---|---|
| Ministral-3-3B  | 26 | 3072 | 9216  | 32 | 8 | 32k (extensible to 256k) |
| Ministral-3-8B  | 34 | 4096 | 14336 | 32 | 8 | 32k (extensible to 256k) |
| Ministral-3-14B | 40 | 5120 | 16384 | 32 | 8 | 32k (extensible to 256k) |

Shared properties across the family:

- **Attention**: GQA with 32 query heads and 8 key/value groups, RoPE
- **MLP**: SwiGLU activation (`gated_linear_unit=True`)
- **Normalization**: RMSNorm
- **Tokenizer**: Mistral tokenizer (vocab size matches HF checkpoint)

## Conversion with 🤗 Hugging Face

### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Ministral-3-8B-Base
bridge = AutoBridge.from_hf_pretrained("mistralai/Ministral-3-8B-Base-2512")
provider = bridge.to_megatron_provider()

# Configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

The 3B variant runs comfortably with `TP=1, PP=1` on a single H100 80GB. The
14B variant typically wants `TP=2` or higher.

### Export Megatron → HF

```python
# Convert from a Megatron checkpoint directory to HF format
bridge.export_ckpt(
    megatron_path="/results/ministral3_8b/checkpoints/iter_0010000",
    hf_path="./ministral3-8b-hf-export",
)
```

## Examples

- Checkpoint conversion: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)

## Recipes

The Bridge ships SFT and PEFT recipes for all three variants. They use the
parameterless recipe API — pass overrides on the CLI rather than as
function arguments.

| Variant | SFT recipe | PEFT recipe |
|---|---|---|
| 3B  | `ministral3_3b_sft_config`  | `ministral3_3b_peft_config`  |
| 8B  | `ministral3_8b_sft_config`  | `ministral3_8b_peft_config`  |
| 14B | `ministral3_14b_sft_config` | `ministral3_14b_peft_config` |

```python
from megatron.bridge.recipes.ministral3 import ministral3_8b_peft_config

cfg = ministral3_8b_peft_config(peft_scheme="lora")
```

See [`bridge.recipes.ministral3`](../../apidocs/bridge/bridge.recipes.ministral3.md)
for the full module reference and [`peft.md`](../../training/peft.md) for
LoRA / DoRA configuration knobs that apply to these recipes.

## Notes

- The Bridge defines providers for the **language tower only**. The vision
  encoder side of Ministral 3 is wired up by the recipes when the corresponding
  HF checkpoint is loaded.
- Since Ministral 3 extends the Mistral provider, any parallelism / mixed
  precision / packed-sequence pattern that works for Mistral applies here too
  — see [`../../recipe-usage.md`](../../recipe-usage.md).
