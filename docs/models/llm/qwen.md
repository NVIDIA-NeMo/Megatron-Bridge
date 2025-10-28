# Qwen

Qwen2/2.5/3 models are supported via the Bridge with QK layernorm handling (Qwen3) and bias in QKV (Qwen2).

## Conversion with 🤗 Hugging Face

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: Qwen3 7B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-7B")
provider = bridge.to_megatron_provider()

provider.tensor_model_parallel_size = 8
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Export Megatron → HF
```python
bridge.save_hf_pretrained(model, "./qwen-hf-export")
# or convert a checkpoint directory
bridge.export_ckpt(
    megatron_path="/results/qwen3_8b/checkpoints/iter_00002000",
    hf_path="./qwen-hf-export",
)
```

### Examples
- Checkpoint import/export: `examples/conversion/convert_checkpoints.py`
- Generate text (HF→Megatron): `examples/conversion/hf_to_megatron_generate_text.py`

## Pretrain recipes
- Example usage (Qwen3 8B)
```python
from megatron.bridge.recipes.qwen import qwen3_8b_pretrain_config

cfg = qwen3_8b_pretrain_config(
    hf_path="Qwen/Qwen3-8B",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/qwen3_8b",
)
```

- API reference for Qwen recipes:
  - Qwen recipes overview: [bridge.recipes.qwen](../../apidocs/bridge/bridge.recipes.qwen.md)
  - Qwen2 recipes: [bridge.recipes.qwen.qwen2](../../apidocs/bridge/bridge.recipes.qwen.qwen2.md)
  - Qwen3 recipes: [bridge.recipes.qwen.qwen3](../../apidocs/bridge/bridge.recipes.qwen.qwen3.md)
  - Qwen3 MoE recipes: [bridge.recipes.qwen.qwen3_moe](../../apidocs/bridge/bridge.recipes.qwen.qwen3_moe.md)

## Finetuning recipes
- Coming soon

## Hugging Face model cards
- Qwen2: `https://huggingface.co/Qwen/Qwen2-7B`
- Qwen2.5: `https://huggingface.co/Qwen/Qwen2.5-7B`
- Qwen3: `https://huggingface.co/Qwen/Qwen3-7B`

## Related docs
- Recipe usage and customization: [Recipe usage](../../recipe-usage.md)
- Training configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
