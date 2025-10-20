# Qwen3

Qwen3 models are supported via the Bridge, including QK layernorm and MoE variants (A3B, A22B).

## Conversion with 🤗 Hugging Face

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: Qwen3 7B
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-7B")
provider = bridge.to_megatron_provider()

provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 1
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Export Megatron → HF
```python
# Convert from a Megatron checkpoint directory to HF format
bridge.export_ckpt(
    megatron_path="/results/qwen3_7b/checkpoints/iter_00002000",
    hf_path="./qwen3-hf-export",
)
```

## Examples
- Checkpoint import/export: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- Generate text (HF→Megatron): [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

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

- API reference for Qwen3 recipes:
  - Qwen recipes overview: [bridge.recipes.qwen](../../apidocs/bridge/bridge.recipes.qwen.md)
  - Qwen3 recipes: [bridge.recipes.qwen.qwen3](../../apidocs/bridge/bridge.recipes.qwen.qwen3.md)
  - Qwen3 MoE recipes: [bridge.recipes.qwen.qwen3_moe](../../apidocs/bridge/bridge.recipes.qwen.qwen3_moe.md)

## Finetuning recipes
- Coming soon

## Hugging Face model cards
- Qwen3: `https://huggingface.co/Qwen/Qwen3-7B`

## Related docs
- Recipe usage and customization: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

