# Llama 3

[Meta’s Llama](https://ai.meta.com/blog/meta-llama-3/) builds on the general transformer decoder framework with some key additions such as pre-normalization, SwiGLU activations, and Rotary Positional Embeddings (RoPE). More information is available in the companion paper [“Llama: Open and Efficient Foundation Language Models”](https://arxiv.org/abs/2302.13971). With a wide variety of model sizes - Llama has options for every inference budget.

Llama family models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Conversion with 🤗 Hugging Face

### Load HF → Megatron
```python
from megatron.bridge import AutoBridge

# Example: Llama 3.1 8B
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.1-8B-Instruct")
provider = bridge.to_megatron_provider()

# Configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Export Megatron → HF
```python
# Convert from a Megatron checkpoint directory to HF format
bridge.export_ckpt(
    megatron_path="/results/llama3_8b/checkpoints/iter_00002000",
    hf_path="./llama-hf-export",
)
```

## Examples
- Checkpoint import/export: [examples/conversion/convert_checkpoints.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- Generate text (HF→Megatron): [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Pretrain recipes
- See: [bridge.recipes.llama.llama3](../../apidocs/bridge/bridge.recipes.llama.llama3.md)

## Related docs
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
