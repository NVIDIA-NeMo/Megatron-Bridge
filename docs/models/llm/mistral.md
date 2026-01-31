# Mistral

[Mistral AI](https://mistral.ai/) develops frontier large language models with a focus on efficiency and performance. The Mistral family includes both dense and Mixture-of-Experts architectures, featuring innovations like sliding window attention and efficient context handling.

Mistral models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

Megatron Bridge supports the following Mistral model variants:

- **Mistral Small 3 (24B)**: 24B parameters with 128K context length
- **Mistral 7B**: 7B parameters, efficient baseline model
- **Mistral 7B Instruct**: Instruction-tuned variant

Additional Mistral models (including MoE variants like Mixtral) may be supported through the standard conversion pipeline.

## Model Architecture Features

- **Sliding Window Attention**: Efficient attention mechanism for long sequences
- **Grouped Query Attention (GQA)**: Memory-efficient attention mechanism
- **Rotary Positional Embeddings (RoPE)**: Relative position encoding
- **SwiGLU Activation**: Gated linear units in the feedforward network
- **Extended Context**: Support for sequences up to 128K tokens (Mistral Small 3)
- **YaRN RoPE Scaling**: Advanced rope scaling for extended context lengths

## Conversion with 🤗 Hugging Face

### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Mistral Small 3 24B
bridge = AutoBridge.from_hf_pretrained("mistralai/Mistral-Small-24B-Base-2501")
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Import Checkpoint from HF

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model mistralai/Mistral-Small-24B-Base-2501 \
  --megatron-path /checkpoints/mistral_small_24b_megatron
```

### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained("mistralai/Mistral-Small-24B-Base-2501")

# Export a trained Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/mistral_small_24b/checkpoints/iter_0000500",
    hf_path="/exports/mistral_small_24b_hf",
)
```

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path mistralai/Mistral-Small-24B-Base-2501 \
  --megatron_model_path /checkpoints/mistral_small_24b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2
```

For more details, see [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Recipes

Training recipes for Mistral models are not currently available. The Bridge supports checkpoint conversion for inference and deployment use cases.

## Hugging Face Model Cards & References

### Hugging Face Model Cards
- Mistral Small 3 (24B): https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501
- Mistral Small 3 (24B) Instruct: https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
- Mistral 7B v0.1: https://huggingface.co/mistralai/Mistral-7B-v0.1
- Mistral 7B Instruct v0.2: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

### Technical Papers
- Mistral 7B: https://arxiv.org/abs/2310.06825

### Additional Resources
- Mistral AI Website: https://mistral.ai/
- Mistral Documentation: https://docs.mistral.ai/

## Related Docs
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

