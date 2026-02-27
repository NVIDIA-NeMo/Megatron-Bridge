# Llama Nemotron

[Llama Nemotron](https://huggingface.co/collections/nvidia/llama-nemotron) is NVIDIA's family of large language models derived from Meta's Llama architecture, post-trained for enhanced reasoning, human chat preferences, and agentic tasks such as RAG and tool calling. The models feature neural architecture search (NAS) optimizations for improved efficiency and accuracy trade-offs.

Llama Nemotron models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

Megatron Bridge supports the following Llama Nemotron model variants:

- **Llama-3.3-Nemotron-Super-49B**: 49B parameters (NAS-optimized from 70B)
- **Llama-3.1-Nemotron-Ultra-253B**: 253B parameters (large-scale reasoning model)
- **Llama-3.1-Nemotron-70B**: 70B parameters (standard size)
- **Llama-3.1-Nemotron-Nano-8B**: 8B parameters (efficient variant)
- **Llama-3.1-Nemotron-Nano-4B**: 4B parameters (ultra-compact variant)

All models are ready for commercial use and support context lengths up to 128K tokens.

## Model Architecture Features

- **Neural Architecture Search (NAS)**: Novel approach to reduce memory footprint while maintaining accuracy
- **Heterogeneous Blocks**: Non-standard and non-repetitive layer configurations for efficiency
  - Skip attention in some blocks
  - Variable FFN expansion/compression ratios between blocks
- **Multi-Phase Post-Training**:
  - Supervised fine-tuning for Math, Code, Science, and Tool Calling
  - Reward-aware Preference Optimization (RPO) for chat
  - Reinforcement Learning with Verifiable Rewards (RLVR) for reasoning
  - Iterative Direct Preference Optimization (DPO) for tool calling
- **Extended Context**: Native support for sequences up to 128K tokens
- **Commercial Ready**: Fully licensed for commercial deployment

## Conversion with 🤗 Hugging Face

### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: Llama-3.3-Nemotron-Super-49B
bridge = AutoBridge.from_hf_pretrained(
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    trust_remote_code=True
)
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 2
provider.pipeline_model_parallel_size = 1

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

**Note**: Heterogeneous Llama-Nemotron models (Super/Ultra) require `trust_remote_code=True` as they use custom `DeciLMForCausalLM` architecture. Homogeneous models (Nano/70B) use standard Llama architecture and don't require this flag.

### Import Checkpoint from HF

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --megatron-path /checkpoints/llama_nemotron_super_49b_megatron \
  --trust-remote-code
```

### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained(
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    trust_remote_code=True
)

# Export a trained Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/llama_nemotron_super_49b/checkpoints/iter_0000500",
    hf_path="/exports/llama_nemotron_super_49b_hf",
)
```

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --megatron_model_path /checkpoints/llama_nemotron_super_49b_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --tp 2 \
  --trust-remote-code
```

For more details, see [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Recipes

Training recipes for Llama Nemotron models are not currently available.

## Hugging Face Model Cards & References

### Hugging Face Model Cards
- Llama Nemotron Collection: https://huggingface.co/collections/nvidia/llama-nemotron
- Llama-3.3-Nemotron-Super-49B-v1.5: https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
- Llama-3.1-Nemotron-Ultra-253B-v1: https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
- Llama-3.1-Nemotron-Nano-8B-v1: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1
- Llama-3.1-Nemotron-Nano-4B-v1.1: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1

### Technical Papers
- Llama-Nemotron: Efficient Reasoning Models: [arXiv:2505.00949](https://arxiv.org/abs/2505.00949)
- Puzzle: Distillation-Based NAS for Inference-Optimized LLMs: [arXiv:2411.19146](https://arxiv.org/abs/2411.19146)
- Reward-aware Preference Optimization: [arXiv:2502.00203](https://arxiv.org/abs/2502.00203)

### Additional Resources
- NVIDIA Build Platform: https://build.nvidia.com/
- Llama Nemotron Post-Training Dataset: https://huggingface.co/nvidia/Llama-Nemotron-Post-Training-Dataset

## Related Docs
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

