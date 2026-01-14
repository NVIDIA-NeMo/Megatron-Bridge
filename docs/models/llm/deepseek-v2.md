# DeepSeek V2

[DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) is a Mixture-of-Experts (MoE) language model that uses innovative Multi-head Latent Attention (MLA) for efficient inference and DeepSeekMoE architecture for economical training and inference. The model achieves performance comparable to GPT-4 while using significantly fewer activated parameters. More information is available in the companion paper ["DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"](https://arxiv.org/abs/2405.04434).

DeepSeek V2 models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

Megatron Bridge supports the following DeepSeek V2 model variants:

- **DeepSeek-V2**: 236B parameters (21B activated per token)
- **DeepSeek-V2-Lite**: 16B parameters (2.4B activated per token)

Both models support pretraining with expert parallelism for efficient MoE training.

## Model Architecture Features

- **Multi-head Latent Attention (MLA)**: Novel attention mechanism that reduces KV cache requirements
- **DeepSeekMoE**: Efficient MoE architecture with routed and shared experts
- **Expert Parallelism**: Distributes experts across GPUs for scalable training
- **RoPE Embeddings**: Rotary position embeddings for position encoding
- **128K Context Length**: Native support for long sequences (DeepSeek-V2)
- **Pre-normalization**: RMSNorm before each transformer sub-layer

## Conversion with 🤗 Hugging Face

### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: DeepSeek-V2-Lite
bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
provider = bridge.to_megatron_provider()

# Optionally configure parallelism before instantiating the model
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = 8

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Import Checkpoint from HF

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model deepseek-ai/DeepSeek-V2-Lite \
  --megatron-path /checkpoints/deepseek_v2_lite_megatron \
  --trust-remote-code
```

### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

# Load the bridge from HF model ID
bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)

# Export a trained Megatron checkpoint to HF format
bridge.export_ckpt(
    megatron_path="/results/deepseek_v2_lite/checkpoints/iter_0000500",
    hf_path="/exports/deepseek_v2_lite_hf",
)
```

### Run Inference on Converted Checkpoint

```bash
python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path deepseek-ai/DeepSeek-V2-Lite \
  --megatron_model_path /checkpoints/deepseek_v2_lite_megatron \
  --prompt "What is artificial intelligence?" \
  --max_new_tokens 100 \
  --ep 8 \
  --trust-remote-code
```

For more details, see [examples/conversion/hf_to_megatron_generate_text.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)

## Recipes

See: [bridge.recipes.deepseek.deepseek_v2](../../apidocs/bridge/bridge.recipes.deepseek.deepseek_v2.md)

### Available Recipes

- **Pretrain recipes**:
  - `deepseek_v2_lite_pretrain_config`: Pre-training for DeepSeek-V2-Lite (16B parameters, 2.4B activated per token)
  - `deepseek_v2_pretrain_config`: Pre-training for DeepSeek-V2 (236B parameters, 21B activated per token)

### Parallelism Configurations

| Model | TP | PP | EP | Total GPUs | Use Case |
|-------|----|----|----|-----------:|----------|
| **DeepSeek-V2-Lite** | 1 | 1 | 8 | 8 | Pre-training (single node) |
| **DeepSeek-V2** | 1 | 4 | 32 | 128 | Pre-training (16 nodes) |

**Key Features**:
- **Expert Parallelism**: EP=8 (V2-Lite) or EP=32 (V2) for efficient MoE training
- **Selective Recomputation**: Enabled by default for memory optimization
- **Sequence Length**: Default 4096, V2 supports up to 128K tokens

### Pre-training Examples

#### DeepSeek-V2-Lite (16B)

```python
from megatron.bridge.recipes.deepseek import deepseek_v2_lite_pretrain_config

config = deepseek_v2_lite_pretrain_config(
    name="deepseek_v2_lite_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/deepseek_v2_lite",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=4096,
    # Uses TP=1, PP=1, EP=8 (8 GPUs) automatically
)
```

#### DeepSeek-V2 (236B)

```python
from megatron.bridge.recipes.deepseek import deepseek_v2_pretrain_config

config = deepseek_v2_pretrain_config(
    name="deepseek_v2_pretrain",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/deepseek_v2",
    train_iters=500_000,
    global_batch_size=512,
    seq_length=4096,
    # Uses TP=1, PP=4, EP=32 (128 GPUs) automatically
)
```

### Finetuning Recipes

Finetuning recipes for DeepSeek V2 models are not currently available.


## Hugging Face Model Cards & References

### Hugging Face Model Cards
- DeepSeek-V2: https://huggingface.co/deepseek-ai/DeepSeek-V2
- DeepSeek-V2-Lite: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
- DeepSeek-V2-Chat: https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat
- DeepSeek-V2-Lite-Chat: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat

### Technical Papers
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

### Additional Resources
- GitHub Repository: https://github.com/deepseek-ai/DeepSeek-V2

## Related Docs
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)

