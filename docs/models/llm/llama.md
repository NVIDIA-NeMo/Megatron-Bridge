# Llama

Llama family models are supported via the Bridge system with auto-detected configuration and weight mapping.

For background on Llama 3 model family, recipes, and recommended configurations in NeMo, see the official guide: [NeMo Llama 3](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama3.html).

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
# Option 1: Save directly from a live Megatron model instance
bridge.save_hf_pretrained(model, "./llama-hf-export")

# Option 2: Convert from a Megatron checkpoint directory to HF format
bridge.export_ckpt(megatron_path="./checkpoints/global_step123", hf_path="./llama-hf-export")
```

## Pretrain recipes
- Example usage
```python
from megatron.bridge.recipes.llama import llama3_8b_pretrain_config

cfg = llama3_8b_pretrain_config(
    hf_path="meta-llama/Meta-Llama-3-8B",
    data_paths=["/path/to/dataset.nvjsonl"],
    dir="/results/llama3_8b",
)
```

- API docs: see module `megatron.bridge.recipes.llama` and functions such as
  `llama3_8b_pretrain_config`, `llama3_70b_pretrain_config`, `llama31_8b_pretrain_config`, `llama2_7b_pretrain_config` in the API reference.

## Finetuning recipes
- Not yet available in this repository.

Reference: [NeMo Llama 3](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama3.html)
