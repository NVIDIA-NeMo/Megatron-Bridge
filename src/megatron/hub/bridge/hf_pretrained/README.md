# HuggingFace pre-trained checkpoints

Structured, type-safe classes for working with Hugging Face checkpoints. Each checkpoint type has its own structure - `megatron.hub._lib.hf` makes these contracts explicit and provides a clean interface.

## Quick Start

```python
from megatron.hub._lib.hf import PreTrainedCausalLM

# Load any Hugging Face model with proper structure
model = PreTrainedCausalLM.from_pretrained("gpt2")

# See exactly what the checkpoint contains
print(model)
# PreTrainedCausalLM(
#   (model): GPT2LMHeadModel [layers=12, hidden_size=768]
#   (tokenizer): GPT2TokenizerFast [vocab_size=50257]
#   (config): GPT2Config [model_type=gpt2]
#   (generation_config): GenerationConfig [loaded]
#   (parameters): 124,439,808
#   (device): cpu
#   (dtype): torch.float32
# )

# Use it naturally
text = model.encode("Hello world")
output = model.generate(text.input_ids, max_length=50)
result = model.decode(output[0])
```

## Available Classes

### PreTrainedCausalLM
For text generation models (GPT, LLaMA, etc.)

```python
from megatron.hub._lib.hf import PreTrainedCausalLM

# Type-safe loading with lazy evaluation
llama = PreTrainedCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device="cuda"
)

# Components load on demand
config = llama.config        # Loads just config
tokenizer = llama.tokenizer  # Loads just tokenizer  
model = llama.model          # Loads model weights
```

### PreTrainedVLM
For vision-language models (CLIP, LLaVA, etc.)

```python
from megatron.hub._lib.hf import PreTrainedVLM

vlm = PreTrainedVLM.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Unified processing for images and text
inputs = vlm.process_images_and_text(
    images=my_image,
    text="What's in this image?"
)

output = vlm.generate(**inputs)
```

## Key Features

### 🔍 Transparent Inspection
See exactly what's in a checkpoint without loading everything:

```python
model = PreTrainedCausalLM.from_pretrained("microsoft/phi-2")
print(model)  # Shows architecture, parameters, device, dtype
```

### 💾 Lazy Loading
Components load only when accessed, saving memory:

```python
# Nothing loaded yet
model = PreTrainedCausalLM.from_pretrained("gpt2")

# Still nothing loaded - just returns the config
config = model.config  

# Now the model weights are loaded
outputs = model.generate(...)
```

### 🎯 Type Safety
Full type hints for better IDE support:

```python
from transformers import GPT2LMHeadModel

gpt2: PreTrainedCausalLM[GPT2LMHeadModel] = PreTrainedCausalLM.from_pretrained("gpt2")
# IDE knows exact model type for autocomplete
```

### 🔧 Unified State Dict Access
Access model weights consistently:

```python
# Works for any model type
model.state["*.attention.*.weight"]  # Get attention weights
model.state.regex(r".*\.bias$")      # Find all biases
model.state.glob("*.layer.*.weight") # Pattern matching
```
