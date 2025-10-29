# CPU Offloading

## Overview

CPU Offloading in Megatron Bridge is a feature that reduces the peak memory usage of the GPU by offloading activations and inactive weights to CPU storage. Megatron Bridge supports offloading at the transformer layer level, allowing users to specify the number of transformer layers in their language model that require CPU offloading. During the forward pass, Megatron Bridge offloads activations at the optimal time and reloads them as needed during the backward pass.

## Features

- Supports training models with long sequence lengths by managing activation memory efficiently
- Enables high batch sizes per GPU by offloading activation memory
- Overlaps computation with data transfers (Host2Device and Device2Host) during offloading and reloading

## Configuration

CPU offloading is configured through the model provider parameters:

```python
from megatron.bridge.models import GPTModelProvider

# Basic CPU offloading configuration
model_config = GPTModelProvider(
    # Model architecture
    hidden_size=4096,
    num_layers=32,
    
    # CPU offloading settings
    cpu_offloading=True,              # Enable CPU offloading
    cpu_offloading_num_layers=16,     # Number of layers to offload (0 to num_layers-1)
    cpu_offloading_activations=True,  # Offload activations
    cpu_offloading_weights=True,      # Offload weights
    
    # ... other model parameters
)
```

### Configuration Parameters

- **`cpu_offloading`**: Set to `True` to enable CPU offloading
- **`cpu_offloading_num_layers`**: Number of transformer layers to offload (value between 0 and total number of layers minus one)
- **`cpu_offloading_activations`**: Whether to offload activations to CPU memory (default: `True`)
- **`cpu_offloading_weights`**: Whether to offload inactive weights to CPU memory (default: `False`)
- **`cpu_offloading_double_buffering`**: Enable double buffering across layers while reloading activations from CPU (default: `False`)

### Offloading Strategies

You can configure different combinations of offloading based on your memory requirements:

#### Activations Only
```python
model_config = GPTModelProvider(
    cpu_offloading=True,
    cpu_offloading_num_layers=8,
    cpu_offloading_activations=True,   # Offload activations
    cpu_offloading_weights=False,      # Keep weights on GPU
)
```

#### Weights Only
```python
model_config = GPTModelProvider(
    cpu_offloading=True,
    cpu_offloading_num_layers=8,
    cpu_offloading_activations=False,  # Keep activations on GPU
    cpu_offloading_weights=True,       # Offload weights
)
```

#### Both Activations and Weights
```python
model_config = GPTModelProvider(
    cpu_offloading=True,
    cpu_offloading_num_layers=8,
    cpu_offloading_activations=True,   # Offload activations
    cpu_offloading_weights=True,       # Offload weights
)
```
