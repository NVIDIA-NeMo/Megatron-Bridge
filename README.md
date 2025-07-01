# Megatron-Hub

## Overview

Megatron-Hub is a comprehensive training framework built on top of NVIDIA's Megatron-Core library for pretraining and finetuning Megatron-based large language models (LLMs) and vision-language models (VLMs). It provides a highly scalable and configurable training loop, optimized production-ready training recipes, and bidirectional conversion between HuggingFace and Megatron formats.

Megatron-Hub is designed for researchers and engineers who need to train large-scale models efficiently while maintaining flexibility for experimentation and customization.

## Key Features

- **Training Infrastructure**: Complete training loop that handles data loading, distributed training, checkpointing, and logging
- **Parameter-Efficient Finetuning**: In-house PEFT implementation that supports LoRA, DoRA, or custom PEFT methods
- **Training Recipes**: Pre-configured training recipes for popular models like Llama3, with optimized hyperparameters and distributed training configuration
- **Model Conversion**: Seamless conversion between Hugging Face and Megatron formats for interoperability
- **Performance Optimization**: Built-in support for FP8 training, model parallelisms, and memory-efficient techniques

## Installation 

### Pip Installation
To install with pip, use the following command:
```
pip install git+https://github.com/NVIDIA-NeMo/Megatron-Hub.git
```


### uv Installation
To install Megatron-Hub to an active virtual environment or project environment, use the command:
```
uv pip install git+https://github.com/NVIDIA-NeMo/Megatron-Hub.git
```

To add Megatron-Hub as a dependency for your project, use the following command:
```
uv add git+https://github.com/NVIDIA-NeMo/Megatron-Hub.git
```

If you are a contributor, you can install this project for development with the following commands:
```
git clone https://github.com/NVIDIA-NeMo/Megatron-Hub.git
cd Megatron-Hub
uv sync
```

To install additional dependency groups use one of the following commands instead:
```
uv sync --group docs # for building the documentation
uv sync --group dev --group test # for running linters and tests
```

If you do not have `uv` installed, please refer to the installation [docs](https://docs.astral.sh/uv/getting-started/installation/).
