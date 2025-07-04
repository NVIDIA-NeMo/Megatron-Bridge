<div align="center">

# Megatron Hub

<!-- [![codecov](https://codecov.io/github/NVIDIA-NeMo/Megatron-Hub/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Megatron-Hub) -->
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Megatron-Hub/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Megatron-Hub/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
<!-- [![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Megatron-Hub.svg?style=social&label=Star&maxAge=2592000)](https://github.com/NVIDIA-NeMo/Megatron-Hub/stargazers/) -->

[Documentation](https://nemo-framework-documentation.gitlab-master-pages.nvidia.com/megatron-hub-build/) | [Recipes](#supported-models) | [Examples](https://github.com/NVIDIA-NeMo/Megatron-Hub/tree/maanug/readme-content/examples) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/CONTRIBUTING.md)
</div>

## Overview

Megatron Hub is an extension of NVIDIA's Megatron Core library that enables pretraining and finetuning HuggingFace models using a Megatron-style training loop, with features like model parallelisms and FP8 precision.

Megatron Hub is designed for researchers and engineers who need to train large-scale models efficiently while maintaining flexibility for experimentation and customization.

## Key Features

- **Model Conversion**: Seamless bidirectional conversion between Hugging Face and Megatron formats for interoperability
- **Training Infrastructure**: Configurable and scalable training loop that handles data loading, distributed training, checkpointing, and logging
- **Parameter-Efficient Finetuning**: In-house PEFT implementation that supports LoRA, DoRA, or custom PEFT methods
- **Training Recipes**: Pre-configured production-ready training recipes for popular models like Llama3, with optimized hyperparameters and distributed training configuration
- **Performance Optimization**: Built-in support for FP8 training, model parallelisms, and memory-efficient techniques

## Supported Models

Megatron Hub provides out-of-the-box configuration recipes for a wide range of models, built on top of base model architectures from Megatron Core:

### Large Language Models

| Model                  | Style     | Sizes     | Pretrain     | Finetune     |
|------------------------|-----------|-----------|--------------|--------------|
| Llama 3                | [GPT](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/src/megatron/hub/models/gpt_provider.py)       | [8b](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/src/megatron/hub/recipes/llama/llama3_8b.py), [70b](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/src/megatron/hub/recipes/llama/llama3_70b.py)   | ✅ | ❌ |
| Llama 3.1              | [GPT](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/src/megatron/hub/models/gpt_provider.py)       | 8b, 70b, 405b    | ❌ | ❌ |
| Llama 3.2              | [GPT](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/src/megatron/hub/models/gpt_provider.py)       | 1b, 3b           | ❌ | ❌ |

#### Launching Recipes

All recipes are ready to train out of the box, using mock data by default. For an example of how to override the default configuration through YAML or Hydra-style CLI overrides, please have a look at this [script](https://github.com/NVIDIA-NeMo/Megatron-Hub/examples/recipes/llama3_8b/pretrain_llama3_8b.py). The script can then be launched with `torchrun`. For example, with the aforementioned script:
```sh
torchrun --nproc-per-node=2 pretrain_llama3_8b.py model.tensor_model_parallel_size=1 <additional overrides ...>
```

Optionally, Megatron Hub also supports launching with [NeMo-Run](https://github.com/NVIDIA-NeMo/Run). See the following examples for reference on launching with NeMo-Run:

- [pretrain_llama3_8b_nemo_run_script.py](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/examples/recipes/llama3_8b/pretrain_llama3_8b_nemo_run_script.py)
- [pretrain_llama3_8b_nemo_run_partial.py](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/examples/recipes/llama3_8b/pretrain_llama3_8b_nemo_run_partial.py)


These examples can also be run as is with the Llama3 8b recipe (with NeMo-Run installed).

Launch Llama3 8b Pretraining with NeMo-Run's `run.Script`:
```sh
uv run python pretrain_llama3_8b_nemo_run_script.py \
    --nproc-per-node=2 \
    model.pipeline_model_parallel_size=1 \
    train.train_iters=10 # this script passes Hydra-style overrides to the target script
```

Launch Llama3 8b Pretraining with NeMo-Run's `run.Partial`
```sh
uv run python pretrain_llama3_8b_nemo_run_partial.py \
    --nproc-per-node=2
```

<!-- ### Vision-Language Models -->


## Performance Benchmarks

Coming soon ...

## Installation 

### Pip Installation
To install with pip, use the following command:
```
pip install git+https://github.com/NVIDIA-NeMo/Megatron-Hub.git
```


### uv Installation
To install Megatron Hub to an active virtual environment or project environment, use the command:
```
uv pip install git+https://github.com/NVIDIA-NeMo/Megatron-Hub.git
```

To add Megatron Hub as a dependency for your project, use the following command:
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

## Project Structure

```
Megatron-Hub/
├── examples/
│   ├── bridge/                  # Bridge usage examples
│   └── recipes/                 # Training examples
├── src/megatron/hub/
│   ├── bridge/                  # HuggingFace bridge infrastructure
│   ├── common/                  # Shared utilities and mixins
│   ├── core/
│   │   ├── models/              # Wrapped MCore model builders
│   │   │   └── model_provider.py
│   │   └── utils/               # Utilities intended for mcore
│   ├── data/                    # Dataloaders and iterators
│   ├── models/                  # Model-specific implementations
│   │   ├── llama/               # Llama model providers
│   │   └── .../                 # Other models (gpt, t5, etc.)
│   ├── peft/                    # PEFT transformations and wrappers
│   ├── recipes/                 # Complete training recipes
│   └── training/                # Training loop components
│       ├── tokenizers/          # Tokenizer library
│       └── utils/               # Training-specific utilities
└── tests/                       # Comprehensive test suite
```

## Contributing

We welcome community contributions! Please see our [Contributor Guidelines](https://github.com/NVIDIA-NeMo/Megatron-Hub/blob/main/CONTRIBUTING.md) for more information on how to get involved.
