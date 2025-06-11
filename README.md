# Megatron Hub: Training Recipes for Megatron-based LLM and VLM models

More content coming soon ...

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
cd NeMo
uv sync
```

To install additional dependency groups use one of the following commands instead:
```
uv sync --group docs # for building the documentation
uv sync --group dev --group test # for running linters and tests
```

If you do not have `uv` installed, please refer to the installation [docs](https://docs.astral.sh/uv/getting-started/installation/).
