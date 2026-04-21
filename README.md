<div align="center">

# NeMo Megatron Bridge

[![codecov](https://codecov.io/github/NVIDIA-NeMo/Megatron-Bridge/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Megatron-Bridge)
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Megatron-Bridge/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Megatron-Bridge.svg?style=social&label=Star&cacheSeconds=14400)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/stargazers/)

[Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) | [Supported Models](#supported-models) | [Examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md)
</div>

## 📣 News

- [04/16/2026] **Megatron Bridge 0.4.0 released!** New model support (Kimi 2.5, Nemotron 3 Super, Qwen 3.5 VL, MiniMax M2, Sarvam, MiMo, and more), diffusion model collection, sequence-packing improvements, FP8 export, pruning & quantization, Transformers 5.x compatibility, and Python 3.12 migration. Huge thanks to our community contributors: [@HollowMan6](https://github.com/HollowMan6), [@shaltielshmid](https://github.com/shaltielshmid), [@jaeminh](https://github.com/jaeminh), [@pavelgein](https://github.com/pavelgein), [@ShiftyBlock](https://github.com/ShiftyBlock), [@erictang000](https://github.com/erictang000), [@eternally-z](https://github.com/eternally-z), [@Hayak3](https://github.com/Hayak3), and [@mohit-sarvam](https://github.com/mohit-sarvam)! See the [full release notes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/tag/v0.4.0).

- [04/01/2026] [**Kimi K2.5 VL**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/vlm/kimi_k25_vl) is now supported! Checkpoint conversion, inference, and training recipes for [Moonshot AI’s Kimi-K2.5-VL](https://huggingface.co/moonshotai/Kimi-K2.5) vision-language model are available on **main**.

- [03/31/2026] **Agent Skills for Megatron Bridge!** We've added a [`skills/`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills) directory with structured guides that AI coding agents (Cursor, Claude Code, Codex, etc.) can use to help you add model support, set up dev environments, tune performance, and more. Try them out, and PRs to improve or add new skills are very welcome!

- [03/26/2026] [**Nemotron 3 Super**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/nemotron_3) is now on **main**! Checkpoint conversion and SFT/LoRA recipes (120B-A12B) are available in the main branch. Read the [blog post](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/).

- [03/12/2026] **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.4.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with Megatron-Bridge.

- [12/16/2025] [Mind Lab](https://macaron.im/mindlab) successfully used Megatron-bridge and [VeRL](https://github.com/volcengine/verl) to trained GRPO Lora for Trillion-parameter model on 64 H800 - See their [techblog](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus).

- [12/15/2025] Day 0 support for [NVIDIA-NeMotron-3-Nano-30B-A3B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)! [Reproducible code](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/nemotron_3/nano) and custom NGC container: [nvcr.io/nvidia/nemo:25.11.nemotron_3_nano](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=25.11.nemotron_3_nano)

## Overview

NeMo Megatron Bridge is a PyTorch-native library within the [NeMo Framework](https://github.com/NVIDIA-NeMo) that provides pretraining, SFT and LoRA for popular LLM and VLM models. It serves as a powerful **bridge, conversion, and verification layer** between 🤗 Hugging Face and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). It provides bidirectional checkpoint conversion between these formats, enabling other projects to leverage Megatron Core's parallelism capabilities or export models for various inference engines. The bridge includes built-in verification mechanisms to ensure conversion accuracy and checkpoint integrity across different model formats.

On top of the bridge, NeMo Megatron Bridge provides a performant and scalable PyTorch-native training loop that leverages [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) to deliver state-of-the-art training throughput. It supports pretraining and fine-tuning with features like tensor and pipeline parallelism, and mixed precision (FP8, BF16, FP4, etc.). Users can either use existing 🤗 Hugging Face models or define custom PyTorch model definitions for flexible end-to-end workflows.

NeMo Megatron Bridge is a refactor of the [previous NeMo](https://github.com/NVIDIA/NeMo) training stack that adopts a PyTorch-native training loop to provide greater flexibility and customizability for developers.

![image](Repo-Mbridge.png)

## 🔧 Installation

### 🐳 NeMo Framework container

The best experience, highest performance, and full feature support are provided by the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). Fetch the most recent $TAG and run the following to start a container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}
```

For development installation and additional details, please refer to our [Contribution guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md).

### Megatron-Core Submodule (main & dev)

Megatron Bridge pins [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) as a git submodule at `3rdparty/Megatron-LM`. The repository tracks two pinned commits — one from the upstream **main** branch (default) and one from **dev** — managed by `scripts/switch_mcore.sh`.

The submodule committed to the repo always points to the **main** commit. Use the **dev** commit when you need a Megatron-Core feature or fix that has not yet landed on main, or to validate forward-compatibility with upcoming MCore changes:

```bash
./scripts/switch_mcore.sh status   # Show current commit
./scripts/switch_mcore.sh dev      # Switch to dev; then run: uv sync
./scripts/switch_mcore.sh main     # Switch back; then run: uv sync --locked
```

> **Note:** `uv.lock` is generated against the main commit. After switching to dev, use `uv sync` (without `--locked`). After switching back to main, use `uv sync --locked`.

The dev branch follows Megatron-LM's upstream [dev branch philosophy](https://github.com/NVIDIA/Megatron-LM/tree/dev) — features are experimental, follow a streamlined review process, and must graduate to stable within 6 months or be deprecated.

## ⚡ Quickstart

To get started, install Megatron Bridge or download a NeMo Framework container as described [above](#-installation).

Log in to Hugging Face Hub:

```sh
huggingface-cli login --token <your token>
```

Conversion-only quickstart (✅ Core):

```python
from megatron.bridge import AutoBridge

# 1) Create a bridge from a Hugging Face model (hub or local path)
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)

# 2) Get a Megatron provider and configure parallelism before instantiation
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
# 3) Materialize Megatron Core model(s)
model = provider.provide_distributed_model(wrap_with_ddp=False)

# 4a) Export Megatron → Hugging Face (full HF folder with config/tokenizer/weights)
bridge.save_hf_pretrained(model, "./hf_exports/llama32_1b")

# 4b) Or stream only weights (Megatron → HF)
for name, weight in bridge.export_hf_weights(model, cpu=True):
    print(name, tuple(weight.shape))
```

Training quickstart using pre-configured recipes:

```python
from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

if __name__ == "__main__":
    # The recipe uses the Llama 3.2 1B model configuration from HuggingFace
    cfg = llama32_1b_pretrain_config()

    # Override training parameters
    cfg.train.train_iters = 10
    cfg.scheduler.lr_decay_iters = 10000
    cfg.model.vocab_size = 8192
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    pretrain(cfg, forward_step)
```

You can launch the above script with:

```sh
torchrun --nproc-per-node=<num devices> /path/to/script.py
```

More examples:

- [Conversion scripts overview](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/README.md)
- [Import/Export checkpoints](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/convert_checkpoints.py)
- [Generation with bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_to_megatron_generate_text.py)
- [Multi-GPU loading from HF](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/hf_megatron_roundtrip_multi_gpu.py)
- [Compare HF vs Megatron outputs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/compare_models.py)
- [Toy RLHF with Bridge (HF inference + Megatron training)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/rl/rlhf_with_bridge.py)

For a deeper dive into conversion design and advanced usage, see the [models README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/README.md).

## 🚀 Key Features

- **Bridge with 🤗 Hugging Face**: Seamless bidirectional conversion between 🤗 Hugging Face and Megatron formats for interoperability ([model bridges](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models), [auto bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/auto_bridge.py), [conversion examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/conversion))
  - Online import/export without intermediate full checkpoints
  - Parallelism-aware (TP/PP/VPP/CP/EP/ETP) during conversion
  - Memory-efficient per-parameter streaming
  - Simple high-level `AutoBridge` API with architecture auto-detection
  - Optimized paths when Transformer Engine is available
- **Flexible to Customize**: Lightweight custom training loop making it easy to configure custom logic in data loading, distributed training, checkpointing, evaluation and logging ([training framework](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/training), [training utilities](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/training/utils))
- **Supervised & Parameter-Efficient Finetuning**: SFT & PEFT implementation tailored for Megatron-based models that supports LoRA, DoRA, and user-defined PEFT methods ([PEFT implementations](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/peft), [finetune module](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/finetune.py), [SFT dataset](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/data/datasets/sft.py))
- **SOTA Training Recipes**: Pre-configured production-ready training recipes for popular models like Llama 3, with optimized hyperparameters and distributed training configuration ([Llama recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/recipes/llama), [recipe examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models))
- **Performance Optimization**: Built-in support for FP8 training, model parallelism, and memory-efficient techniques to offer high utilization and near-linear scalability to thousands of nodes. ([mixed precision](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/mixed_precision.py), [communication overlap](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/comm_overlap.py), [optimizer utilities](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/utils/optimizer_utils.py))

## Supported Models

Megatron Bridge provides out-of-the-box bridges and training recipes for a wide range of models, built on top of base model architectures from [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). Refer to the [models directory](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models) for the full list of model bridges.

### Nemotron (NVIDIA)

- [Nemotron Nano v2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/nemotronh) — [recipes (9B/12B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_nano_v2.py)
- [Nemotron Nano v3](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/nemotronh) — [recipes (30B-A3B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py)
- [Nemotron Super v3](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/nemotronh) — [recipes (120B-A12B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_3_super.py)
- [Llama Nemotron](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/llama_nemotron)
- [Nemotron Nano v2 VL](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/nemotron_vl) — [recipes (9B/12B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotron_vl/nemotron_nano_v2_vl.py)

### Large Language Models (LLM)

- [DeepSeek V2 / V2 Lite](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/deepseek) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/deepseek/deepseek_v2.py)
- [DeepSeek V3](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/deepseek) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/deepseek/deepseek_v3.py)
- [Gemma / Gemma 2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/gemma) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/gemma/gemma2.py)
- [Gemma 3](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/gemma) — [recipes (1B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/gemma/gemma3.py)
- [GLM-4.5](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/glm) — [recipes (106B-Air/355B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/glm/glm45.py)
- [GPT-oss](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/gpt_oss) — [recipes (20B/120B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/gpt_oss/gpt_oss.py)
- [Kimi K2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/kimi) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/kimi/kimi_k2.py)
- [Llama 2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/llama) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/llama/llama2.py)
- [Llama 3 / 3.1 / 3.2 / 3.3](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/llama) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/llama/llama3.py)
- [Mamba](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/mamba)
- [Ministral](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/ministral3) — [recipes (3B/8B/14B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/ministral3/ministral3.py)
- [Mistral](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/mistral)
- [MiniMax-M2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/minimax_m2)
- [Moonlight](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/deepseek) — [recipes (16B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/moonlight/moonlight_16b.py)
- [OlMoE](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/olmoe) — [recipes (7B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/olmoe/olmoe_7b.py)
- [Qwen2 / Qwen2.5](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen/qwen2.py)
- [Qwen3](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen/qwen3.py)
- [Qwen3-MoE](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen/qwen3_moe.py)
- [Qwen3 Next](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen/qwen3_next.py)
- [Sarvam](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/sarvam)

### Vision Language Models (VLM)

- [Gemma 3-VL](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/gemma_vl) — [recipes (4B/12B/27B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/gemma3_vl/gemma3_vl.py)
- [GLM-4.5V](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/glm_vl) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/glm_vl/glm_45v.py)
- [MiMo](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/mimo)
- [Qwen2.5-VL](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen_vl) — [recipes (3B/7B/32B/72B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen_vl/qwen25_vl.py)
- [Qwen3-VL](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen_vl) — [recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen_vl/qwen3_vl.py)
- [Qwen3.5-VL](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen_vl) — [recipes (800M/2B/4B/9B/27B/35B-A3B/122B-A10B/397B-A17B)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py)

### Omni Models

- [Qwen2 Audio](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen_audio)
- [Qwen2.5-Omni](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen_omni)

#### Launching Recipes

For a conceptual overview of how recipes are structured, overridden, and launched with either `torchrun` or NeMo-Run, read the [Using Recipes guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html).

Runnable tutorials live in `tutorials/recipes/llama` that covers:

- `00_quickstart_pretrain.py` for mock-data pretraining
- `01_quickstart_finetune.py` + LoRA configs
- YAML-driven flows and launch helpers

## Performance Benchmarks

For detailed performance benchmarks including throughput metrics across different GPU systems (DGX-GB200, DGX-B200, DGX-H100) and model configurations, see the [Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) in our documentation.

## Project Structure

```
Megatron-Bridge/
├── examples/
│   ├── models/                  # Bridge usage examples
│   └── recipes/                 # Training examples
├── src/megatron/bridge/
│   ├── data/                    # Dataloaders and iterators
│   ├── models/                  # Hugging Face bridge infrastructure and model-specific implementations
│   │   ├── llama/               # Llama model providers
│   │   └── .../                 # Other models (gpt, t5, etc.)
│   ├── peft/                    # PEFT transformations and wrappers
│   ├── recipes/                 # Complete training recipes
│   ├── training/                # Training loop components
│   │   ├── tokenizers/          # Tokenizer library
│   │   └── utils/               # Training-specific utilities
│   └── utils/                   # Generic utilities for repo-wide usage
└── tests/                       # Comprehensive test suite
```

## Acknowledgement & Contributing

Megatron-Bridge is the continuation of [MBridge](https://github.com/ISEEKYAN/mbridge) by [Yan Bai](https://github.com/ISEEKYAN). We appreciate all the contribution and adoptions by the community partners:

- [Mind Lab](https://macaron.im/mindlab) successfully used Megatron-bridge and [VeRL](https://github.com/volcengine/verl) to trained GRPO Lora for Trillion-parameter model on 64 H800 - See their [techblog](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus).
- [VeRL](https://github.com/volcengine/verl) has adopted Megatron-Bridge as a connector to Megatron-Core and for LoRA support.
- [Slime](https://github.com/THUDM/slime) has adopted Megatron-Bridge as Megatron-Core checkpoint converter.
- [SkyRL](https://github.com/NovaSky-AI/SkyRL) has adopted Megatron-Bridge as Megatron-Core connector.
- [Nemo-RL](https://github.com/NVIDIA/nemo-rl) has adopted Megatron-Bridge as Megatron-Core connector.
- Community contributions: Special thanks to [Guanyou He](https://github.com/Thaurun) and [Junyu Wu](https://github.com/nrailg) from Weixin Group Infrastructure Center.

Please see our [Contributor Guidelines](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) for more information on how to get involved.
