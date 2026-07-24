<div align="center">

# NeMo Megatron Bridge

[![codecov](https://codecov.io/github/NVIDIA-NeMo/Megatron-Bridge/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Megatron-Bridge)
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Megatron-Bridge/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Megatron-Bridge.svg?style=social&label=Star&cacheSeconds=14400)](https://github.com/NVIDIA-NeMo/Megatron-Bridge/stargazers/)

[Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) | [Supported Models](#supported-models) | [Examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples) | [Contributing](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md)
</div>

## 📣 News
- [07/17/2026] [**LongStraw (MinT-2M)**](https://github.com/MindLab-Research/longstraw) is a long-context RL research system from MindLab Research that uses Megatron Bridge's modeling and LoRA capabilities. LongStraw explores resident-prefix, response-only GRPO by capturing a shared prompt once and replaying only the trainable response branches, and reports a 2.1M-token GLM-5.2 execution path across 32 NVIDIA H20 GPUs. The broader MinT system behind this work was also used to post-train [**Macaron-V1-Preview**](https://macaron.im/mindlab/research/macaron-v1-preview), MindLab's 749B Mixture-of-LoRA agent model derived from GLM-5.1. Learn more in the [paper](https://arxiv.org/abs/2607.14952).

- [06/22/2026] **Megatron Bridge 0.5.0 released!** Highlights include expanded LLM and multimodal support (Qwen3.5, DeepSeek V4, Ernie 4.5, GLM-5/4.7, StepFun Step-3.5/3.7, MiMo-V2, Gemma 4, Falcon H1, Ling MoE V2, Nemotron-3 Nano Omni, Qwen3-Omni, Qwen3-ASR, and Nemotron Diffusion), MegatronMIMO and Energon v7 training updates, evaluator backend integration, eval-time context parallelism, deterministic recipes, quantized FP8/MXFP4 export, CUDA graph/performance improvements, and Megatron Inference/tokenizer unification with Megatron-LM. Huge thanks to our community contributors: [@HowardZorn](https://github.com/HowardZorn), [@hy2826](https://github.com/hy2826), [@bo-ke](https://github.com/bo-ke), [@beccohov](https://github.com/beccohov), [@dhiaEddineRhaiem](https://github.com/dhiaEddineRhaiem), [@pavelgein](https://github.com/pavelgein), [@ccclyu](https://github.com/ccclyu), [@hbhflw2000](https://github.com/hbhflw2000), and [@HollowMan6](https://github.com/HollowMan6)! See the [full release notes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/tag/v0.5.0).

- [06/16/2026] NVIDIA topped [MLPerf Training v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/) across every benchmark, including the new DeepSeek-V3 and GPT-OSS MoE training workloads. Megatron Bridge serves as the packaging layer for the NeMo 26.06 training stack that integrates full-iteration CUDA graphs, HybridEP/router optimizations, all-to-all overlap, MXFP8 attention, and pipeline-layout balancing; the blog highlights [DeepSeek-V3 training](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/perf_recipes/deepseek) at **1,648 TFLOPS/GPU** (**6,338 tokens/sec/GPU**) on GB300. To try related runs, start from the [performance recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance); the corresponding container is expected with the NeMo 26.06 release soon.

- [06/04/2026] [**NVIDIA Nemotron 3 Ultra**](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) is now public! Day-0 support for the 550B-A55B hybrid Mamba-Transformer MoE model is available on the [`nemotron_3_ultra`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_ultra) branch, including checkpoint conversion, inference, SFT, PEFT (LoRA), and pretraining examples. Read the [NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-ultra-powers-faster-more-efficient-reasoning-for-long-running-agents/) and see the [examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron_3_ultra/examples/models/nemotron/nemotron_3/ultra/README.md) for the full walkthrough.

- [05/28/2026] [**Step-3.7-Flash**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/stepfun/step37) is now merged on **main**! See the [examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/stepfun/step37/README.md) for sft training details.

- [05/26/2026] [**DeepSeek V4**](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) FP8 support is now available in Megatron Bridge, including HF↔Megatron conversion, quantized checkpoint export with regenerated scale tensors, and downstream verification with Megatron-backend GRPO.

- [05/20/2026] [**DeepSeek V4**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4) is now merged on **main**! See the [examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/deepseek_v4/README.md) for conversion and inference details.

- [05/20/2026] [**Nemotron-3 Nano Omni**](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) day-0 branch support is now merged on **main**! The 30B-A3B MoE multimodal model supports image, video, audio, and text workflows with checkpoint conversion, inference, SFT, and PEFT (LoRA) examples. Read the [NVIDIA Blog](https://blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents/) and see the [examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/nemotron/nemotron_3_omni/README.md) for the full walkthrough.

- [05/19/2026] [**Nemotron-Labs Diffusion**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/nemotron_labs_diffusion) is now supported on **main** with autoregressive-to-diffusion conversion, continuous pretraining, checkpoint conversion, and inference workflows. Read the [NVIDIA Research blog](https://research.nvidia.com/publication/2026-05_nemotron-labs-diffusion-tri-mode-language-model-unifying-autoregressive) for the tri-mode language model overview.

- [05/06/2026] [**Gemma 4 VL 26B-A4B**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/gemma/gemma4_vl) is now supported! Checkpoint conversion, SFT, and PEFT (LoRA) recipes for Google's MoE vision-language model (26B total / 4B active params, 128 experts top-k=8, dual sliding/global attention with K=V tying on full-attention layers) are available on **main**. See the [examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/gemma/gemma4_vl/README.md) for the full walkthrough.

- [04/28/2026] Day 0 support for [**Nemotron-3 Nano Omni**](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16), a 30B-A3B MoE multimodal model that jointly processes image, video, audio, and text. Checkpoint conversion, SFT, and LoRA recipes are available on **main** — see the [examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/nemotron/nemotron_3_omni/README.md) for the full walkthrough.

- [04/19/2026] [**Qwen3.6-35B-A3B**](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is now supported! Qwen3.6 uses the same architecture as Qwen3.5 VL MoE (`Qwen3_5MoeForConditionalGeneration`) and works with the existing [Qwen3.5-VL bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/qwen_vl) out of the box — no code changes needed. HF→Megatron conversion and inference verified.

- [04/16/2026] **Megatron Bridge 0.4.0 released!** New model support (Kimi 2.5, Nemotron 3 Super, Qwen 3.5 VL, MiniMax M2, Sarvam, MiMo, and more), diffusion model collection, sequence-packing improvements, FP8 export, pruning & quantization, Transformers 5.x compatibility, and Python 3.12 migration. Huge thanks to our community contributors: [@HollowMan6](https://github.com/HollowMan6), [@shaltielshmid](https://github.com/shaltielshmid), [@jaeminh](https://github.com/jaeminh), [@pavelgein](https://github.com/pavelgein), [@ShiftyBlock](https://github.com/ShiftyBlock), [@erictang000](https://github.com/erictang000), [@eternally-z](https://github.com/eternally-z), [@Hayak3](https://github.com/Hayak3), and [@mohit-sarvam](https://github.com/mohit-sarvam)! See the [full release notes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/tag/v0.4.0).

- [04/12/2026] [**MiniMax-M2.5 / M2.7**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/minimax/minimax_m2) are now supported! Both models share the same architecture as MiniMax-M2 and work with the existing bridge out of the box — checkpoint conversion and inference verified on real FP8 checkpoints.

- [04/10/2026] [**Qwen3-ASR**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/qwen/qwen3_asr) is now supported! Checkpoint conversion and inference for [Qwen3's ASR model](https://github.com/QwenLM/Qwen3-ASR) are available on **main**.

- [04/09/2026] [**Bailing MoE V2**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/bailing) is now supported! Checkpoint conversion and inference for the Bailing MoE V2 model are available on **main**. Thank you to [@ccclyu](https://github.com/ccclyu) for the community contribution!

- [04/07/2026] Megatron Bridge’s PEFT support was featured at [PyTorch Conference Europe 2026 Talk](https://pytorchconferenceeu2026.sched.com/event/2Juce/optimizing-reinforcement-learning-at-trillion-parameter-scale-songlin-jiang-aalto-university-mind-lab).

- [04/01/2026] [**Kimi K2.5 VL**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/kimi/kimi_k25_vl) is now supported! Checkpoint conversion, inference, and training recipes for [Moonshot AI’s Kimi-K2.5-VL](https://huggingface.co/moonshotai/Kimi-K2.5) vision-language model are available on **main**.

- [03/31/2026] **Agent Skills for Megatron Bridge!** We've added a [`skills/`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/skills) directory with structured guides that AI coding agents (Cursor, Claude Code, Codex, etc.) can use to help you add model support, set up dev environments, tune performance, and more. Try them out, and PRs to improve or add new skills are very welcome!

- [03/26/2026] [**Nemotron 3 Super**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/nemotron/nemotron_3) is now on **main**! Checkpoint conversion and SFT/LoRA recipes (120B-A12B) are available in the main branch. Read the [blog post](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/).

- [03/12/2026] **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.4.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with Megatron-Bridge.

- [12/16/2025] [Mind Lab](https://macaron.im/mindlab) successfully used Megatron-bridge and [VeRL](https://github.com/volcengine/verl) to trained GRPO Lora for Trillion-parameter model on 64 H800 - See their [techblog](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus).

- [12/15/2025] Day 0 support for [NVIDIA-NeMotron-3-Nano-30B-A3B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)! [Reproducible code](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/nemotron/nemotron_3/nano) and custom NGC container: [nvcr.io/nvidia/nemo:25.11.nemotron_3_nano](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=25.11.nemotron_3_nano)

## Overview

NeMo Megatron Bridge is a PyTorch-native library within the [NeMo Framework](https://github.com/NVIDIA-NeMo) that provides pretraining, SFT and LoRA for popular language, vision-language, audio, and multimodal models. It serves as a powerful **bridge, conversion, and verification layer** between 🤗 Hugging Face and [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). It provides bidirectional checkpoint conversion between these formats, enabling other projects to leverage Megatron Core's parallelism capabilities or export models for various inference engines. The bridge includes built-in verification mechanisms to ensure conversion accuracy and checkpoint integrity across different model formats.

On top of the bridge, NeMo Megatron Bridge provides a performant and scalable PyTorch-native training loop that leverages [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) to deliver state-of-the-art training throughput. It supports pretraining and fine-tuning with features like tensor and pipeline parallelism, and mixed precision (FP8, BF16, FP4, etc.). Users can either use existing 🤗 Hugging Face models or define custom PyTorch model definitions for flexible end-to-end workflows.

NeMo Megatron Bridge is a refactor of the [previous NeMo](https://github.com/NVIDIA/NeMo) training stack that adopts a PyTorch-native training loop to provide greater flexibility and customizability for developers.

![image](Repo-Mbridge.png)
### Broad functional support matrix

||Pretrain|SFT|SFT LoRA|RL|RL LoRA|Notes|
|-|-|-|-|-|-|-|
|[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)|Y|Y|Y|N|N|Megatron based *pretraining* library|
|[AutoModel](https://github.com/NVIDIA-NeMo/Automodel)|Y|Y|Y|N|N| PyT DTensor based *pretraining* library|
|[NeMo RL](https://github.com/NVIDIA-NeMo/RL)|N|Y|Y|Y|Y| *Post-training* library with both Megatron and Automodel backends|

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
from megatron.bridge import AutoBridge
from megatron.bridge.recipes.llama.h100 import llama32_1b_pretrain_1gpu_h100_bf16_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

if __name__ == "__main__":
    # The recipe uses the Llama 3.2 1B architecture from Hugging Face.
    # This is random-init pretraining and does not require a converted Megatron checkpoint.
    cfg = llama32_1b_pretrain_1gpu_h100_bf16_config()

    # The recipe already sets cfg.model internally using this pattern.
    # Override cfg.model to choose a different Hugging Face model ID as the architecture source.
    # cfg.model = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B").to_megatron_provider(load_weights=False)

    # Optional: use a local Hugging Face model/config directory instead.
    # cfg.model = AutoBridge.from_hf_pretrained("/path/to/local/hf_model").to_megatron_provider(load_weights=False)

    # Optional: initialize weights from a converted Megatron checkpoint for SFT/PEFT
    # or other pretrained-weight workflows.
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/megatron/checkpoint"

    # Override training parameters
    cfg.train.train_iters = 10
    cfg.scheduler.lr_decay_iters = 10000
    cfg.model.vocab_size = 8192
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    pretrain(cfg, forward_step)
```

You can launch the above script with:

```sh
uv run python -m torch.distributed.run --nproc-per-node=<num devices> /path/to/script.py
```

HF → Megatron conversion is workflow-specific. Use it when you need pretrained HF weights as a Megatron checkpoint,
such as for finetuning from converted weights or for checkpoint round-trip workflows. It is not required for
random-init pretraining from an HF architecture, where `AutoBridge.from_hf_pretrained(...).to_megatron_provider(load_weights=False)`
only reads the architecture configuration.

For runnable recipe, data preparation, and training examples, see the repository
[`tutorials/`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/tutorials) directory.
The [data tutorial index](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/README.md) compares pretraining, text-only SFT, direct Hugging Face SFT, and Energon workflows.

More examples:

- [Conversion scripts overview](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/README.md)
- [Import/Export checkpoints](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/conversion/convert.sh)
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
- **Supervised & Parameter-Efficient Finetuning**: SFT & PEFT implementation tailored for Megatron-based models that supports LoRA, DoRA, and user-defined PEFT methods ([PEFT implementations](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/peft), [finetune module](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/finetune.py), [SFT dataset](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/data/datasets/gpt_sft.py))
- **SOTA Training Recipes**: Pre-configured production-ready training recipes for popular models like Llama 3, with optimized hyperparameters and distributed training configuration ([Llama recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/recipes/llama), [recipe examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models))
- **Performance Optimization**: Built-in support for FP8 training, model parallelism, and memory-efficient techniques to offer high utilization and near-linear scalability to thousands of nodes. ([mixed precision](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/mixed_precision.py), [communication overlap](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/comm_overlap.py), [optimizer utilities](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/utils/optimizer_utils.py))

## Supported Models

Megatron Bridge provides out-of-the-box bridges and training recipes for a wide range of models, built on top of base model architectures from [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). Refer to the [models directory](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models) for the full list of model bridges.

> **Deprecation notice:** Gemma 1 (2B/7B), Gemma 2 (2B/9B/27B), Llama 2,
> Mistral 7B, Mistral Small 3 24B, Nemotron H v1 (4B/8B/47B/56B), and the
> legacy `NemotronForCausalLM` bridge previously documented for Nemotron-4 340B
> are no longer actively maintained or tested against current upstream
> checkpoints. They will be removed in Megatron Bridge 0.7.0.

| Family | Supported variants |
|----------------|--------------------|
| [**Bailing**](docs/models/bailing/index.md) | Ling 2.0 / Ling MoE V2 (Bailing) |
| [**DeepSeek**](docs/models/deepseek/index.md) | DeepSeek V2 / V2 Lite, DeepSeek V3, DeepSeek V4 / V4 Flash |
| [**Diffusion**](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/diffusion/models) | FLUX, LLaDA 1.5, Nemotron-Labs Diffusion, WAN |
| **Ernie** | [Ernie 4.5 MoE](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/ernie), [Ernie 4.5 VL MoE](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/ernie_vl) |
| [**Falcon**](docs/models/falcon/index.md) | Falcon H1 |
| [**Gemma**](docs/models/gemma/index.md) | Gemma / Gemma 2 (deprecated), Gemma 3, Gemma 3-VL, Gemma 4 (26B-A4B MoE / 31B dense), Gemma 4-VL (26B-A4B MoE) |
| [**GLM**](docs/models/glm/index.md) | GLM-4.5 / 4.7 / 4.7-Flash, GLM-4.5V, GLM-5 / 5.1 / 5.2 |
| [**GPT-OSS**](docs/models/gpt_oss/index.md) | GPT-oss |
| [**HY V3**](https://huggingface.co/tencent/Hy3-preview-Base) | Hy3 preview-Base (HF → Megatron checkpoint conversion) |
| [**Kimi**](docs/models/kimi/index.md) | Kimi K2, Kimi-K2.5-VL |
| [**Llama**](docs/models/llama/index.md) | Llama 2 (deprecated), Llama 3 / 3.1 / 3.2 / 3.3 |
| [**MiniMax**](docs/models/minimax/index.md) | MiniMax-M2 / M2.5 / M2.7 |
| [**Mistral**](docs/models/mistral/index.md) | Mistral 7B / Small 3 24B (deprecated), Ministral 3 (3B/8B/14B) |
| [**Xiaomi-MiMo**](docs/models/mimo/index.md) | Xiaomi-MiMo, MiMo-V2-Flash |
| [**Moonlight**](docs/models/moonlight/index.md) | Moonlight |
| [**Nemotron**](docs/models/nemotron/index.md) | Nemotron H v1 (deprecated), Nemotron Nano v2, Nemotron-3 Nano, Nemotron-3 Super, Llama Nemotron, Nemotron Nano v2 VL, Nemotron-3 Nano Omni |
| [**OLMoE**](docs/models/olmoe/index.md) | OLMoE |
| [**Qwen**](docs/models/qwen/index.md) | Qwen2 / Qwen2.5, Qwen3, Qwen3-MoE, Qwen3 Next, Qwen3.5 (dense/MoE), Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Qwen3.6-VL, Qwen2 Audio, Qwen2.5-Omni, Qwen3-Omni, Qwen3-ASR |
| [**Sarvam**](docs/models/sarvam/index.md) | Sarvam |
| [**StepFun**](docs/models/stepfun/index.md) | Step-3.5-Flash, Step-3.7-Flash |

### Launching Recipes

For a conceptual overview of how recipes are structured, overridden, and launched with either `torchrun` or NeMo-Run, read the [Using Recipes guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html).

Runnable tutorials live in `tutorials/recipes/llama` that covers:

- `00_quickstart_pretrain.py` for mock-data pretraining
- `01_quickstart_finetune.py` + LoRA configs
- YAML-driven flows and launch helpers

Dataset preparation tutorials live in [`tutorials/data`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/tutorials/data), including local/materialized text-only SFT and direct Hugging Face SFT examples.

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
