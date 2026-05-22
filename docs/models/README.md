# Supported Models

This directory contains family-organized documentation for models supported by
Megatron Bridge. Each model page covers supported variants, Hugging Face <->
Megatron Bridge conversion, training recipe links, and model-specific notes.

## Family Index

| Family | Model documentation |
|----------------|---------------------|
| **Bailing** | [Bailing](bailing/index.md) |
| **DeepSeek** | [DeepSeek V2](deepseek/deepseek-v2.md), [DeepSeek V3](deepseek/deepseek-v3.md), [DeepSeek V4](deepseek/deepseek-v4.md) |
| **Falcon** | [Falcon](falcon/index.md) |
| **Gemma** | [Gemma 2](gemma/gemma2.md), [Gemma 3](gemma/gemma3.md), [Gemma 3 VL](gemma/gemma3-vl.md), [Gemma 4 VL](gemma/gemma4-vl.md) |
| **GLM** | [GLM 4.5](glm/glm45.md), [GLM-4.5V](glm/glm-45v.md) |
| **GPT-OSS** | [GPT OSS](gpt_oss/gpt-oss.md) |
| **Kimi** | [Kimi](kimi/index.md) |
| **Llama** | [Llama 3](llama/llama3.md) |
| **MiniMax** | [MiniMax](minimax/index.md) |
| **Mistral** | [Mistral](mistral/mistral.md), [Ministral 3](mistral/ministral3.md) |
| **Xiaomi-MiMo** | [Xiaomi-MiMo](mimo/index.md) |
| **Moonlight** | [Moonlight](moonlight/moonlight.md) |
| **Nemotron** | [Llama Nemotron](nemotron/llama-nemotron.md), [Nemotron H and Nemotron Nano v2](nemotron/nemotronh.md), [Nemotron-3 Nano](nemotron/nemotron3-nano.md), [Nemotron-3 Super](nemotron/nemotron3-super.md), [Nemotron Nano V2 VL](nemotron/nemotron-nano-v2-vl.md), [Nemotron-3 Nano Omni](nemotron/nemotron-3-omni.md) |
| **OLMoE** | [OLMoE](olmoe/olmoe.md) |
| **Qwen** | [Qwen](qwen/qwen.md), [Qwen2.5-VL](qwen/qwen2.5-vl.md), [Qwen3-VL](qwen/qwen3-vl.md), [Qwen3.5 / 3.6](qwen/qwen35-vl.md), [Qwen3-Omni](qwen/qwen3-omni.md) |
| **Sarvam** | [Sarvam](sarvam/index.md) |

## Quick Navigation

### I want to

**Find model-specific docs**
-> Browse the family index above or use the navigation for the model's family.

**Convert models between formats**
-> See [Bridge Guide](../bridge-guide.md) for Hugging Face <-> Megatron
conversion basics. Model pages include model-specific commands where available.

**Get started with training**
-> See [Training Documentation](../training/README.md) for training guides and
[Recipe Usage](../recipe-usage.md) for pre-configured training recipes.

**Add support for a new model**
-> Refer to [Adding New Models](../adding-new-models.md).

## Model Documentation Structure

Each model documentation page typically includes:

1. **Model Overview** - Architecture and key features
2. **Available Variants** - Supported model sizes and configurations
3. **Conversion Examples** - Converting between Hugging Face and Megatron formats
4. **Training Recipes** - Links to training configurations and examples
5. **Architecture Details** - Model-specific features and configurations

## Model Support Overview

### Decoder-Only and Hybrid Backbones

- Bailing, DeepSeek, Falcon, Gemma, GLM, GPT-OSS, Kimi, Llama, MiniMax, Mistral, Moonlight, Nemotron, OLMoE, Qwen, Sarvam, and Xiaomi-MiMo
- MoE and hybrid variants including Bailing, DeepSeek, GLM, GPT-OSS, MiniMax, Nemotron-3, OLMoE, Qwen3-MoE, Qwen3-Next, and Sarvam

### Multimodal Variants

- Gemma 3 VL and Gemma 4 VL
- GLM-4.5V
- Kimi-K2.5-VL
- Ministral 3
- Nemotron Nano V2 VL and Nemotron-3 Nano Omni
- Qwen2.5-VL, Qwen3-VL, Qwen3.5 / 3.6, and Qwen3-Omni

## Related Documentation

- **[Main Documentation](../README.md)** - Return to main documentation
- **[Bridge Guide](../bridge-guide.md)** - Hugging Face <-> Megatron conversion
- **[Bridge Tech Details](../bridge-tech-details.md)** - Technical details of the bridge system
- **[Training Documentation](../training/README.md)** - Comprehensive training guides
- **[Adding New Models](../adding-new-models.md)** - Extending model support
- **[Recipe Usage](../recipe-usage.md)** - Using pre-configured training recipes

## Conversion Support

All model pages document support for one or both conversion directions:

- **Hugging Face -> Megatron Bridge**: Load pretrained weights for training
- **Megatron Bridge -> Hugging Face**: Export trained models for deployment

Conversion features:

- Automatic architecture detection
- Parallelism-aware conversion (TP/PP/VPP/CP/EP)
- Streaming and memory-efficient transfers
- Verification mechanisms for conversion accuracy

Refer to the [Bridge Guide](../bridge-guide.md) for detailed conversion instructions.
