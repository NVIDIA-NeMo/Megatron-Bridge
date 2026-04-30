# Supported Models

This directory contains documentation for all models supported by Megatron Bridge, including Large Language Models (LLMs) and Vision Language Models (VLMs). Each model documentation includes architecture details, conversion examples for Hugging Face ↔ Megatron Bridge, and links to training recipes.

## Model Categories

Megatron Bridge supports two main categories of models:

### 🔤 Large Language Models (LLMs)

Text-only models for language understanding and generation tasks.

| Category | Model Count | Documentation |
|----------|-------------|---------------|
| **Large Language Models** | 13 models | [LLM Documentation](llm/README.md) |

**Supported LLM Families:**

- DeepSeek (V2, V3)
- Gemma (2, 3)
- GLM-4.5
- GPT-OSS
- LLaMA (3, Nemotron)
- Mistral
- Moonlight
- Nemotron-H
- OLMoE
- Qwen (2, 2.5, 3, 3 MoE, 3-Next)

### 🖼️ Vision Language Models (VLMs)

Multimodal models that combine vision and language capabilities.

| Category | Model Count | Documentation |
|----------|-------------|---------------|
| **Vision Language Models** | 5 models | [VLM Documentation](vlm/README.md) |

**Supported VLM Families:**

- Gemma 3 VL
- Nemotron-3 Nano Omni (image + video + audio + text)
- Nemotron Nano V2 VL
- Qwen (2.5 VL, 3 VL)

---

## Quick Navigation

### I want to

**🔍 Find a specific LLM model**
→ Browse [Large Language Models](llm/README.md) documentation

**🖼️ Find a specific VLM model**
→ Browse [Vision Language Models](vlm/README.md) documentation

**🔄 Convert models between formats**
→ See [Bridge Guide](../bridge-guide.md) for Hugging Face ↔ Megatron conversion

**🚀 Get started with training**
→ See [Training Documentation](../training/README.md) for training guides

**📚 Understand model architectures**
→ Each model page documents architecture-specific features and configurations

**🔧 Add support for a new model**
→ Refer to [Adding New Models](../adding-new-models.md)

**📊 Use training recipes**
→ Read [Recipe Usage](../recipe-usage.md) for pre-configured training recipes

---

## Model Documentation Structure

Each model documentation page typically includes:

1. **Model Overview** - Architecture and key features
2. **Available Variants** - Supported model sizes and configurations
3. **Conversion Examples** - Converting between Hugging Face and Megatron formats
4. **Training Recipes** - Links to training configurations and examples
5. **Architecture Details** - Model-specific features and configurations

---

## Common Tasks by Model Type

### For LLM Models

**Training:**

- Pretraining on large corpora
- Supervised fine-tuning (SFT)
- Parameter-efficient fine-tuning (PEFT/LoRA)
- Preference optimization (DPO)

**Deployment:**

- Export to Hugging Face format
- Integration with inference engines
- Model serving and deployment

**Use Cases:**

- Text generation
- Question answering
- Conversational AI
- Code generation

### For VLM Models

**Training:**

- Multimodal pretraining
- Vision-language alignment
- Fine-tuning on visual tasks

**Deployment:**

- Export to Hugging Face format
- Multimodal inference

**Use Cases:**

- Image captioning
- Visual question answering
- Document understanding
- Multimodal reasoning

---

## Related Documentation

### Getting Started

- **[Main Documentation](../README.md)** - Return to main documentation
- **[Bridge Guide](../bridge-guide.md)** - Hugging Face ↔ Megatron conversion
- **[Bridge Tech Details](../bridge-tech-details.md)** - Technical details of the bridge system

### Training Resources

- **[Training Documentation](../training/README.md)** - Comprehensive training guides
- **[Configuration Container](../training/config-container-overview.md)** - Training configuration
- **[Parallelisms Guide](../parallelisms.md)** - Data and model parallelism strategies
- **[Performance Guide](../performance-guide.md)** - Performance optimization

### Advanced Topics

- **[Adding New Models](../adding-new-models.md)** - Extending model support
- **[Recipe Usage](../recipe-usage.md)** - Using pre-configured training recipes
- **[Bridge RL Integration](../bridge-rl-integration.md)** - Reinforcement learning integration
- **[PEFT](../training/peft.md)** - Parameter-efficient fine-tuning

---

## Model Support Overview

### By Architecture Type

**Decoder-Only (Autoregressive):**

- GPT-style models (GPT-OSS)
- LLaMA family (LLaMA 3, LLaMA Nemotron)
- Qwen family (Qwen 2, 2.5, 3, 3-Next)
- Gemma family (Gemma 2, 3)
- DeepSeek family (DeepSeek V2, V3)
- Mistral, Moonlight, Nemotron-H, GLM-4.5

**Mixture-of-Experts (MoE):**

- Qwen 3 MoE, Qwen 3-Next
- DeepSeek V2, V3
- OLMoE

**Vision-Language (Multimodal):**

- Gemma 3 VL
- Qwen 2.5 VL, Qwen 3 VL
- Nemotron Nano V2 VL
- Nemotron-3 Nano Omni

### By Provider

**Meta/LLaMA:**

- LLaMA 3

**NVIDIA:**

- LLaMA Nemotron
- Nemotron-H
- Nemotron Nano V2 VL
- Nemotron-3 Nano Omni

**Alibaba Cloud:**

- Qwen (2, 2.5, 3, 3 MoE, 3-Next)
- Qwen VL (2.5, 3)

**Google:**

- Gemma (2, 3)
- Gemma 3 VL

**DeepSeek:**

- DeepSeek (V2, V3)

**Other:**

- Mistral AI (Mistral)
- GLM-4.5
- GPT-OSS
- Moonlight
- OLMoE

---

## Conversion Support

All models support bidirectional conversion:

- **Hugging Face → Megatron Bridge**: Load pretrained weights for training
- **Megatron Bridge → Hugging Face**: Export trained models for deployment

Conversion features:

- Automatic architecture detection
- Parallelism-aware conversion (TP/PP/VPP/CP/EP)
- Streaming and memory-efficient transfers
- Verification mechanisms for conversion accuracy

Refer to the [Bridge Guide](../bridge-guide.md) for detailed conversion instructions.

---

**Ready to explore?** Choose a model category:

- [Large Language Models (LLMs)](llm/README.md)
- [Vision Language Models (VLMs)](vlm/README.md)

Or return to the [main documentation](../README.md).
