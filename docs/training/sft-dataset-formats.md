# SFT Dataset Formats

This guide explains the dataset formats supported by Megatron Bridge for supervised fine-tuning (SFT). Megatron Bridge provides flexible dataset support to accommodate various data structures, from simple prompt-completion pairs to complex multi-turn conversations.

## Overview

Megatron Bridge supports three main dataset classes for SFT:

1. **GPTSFTDataset** - Standard JSONL format with customizable prompt templates
2. **GPTSFTChatDataset** - Chat-based conversations with HuggingFace chat templates or ShareGPT format
3. **GPTSFTPackedDataset** - Packed sequences for efficient training

All dataset classes support the `answer_only_loss` parameter, which controls whether loss is computed on the entire sequence or only on the assistant's responses.

## Loss Masking with `answer_only_loss`

The `answer_only_loss` parameter determines which tokens contribute to the training loss:

- **`answer_only_loss=True`** (default): Loss computed only on assistant responses, not on prompts or system messages
- **`answer_only_loss=False`**: Loss computed on all tokens in the sequence

This parameter is critical for chat-based fine-tuning where you want the model to learn to generate responses without being penalized for the input prompts.

**Comparison with HF TRL**: If you're migrating from HuggingFace TRL, `answer_only_loss=True` is equivalent to TRL's response template approach for masking prompt tokens.

## Standard JSONL Format

The standard JSONL format uses customizable prompt templates to combine input fields.

### Basic Example

**Dataset file** (`training.jsonl`):
```json
{"input": "What is the capital of France?", "output": "The capital of France is Paris."}
{"input": "Explain photosynthesis.", "output": "Photosynthesis is the process by which plants convert light energy into chemical energy."}
```

**Configuration**:
```python
from megatron.bridge.data.datasets.sft import create_sft_dataset

dataset = create_sft_dataset(
    path=Path("training.jsonl"),
    tokenizer=tokenizer,
    seq_length=2048,
    prompt_template="{input} {output}",
    label_key="output",
    answer_only_loss=True,  # Compute loss only on output
    truncation_field="input",
    add_bos=True,
    add_eos=True,
)
```

### Prompt Template Syntax

The `prompt_template` parameter is an f-string that defines how fields are combined:

```python
# Simple prompt-completion
prompt_template="{input} {output}"

# Question-answer format
prompt_template="Q: {question}\n\nA: {answer}"

# Context-based format
prompt_template="Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
```

**Important**: The `label_key` field must appear at the end of the prompt template. This ensures proper loss masking when `answer_only_loss=True`.

### Truncation

When sequences exceed `seq_length`, the `truncation_field` parameter controls which fields to truncate:

```python
# Truncate only the input field
truncation_field="input"

# Truncate multiple fields (comma-separated)
truncation_field="input,context"

# Truncation method
truncation_method="right"  # or "left"
```

### Multiple Fields Example

```json
{"context": "The Eiffel Tower was built in 1889.", "question": "When was the Eiffel Tower built?", "answer": "1889"}
```

```python
dataset = create_sft_dataset(
    path=Path("training.jsonl"),
    tokenizer=tokenizer,
    prompt_template="Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}",
    label_key="answer",
    truncation_field="context,question",
    answer_only_loss=True,
)
```

## HuggingFace Chat Template Format

The chat template format uses HuggingFace's `apply_chat_template` for consistent message formatting.

### Dataset Format

**Training file** (`chat_training.jsonl`):
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
{
  "messages": [
    {"role": "user", "content": "Explain quantum computing in simple terms."},
    {"role": "assistant", "content": "Quantum computing uses quantum mechanics principles to process information differently than classical computers."}
  ]
}
```

### Configuration

```python
dataset = create_sft_dataset(
    path=Path("chat_training.jsonl"),
    tokenizer=tokenizer,
    seq_length=2048,
    chat=True,
    use_hf_tokenizer_chat_template=True,
    answer_only_loss=True,
)
```

### Multi-Turn Conversations

The chat format naturally supports multi-turn conversations:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a string in Python?"},
    {"role": "assistant", "content": "You can reverse a string using slicing: `text[::-1]`"},
    {"role": "user", "content": "Can you show a function example?"},
    {"role": "assistant", "content": "Sure! Here's a function:\n\n```python\ndef reverse_string(text):\n    return text[::-1]\n```"}
  ]
}
```

With `answer_only_loss=True`, loss is computed only on the assistant's responses across all turns.

### Tool/Function Calling Support

The chat format supports tool schemas for function calling:

```python
# Define tool schemas
tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
]

dataset = create_sft_dataset(
    path=Path("tool_training.jsonl"),
    tokenizer=tokenizer,
    chat=True,
    use_hf_tokenizer_chat_template=True,
    tool_schemas=tool_schemas,
    answer_only_loss=True,
)
```

**Dataset with tools**:
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "Let me check that for you.", "tool_calls": [{"name": "get_weather", "arguments": {"location": "Paris"}}]}
  ],
  "tools": [...]
}
```

## ShareGPT Format

ShareGPT format is automatically converted to HuggingFace chat template format when `use_hf_tokenizer_chat_template=True`.

### Dataset Format

```json
{
  "conversations": [
    {"from": "user", "value": "What is machine learning?"},
    {"from": "assistant", "value": "Machine learning is a subset of AI that enables systems to learn from data."}
  ]
}
{
  "system": "You are a helpful assistant.",
  "conversations": [
    {"from": "user", "value": "Explain neural networks."},
    {"from": "assistant", "value": "Neural networks are computing systems inspired by biological neural networks."}
  ]
}
```

### Configuration

```python
dataset = create_sft_dataset(
    path=Path("sharegpt_training.jsonl"),
    tokenizer=tokenizer,
    seq_length=2048,
    chat=True,
    use_hf_tokenizer_chat_template=True,
    answer_only_loss=True,
)
```

**Note**: When using ShareGPT format, the dataset automatically converts to the HuggingFace messages format internally.

## Packed Sequences

Packed sequences concatenate multiple training examples into single sequences for efficient GPU utilization. See [Packed Sequences](packed-sequences.md) for detailed documentation.

### Quick Example

```python
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

# The dataset is automatically packed during training
dataset = create_sft_dataset(
    path=Path("training.npy"),  # .npy file indicates packed dataset
    tokenizer=tokenizer,
    seq_length=2048,
    answer_only_loss=True,
)
```

To prepare packed datasets, configure `PackedSequenceSpecs` in your training configuration. The packing process preserves loss masking from `answer_only_loss`.

## Configuration Reference

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `Path` | Required | Path to dataset file |
| `tokenizer` | `MegatronTokenizer` | Required | Tokenizer instance |
| `seq_length` | `int` | `2048` | Maximum sequence length |
| `answer_only_loss` | `bool` | `True` | Compute loss only on answers |
| `add_bos` | `bool` | `False` | Add beginning-of-sequence token |
| `add_eos` | `bool` | `True` | Add end-of-sequence token |
| `seed` | `int` | `1234` | Random seed for shuffling |

### Standard JSONL Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_template` | `str` | Required | Template for combining fields |
| `label_key` | `str` | `"output"` | Key for the answer/completion field |
| `truncation_field` | `str` | `"input"` | Fields to truncate (comma-separated) |
| `truncation_method` | `str` | `"right"` | Truncation direction (`"left"` or `"right"`) |

### Chat Format Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat` | `bool` | `False` | Enable chat dataset mode |
| `use_hf_tokenizer_chat_template` | `bool` | `False` | Use HF's `apply_chat_template` |
| `tool_schemas` | `str \| dict` | `None` | Tool/function schemas for function calling |

## Migration from HuggingFace TRL

If you're migrating from HuggingFace TRL's SFTTrainer, here's how Megatron Bridge parameters map to TRL:

| TRL Parameter | Megatron Bridge Equivalent | Notes |
|---------------|---------------------------|-------|
| `dataset_text_field` | `prompt_template` | TRL uses a single field; Megatron Bridge uses a template |
| `packing` | `PackedSequenceSpecs.packed_sequence_size > 0` | See [Packed Sequences](packed-sequences.md) |
| Response template masking | `answer_only_loss=True` | Automatically masks prompt tokens |
| `max_seq_length` | `seq_length` | Maximum sequence length |

### Example Migration

**TRL configuration**:
```python
# HuggingFace TRL
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
)
```

**Megatron Bridge equivalent**:
```python
# Megatron Bridge
from megatron.bridge.data.datasets.sft import create_sft_dataset
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

dataset = create_sft_dataset(
    path=Path("training.jsonl"),
    tokenizer=tokenizer,
    seq_length=2048,
    prompt_template="{text}",  # If data has single 'text' field
    answer_only_loss=True,
    # Packing configured separately via PackedSequenceSpecs
)
```

## Best Practices

### Choosing a Format

- **Standard JSONL**: Use for simple prompt-completion pairs with custom formatting needs
- **HF Chat Template**: Use for chat-based fine-tuning with consistent formatting across models
- **ShareGPT**: Use when working with existing ShareGPT-format datasets
- **Packed Sequences**: Use for efficient training with variable-length sequences

### Loss Masking

- Set `answer_only_loss=True` for instruction-following and chat models
- Set `answer_only_loss=False` for completion-style training where the entire sequence is relevant

### Template Design

- Always place the `label_key` field at the end of `prompt_template`
- Include necessary formatting tokens (newlines, colons, etc.) in the template
- Test your template with a few examples before full training

### Truncation Strategy

- Truncate context/input fields rather than answer fields
- Use `truncation_method="left"` for contexts where recent information is more important
- Use `truncation_method="right"` for most other cases

## Troubleshooting

### Loss is not converging

- Verify `answer_only_loss=True` if training a chat/instruction model
- Check that your `label_key` is at the end of `prompt_template`
- Confirm dataset format matches your configuration

### Tokenization issues

- Ensure `add_bos` and `add_eos` match your model's requirements
- Verify tokenizer compatibility with your dataset format
- For chat formats, ensure your tokenizer has a chat template defined

### Memory issues

- Consider using [packed sequences](packed-sequences.md) for better GPU utilization
- Reduce `seq_length` if sequences are consistently shorter
- Enable mixed precision training

## API Reference

For detailed API documentation, see:

- {py:func}`bridge.data.datasets.sft.create_sft_dataset` - Dataset factory function
- {py:class}`bridge.data.datasets.sft.GPTSFTDataset` - Standard JSONL dataset
- {py:class}`bridge.data.datasets.sft.GPTSFTChatDataset` - Chat format dataset
- {py:class}`bridge.data.datasets.sft.GPTSFTPackedDataset` - Packed sequence dataset
- {py:func}`bridge.data.datasets.utils._chat_preprocess` - Chat preprocessing implementation
- {py:class}`bridge.data.datasets.packed_sequence.PackedSequenceSpecs` - Packed sequence configuration
