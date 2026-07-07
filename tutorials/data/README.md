# Data Tutorials

Choose the tutorial that matches how examples reach the training loop:

| Workflow | Tutorial | Runtime path |
| --- | --- | --- |
| LLM pretraining from Megatron binary data | [DCLM preprocessing](dclm/README.md) | `GPTDatasetConfig` → Megatron Core dataset builder |
| Text SFT or PEFT from local JSONL or materialized Hugging Face data | [GPT SFT](gpt-sft/README.md) | `GPTSFTDatasetConfig` → `GPTSFTDatasetBuilder` |
| Direct Hugging Face conversations for text, vision, or audio | [Hugging Face conversation](hf-conversation/README.md) | `HFConversationDatasetConfig` → `HFConversationDatasetBuilder` |
| Large multimodal WebDataset/Energon data | [VALOR32K-AVQA](valor32k-avqa/data-preparation.md) | Energon provider |

The configuration objects are serializable and declarative. Builders are resolved by the training framework and own runtime objects such as tokenizers, processors, collators, and datasets.
