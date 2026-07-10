# Data Tutorials

Choose the tutorial that matches how your examples should reach the training loop:

| Workflow | Tutorial | Runtime path |
| --- | --- | --- |
| LLM pretraining from Megatron binary data | [DCLM preprocessing](dclm/README.md) | `GPTDatasetConfig` → Megatron Core dataset builder |
| Text SFT or PEFT from local JSONL or materialized Hugging Face data | [Text-only SFT](text-only-sft/README.md) | `GPTSFTDatasetConfig` → `GPTSFTDatasetBuilder` |
| Direct Hugging Face SFT for text, vision, or audio | [Direct Hugging Face SFT](direct-hf-sft/README.md) | `DirectHFSFTDatasetConfig` → `DirectHFSFTDatasetBuilder` |
| Large multimodal WebDataset/Energon data | [VALOR32K-AVQA](valor32k-avqa/data-preparation.md) | Energon provider |

## Which SFT path should I use?

- Choose [text-only SFT](text-only-sft/README.md) when you want local or Hugging Face text data normalized into reusable JSONL, need offline sequence packing, or want finite `num_epochs` semantics.
- Choose [direct Hugging Face SFT](direct-hf-sft/README.md) when you want to consume hosted or local Hugging Face rows directly, need text or multimodal examples, or want collate-time in-batch packing.
- Choose [Energon](valor32k-avqa/data-preparation.md) for large sharded multimodal datasets.

Both text SFT paths use serializable, declarative configuration and runtime builders. `GPTSFTDatasetBuilder` materializes Hugging Face text rows before constructing `GPTSFTDataset`; `DirectHFSFTDatasetBuilder` loads and adapts Hugging Face rows directly into `DirectSFTDataset` without intermediate JSONL materialization. The word “Direct” distinguishes that runtime lifecycle—it does not mean that the GPT SFT config cannot select a Hugging Face source. The builders own runtime objects such as tokenizers, processors, collators, and datasets.
