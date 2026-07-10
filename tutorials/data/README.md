# Data Tutorials

Choose the tutorial that matches how your examples should reach the training loop:

| Workflow | Status and starting data | Tutorial | Runtime path |
| --- | --- | --- | --- |
| Pretraining from Megatron `.bin`/`.idx` | Recommended; prepared token IDs | [DCLM preprocessing](dclm/README.md) | `GPTDatasetConfig` → Megatron Core dataset builder |
| Text SFT or PEFT directly from Hugging Face | Recommended for hosted chat/prompt-completion rows or local JSON/JSONL loaded through Hugging Face datasets | [Hugging Face text-only](hf-text-only/README.md) | `HFDatasetSourceConfig` → `DirectHFSFTDatasetConfig` → `DirectHFSFTDatasetBuilder` |
| Text SFT or PEFT from current prepared data | Transitional; materialized JSONL and optional packed Parquet | [Prepared text-only](text-only-sft/README.md) | `GPTSFTDatasetConfig` → `GPTSFTDatasetBuilder` |
| Text SFT or PEFT from prepared `.bin`/`.idx` | Planned Issue #4664 workflow; not available yet | Follow Issue #4664 | Future prepared-SFT builder |
| Multimodal SFT or PEFT from Hugging Face | Recommended for hosted vision, video, audio, and omni data or local JSON/JSONL loaded through Hugging Face datasets | [Hugging Face multimodal](hf-multimodal/README.md) | `HFDatasetSourceConfig` → `DirectHFSFTDatasetConfig` → `DirectHFSFTDatasetBuilder` |
| Large sharded multimodal training | Recommended for WebDataset/Energon data | [Multimodal Energon](energon/README.md) | `EnergonDatasetConfig` → `EnergonDatasetBuilder` |

## Which SFT path should I use?

- Choose [Hugging Face text-only](hf-text-only/README.md) for new on-the-fly text SFT, including hosted chat datasets, local conversation JSON/JSONL, and prompt-completion rows.
- Choose [Hugging Face multimodal](hf-multimodal/README.md) for processor-native image, video, audio, or omni conversations. Hosted datasets and local JSON/JSONL accepted by the Hugging Face `json` loader use the same `HFDatasetSourceConfig`; there is no separate preloaded/local VLM provider.
- Choose the current [text-only SFT](text-only-sft/README.md) path when you specifically need reusable local JSONL, finite `num_epochs`, or offline packed-Parquet behavior. This is a transitional prepared-data path until the planned `.bin`/`.idx` SFT replacement is available.
- Choose the [Energon tutorial](energon/README.md) for large sharded multimodal datasets; [VALOR32K-AVQA](valor32k-avqa/data-preparation.md) is the production audio-video example.

All current SFT paths use serializable, declarative configuration and runtime builders. `GPTSFTDatasetBuilder` materializes Hugging Face text rows before constructing `GPTSFTDataset`; `DirectHFSFTDatasetBuilder` loads rows from Hugging Face datasets directly into `DirectSFTDataset` without intermediate JSONL materialization; `EnergonDatasetBuilder` constructs the configured task encoder and WebDataset loaders. The builders own runtime objects such as tokenizers, processors, task encoders, collators, and datasets.
