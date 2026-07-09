# Data Tutorials

Choose the tutorial that matches how your examples should reach the training loop:

| Workflow | Status and starting data | Tutorial | Runtime path |
| --- | --- | --- | --- |
| Pretraining from Megatron `.bin`/`.idx` | Recommended; prepared token IDs | [DCLM preprocessing](dclm/README.md) | `GPTDatasetConfig` → Megatron Core dataset builder |
| Text SFT or PEFT directly from Hugging Face | Recommended for on-the-fly chat-template processing | [Direct SFT](direct-hf-sft/README.md) | `DirectHFSFTDatasetConfig` → `DirectHFSFTDatasetBuilder` |
| Text SFT or PEFT from prepared `.bin`/`.idx` | Planned Issue #4664 workflow; not available yet | Follow Issue #4664 | Future prepared-SFT builder |
| Multimodal SFT or PEFT from Hugging Face | Recommended for ordinary-size vision, video, audio, and omni data; local JSON/JSONL must use the HF `json` loader | [Direct SFT](direct-hf-sft/README.md) | `HFDatasetSourceConfig` → `DirectHFSFTDatasetConfig` → `DirectHFSFTDatasetBuilder` |
| Large sharded multimodal training | Recommended for WebDataset/Energon data | [VALOR32K-AVQA](valor32k-avqa/data-preparation.md) | Energon provider |

## Which SFT path should I use?

- Choose [direct SFT](direct-hf-sft/README.md) for new on-the-fly text and multimodal SFT. Hub datasets and local JSON/JSONL accepted by the Hugging Face `json` loader both use `HFDatasetSourceConfig`; there is no separate preloaded/local VLM provider.
- Choose the current [text-only SFT](text-only-sft/README.md) path when you specifically need reusable local JSONL, finite `num_epochs`, or offline packed-Parquet behavior. This is a transitional prepared-data path until the planned `.bin`/`.idx` SFT replacement is available.
- Choose [Energon](valor32k-avqa/data-preparation.md) for large sharded multimodal datasets.

Both current SFT paths use serializable, declarative configuration and runtime builders. `GPTSFTDatasetBuilder` materializes Hugging Face text rows before constructing `GPTSFTDataset`; `DirectHFSFTDatasetBuilder` loads rows from Hugging Face datasets directly into `DirectSFTDataset` without intermediate JSONL materialization. The word “Direct” distinguishes that runtime lifecycle. The builders own runtime objects such as tokenizers, processors, collators, and datasets.
