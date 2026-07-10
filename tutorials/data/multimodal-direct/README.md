# Multimodal Direct-HF Tutorial

Use this path when image, video, audio, or omni conversations fit naturally in a Hugging Face dataset or local JSON/JSONL files. Hugging Face datasets owns source loading; `DirectHFSFTDatasetBuilder` constructs the runtime processor, model collator, and repeating training datasets. No preloaded or local-file provider is involved.

This tutorial runs Qwen3-VL 8B LoRA on one H100 with local images. For large sharded media collections, use the [Energon tutorial](../energon/README.md) instead.

## 1. Prepare processor-native JSONL

From the repository root:

```bash
export DATA_DIR=/tmp/bridge-multimodal-direct

uv run python tutorials/data/multimodal-direct/prepare_example_data.py \
  --output-dir "$DATA_DIR"
```

The script creates six deterministic 448×448 PNGs plus train, validation, and test JSONL files:

```text
/tmp/bridge-multimodal-direct/
├── images/
│   ├── red.png
│   └── ...
├── training.jsonl
├── validation.jsonl
└── test.jsonl
```

Each row stores media inside the conversation using the selected processor's schema:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "/absolute/path/red.png"}, {"type": "text", "text": "What is the dominant color?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "The image is red."}]}]}
```

Paths and URLs are interpreted by the model processor and must be resolvable by every data worker. The removed preloaded `<image>` plus top-level `images` schema is not rewritten automatically.

## 2. Import the model checkpoint

The training processor can come from the Hub, but model weights should be imported once into a native Megatron checkpoint:

```bash
export MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
export WORKSPACE=${WORKSPACE:-/workspace}
export PRETRAINED_CHECKPOINT="$WORKSPACE/models/Qwen3-VL-8B-Instruct"

uv run python examples/conversion/convert_checkpoints.py import \
  --hf-model "$MODEL_ID" \
  --megatron-path "$PRETRAINED_CHECKPOINT"
```

The import requires enough host/GPU memory for the 8B checkpoint and access to the model files through `HF_HOME`/`HF_TOKEN` when needed.

## 3. Run a one-GPU LoRA smoke

The Qwen3-VL 8B PEFT recipe is a one-GPU recipe. Start unpacked with one iteration and no evaluation:

```bash
export WANDB_MODE=disabled
export OUTPUT_DIR="$WORKSPACE/results/qwen3-vl-direct-smoke"

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_config \
  --dataset vlm-hf \
  --step_func qwen3_vl_step \
  --peft_scheme lora \
  --seq_length 1024 \
  checkpoint.pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
  checkpoint.load=null \
  checkpoint.save="$OUTPUT_DIR/checkpoints" \
  checkpoint.save_interval=1 \
  train.train_iters=1 \
  train.global_batch_size=1 \
  train.micro_batch_size=1 \
  dataset.source.dataset_name=null \
  dataset.source.path_or_dataset=json \
  dataset.source.split=train \
  "dataset.source.load_kwargs={data_files:{train:${DATA_DIR}/training.jsonl}}" \
  dataset.hf_processor_path="$MODEL_ID" \
  dataset.do_validation=False \
  dataset.do_test=False \
  dataset.num_workers=0 \
  dataset.persistent_workers=False \
  dataset.enable_in_batch_packing=False \
  logger.log_interval=1
```

Success means the launcher loads the native checkpoint, the Qwen processor opens the local PNGs, and iteration 1 reports a finite loss before writing the adapter checkpoint.

`--dataset vlm-hf` selects the generic `DirectHFSFTDatasetConfig`; the remaining overrides select Qwen's processor and the local JSON source. Passing the selector also makes `--seq_length` update both model and dataset lengths.

## 4. Run the MedPix Hub preset with W&B

`medpix` is a built-in source preset for
[`mmoukouba/MedPix-VQA`](https://huggingface.co/datasets/mmoukouba/MedPix-VQA). It owns the Hub path,
the MedPix schema adapter, and the published train/validation split names. The dataset has no test split, so disable
test data. With `WANDB_API_KEY` configured (or after `wandb login`), this command runs three Qwen3-VL LoRA steps,
one validation step, checkpoint save, and online metric logging:

```bash
export OUTPUT_DIR="$WORKSPACE/results/qwen3-vl-medpix-direct"

uv run python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe qwen3_vl_8b_peft_config \
  --dataset vlm-hf \
  --step_func qwen3_vl_step \
  --peft_scheme lora \
  --seq_length 1024 \
  checkpoint.pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
  checkpoint.load=null \
  checkpoint.save="$OUTPUT_DIR/checkpoints" \
  checkpoint.save_interval=3 \
  train.train_iters=3 \
  train.global_batch_size=1 \
  train.micro_batch_size=1 \
  validation.eval_interval=3 \
  validation.eval_iters=1 \
  validation.eval_micro_batch_size=1 \
  dataset.source.dataset_name=medpix \
  dataset.hf_processor_path="$MODEL_ID" \
  dataset.do_validation=True \
  dataset.do_test=False \
  dataset.num_workers=0 \
  dataset.persistent_workers=False \
  dataset.enable_in_batch_packing=False \
  scheduler.lr_warmup_iters=0 \
  scheduler.lr_decay_iters=3 \
  logger.log_interval=1 \
  logger.tensorboard_log_interval=1 \
  logger.log_throughput_to_tensorboard=True \
  logger.log_memory_to_tensorboard=True \
  logger.wandb_project=bridge-qwen3-vl-medpix \
  logger.wandb_exp_name=direct-hf-medpix \
  logger.wandb_save_dir="$OUTPUT_DIR/wandb"
```

The current Direct-HF implementation materializes each requested split before training. The first MedPix run downloads
the complete Hub shards and needs substantially more host memory than the three target training samples alone imply.
For a smaller training-only data-path check, add the following overrides; the nested quotes around the sliced split are
required by Hydra:

```bash
'dataset.source.split="train[:16]"' \
dataset.do_validation=False \
validation.eval_iters=0
```

Use the normal preset-derived validation split for a real training run. A CLI mapping cannot construct an optional
`validation_source` that started as `None`; define custom per-split sources in Python as shown next.

## 5. Configure explicit validation and test files

File-backed sources do not invent missing splits. For a production Python recipe, give every enabled split its own serializable source:

```python
from megatron.bridge.data.builders import DirectHFSFTDatasetConfig, HFDatasetSourceConfig

cfg.dataset = DirectHFSFTDatasetConfig(
    seq_length=cfg.model.seq_length,
    hf_processor_path="Qwen/Qwen3-VL-8B-Instruct",
    source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="train",
        load_kwargs={"data_files": {"train": "/data/vlm/training.jsonl"}},
    ),
    validation_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="validation",
        load_kwargs={"data_files": {"validation": "/data/vlm/validation.jsonl"}},
    ),
    test_source=HFDatasetSourceConfig(
        path_or_dataset="json",
        split="test",
        load_kwargs={"data_files": {"test": "/data/vlm/test.jsonl"}},
    ),
)
```

Disable `do_validation` or `do_test` when that split is intentionally absent.

## 6. Enable Qwen in-batch packing

Qwen3-VL defers packing until its model step has consumed the original visual tensors. Packing requires a configured micro batch greater than one:

```bash
train.micro_batch_size=2 \
train.global_batch_size=2 \
dataset.enable_in_batch_packing=True \
dataset.defer_in_batch_packing_to_step=True
```

Do not apply text-only offline packed-SFT settings to this path. With context parallelism, keep `model.calculate_per_token_loss=True` and `ddp.average_in_collective=False`; `ConfigContainer` derives the required CP/SP packing multiple.

## Config and runtime ownership

| Setting | Meaning | Owner |
| --- | --- | --- |
| `source.*` | Hub/custom/JSON source and optional schema adapter | Serializable config |
| `hf_processor_path` | Processor identity | Serializable config; loaded by builder |
| typed `content` media | Processor-native image/video/audio references | Dataset rows |
| `pad_*`, `enable_in_batch_packing` | Batch shape policy | Serializable config |
| processor, collator, `DirectSFTDataset` | Runtime objects | `DirectHFSFTDatasetBuilder` |

## Troubleshooting

- `Unknown split`: provide an explicit `validation_source`/`test_source`, or disable the missing stage.
- `No VLM collate function is registered`: `hf_processor_path` resolved to a processor type without a registered model collator.
- Image cannot be opened: use absolute paths, shared paths, or URLs reachable from every worker.
- All tokens are masked: verify the conversation has an assistant turn accepted by the processor chat template.
- Packing validation fails: use micro batch size greater than one and retain Qwen's deferred-packing setting.
- OOM on one GPU: reduce sequence length first; this tutorial uses LoRA because full Qwen3-VL 8B SFT is a two-GPU recipe.

For hosted text/chat sources and prompt-completion preprocessing, see the broader [Direct SFT tutorial](../direct-hf-sft/README.md).
