# Nemotron Image Training v3 with Energon and Qwen3-VL

This tutorial prepares a small, self-contained slice of
[`nvidia/Nemotron-Image-Training-v3`](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3)
for the Qwen3-VL Energon path. It does **not** download the complete release. The example uses the `turing` subset:

- 193 image/reasoning conversations
- one 31.7 MiB media tar plus a 3.2 MiB JSONL file
- media included in the Hugging Face repository
- CC-BY-4.0, matching the subset card

The complete release contains 76 subdatasets and about 6.9 million samples. Many subsets provide media only by
reference; read each subset's `README.md`, obtain its upstream media, and reproduce the documented local layout before
converting it. Review the governing terms for every selected subset.

## 1. Download one pinned subset

From the Megatron Bridge repository root, choose separate source and prepared-data directories:

```bash
export NEMOTRON_IMAGE_V3_SOURCE=/data/nemotron-image-training-v3-source
export ENERGON_PATH=/data/nemotron-image-training-v3-turing-energon
export NEMOTRON_IMAGE_V3_REVISION=7656391d4d4cb11ec3722b34f10d499435de0460

uvx --from huggingface_hub hf download nvidia/Nemotron-Image-Training-v3 \
  --repo-type dataset \
  --revision "$NEMOTRON_IMAGE_V3_REVISION" \
  --include "turing/**" \
  --local-dir "$NEMOTRON_IMAGE_V3_SOURCE"
```

The pinned download should contain:

```text
$NEMOTRON_IMAGE_V3_SOURCE/turing/
├── README.md
├── turing.jsonl
└── media/
    └── shard_000000.tar
```

The source media tar is not directly trainable: its image members and the conversations live in separate files. The
next step joins them into matching-key WebDataset records.

## 2. Convert and index it for Energon

Use a fresh output directory:

```bash
uv run python tutorials/data/energon/prepare_nemotron_image_v3.py \
  --source-dir "$NEMOTRON_IMAGE_V3_SOURCE" \
  --output-dir "$ENERGON_PATH" \
  --subsets turing \
  --validation-fraction 0.05 \
  --max-samples-per-tar 1000 \
  --num-workers 2
```

The converter performs four operations:

1. Verifies the pinned `turing` JSONL and media tar sizes and SHA-256 digests.
2. Reads `turing.jsonl` and resolves every image filename from the downloaded media tar.
3. Normalizes the dataset's mixed string/typed message content to processor-ready typed ChatML content.
4. Writes deterministic `train-shard-*` and `val-shard-*` files, then indexes them with the Bridge
   `ChatMLWebdataset` loader metadata.

Inspect the result before training:

```bash
python -m json.tool "$ENERGON_PATH/manifest.json"
tar -tf "$ENERGON_PATH/train-shard-000000.tar" | head
find "$ENERGON_PATH/.nv-meta" -maxdepth 1 -type f -print
```

`manifest.json` pins the dataset revision, selected subsets, split fraction, optional sample cap, and output counts.
Use `--max-samples N` for a smaller converter smoke. If that cap selects no validation records, increase the cap or
set `--validation-fraction 0` and disable validation in the launch. The converter requires an empty output directory,
preventing stale shards, indexes, or unrelated files from being mixed with new data.

To prepare another image-only subset, download both its JSONL and the media layout specified by its subset card, then
add its directory name after `--subsets` and pass `--skip-source-integrity-check` after verifying its provenance
yourself. The converter's built-in size and SHA-256 manifest currently covers only `turing`; in verified mode it reads
media only from that verified archive, so an unrelated loose file cannot shadow an archive member. Video and audio rows
are rejected by this image-focused Qwen3-VL tutorial
rather than being silently dropped.

## 3. Import the Qwen3-VL checkpoint

Import the Hugging Face checkpoint once if a native Megatron checkpoint is not already available:

```bash
export MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
export MODEL_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
export PRETRAINED_CHECKPOINT=/checkpoints/Qwen3-VL-8B-Instruct

./scripts/conversion/convert.sh import \
  --hf-model "$MODEL_ID" \
  --hf-revision "$MODEL_REVISION" \
  --megatron-path "$PRETRAINED_CHECKPOINT"
```

## 4. Launch Qwen3-VL LoRA through `train.sh`

The exact recipe already selects LoRA, the Energon dataset config, and `vlm_step`; those launcher selectors do not need
to be repeated. Mount the prepared dataset, checkpoint, and output paths into the training container:

```bash
export OUTPUT_DIR=/results/qwen3-vl-nemotron-image-v3
mkdir -p "$OUTPUT_DIR"

./scripts/training/train.sh \
  --gpus-per-node 1 \
  --account ACCOUNT --partition PARTITION \
  --container-image /path/to/container.sqsh \
  --mount "$ENERGON_PATH" \
  --mount "$PRETRAINED_CHECKPOINT" \
  --mount "$OUTPUT_DIR" \
  --recipe qwen3_vl_8b_peft_energon_config \
  --seq_length 16384 \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --save_dir "$OUTPUT_DIR/checkpoints" \
  --save_interval 1 \
  --max_steps 1 \
  --global_batch_size 1 \
  --micro_batch_size 1 \
  checkpoint.load=null \
  validation.eval_interval=1 \
  validation.eval_iters=1 \
  dataset.path="$ENERGON_PATH" \
  dataset.micro_batch_size=1 \
  dataset.num_workers=0 \
  dataset.packing_buffer_size=16 \
  dataset.task_encoder.hf_processor_revision="$MODEL_REVISION" \
  model.recompute_granularity=full \
  model.recompute_method=uniform \
  model.recompute_num_layers=1 \
  model.calculate_per_token_loss=True \
  ddp.average_in_collective=False \
  logger.log_interval=1 \
  logger.save_config_filepath="$OUTPUT_DIR/resolved-config.yaml"
```

The pinned `turing` subset contains long reasoning targets: with the pinned Qwen3-VL processor and default image
pixel bounds, the longest observed row is 15,608 tokens. The explicit 16,384-token setting keeps every source row
eligible; the recipe's 4,096-token default would skip overlength rows. Full activation recompute keeps the documented
one-GPU LoRA smoke within an 80 GiB H100. Native packing requires physical micro-batch size 1 in both the training and
Energon configs, per-token loss, and non-averaged DDP collectives; these settings are not optional for this packed run.
The processor revision is also required to reproduce the measured sequence lengths and pins the processor's tokenizer
artifacts to the same model revision.

The one-step limit, global batch size 1, validation/checkpoint interval 1, worker count 0, and log interval 1 are
smoke/debug settings rather than packing requirements. They preserve the validated baseline and make its loss and
checkpoint evidence visible. After it passes, increase the step count, global batch size, and worker count for the
target run. `num_val_workers` defaults to `num_workers`. Benchmark recompute against context parallelism on the target
hardware instead of silently reducing the dataset.

For a local interactive smoke, export the same `ENERGON_PATH` and `PRETRAINED_CHECKPOINT`, then run
`bash examples/models/qwen/qwen3_vl/peft_energon.sh`.

Successful preparation is not sufficient training evidence. Confirm that the resolved config points at the prepared
root and uses `QwenVLEnergonTaskEncoderConfig`, then require finite training and validation loss, zero skipped/NaN
iterations, and a written checkpoint. Native packing uses physical MBS1; the global batch size counts packs rather than
source conversations. Keep shards, split metadata, worker counts, sequence length, topology, processor, and
`packing_buffer_size` unchanged when resuming.
