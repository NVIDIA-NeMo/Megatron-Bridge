# BAGEL WebDataset Preparation

This tutorial converts the three modalities in the official BAGEL example
dataset into deterministic WebDataset archives and prepares each archive for
Megatron Energon. The converter preserves image bytes and source row order; it
does not tokenize, transform, sample, shuffle, or pack training examples.

Run the commands from the Megatron Bridge repository root. The input directory
must have this layout:

```text
bagel_example/
├── t2i/*.parquet
├── editing/
│   ├── parquet_info/seedxedit_multi.json
│   └── seedxedit_multi/*.parquet
└── vlm/
    ├── llava_ov_si.jsonl
    └── images/
```

Convert every source row into modality-specific WebDataset members:

```bash
uv run python tutorials/data/bagel/convert_bagel_dataset_to_wds.py \
  --data-root work/data/bagel_example \
  --output work/data/bagel-wds
```

The output contains one tar and one source-order manifest per modality:

```text
bagel-wds/
├── t2i/{t2i.tar,manifest.json}
├── editing/{editing.tar,manifest.json}
└── vlm/{vlm.tar,manifest.json}
```

Prepare each modality independently because each directory represents one
Energon dataset with exactly one tar:

```bash
for group in t2i editing vlm; do
  uv run python tutorials/data/bagel/prepare_bagel_energon.py \
    --dataset-dir "work/data/bagel-wds/${group}" \
    --num-workers 1
done
```

Preparation leaves the tar bytes unchanged and writes Energon indexes,
`dataset.yaml`, and the train split under each directory's `.nv-meta/`.
Training still requires the BAGEL-specific order planner, task encoders, and
packer supplied by the BAGEL dataset provider; the WebDataset tar order alone
does not define the final packed-batch order.
