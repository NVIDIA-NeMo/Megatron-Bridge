# DCLM Data Preprocessing Tutorial

This guide explains how to download, decompress, merge, and preprocess the **DCLM-baseline** dataset for language model pretraining.  

The **DCLM-baseline** dataset contains **4T tokens** across **3B documents**, achieving strong performance on language model benchmarks.

**Dataset source:** [Hugging Face DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/tree/main/global-shard_01_of_10)

You can follow the examples below, which use Python scripts (useful for preprocessing subdatasets in parallel), or follow our [tutorial](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/dclm/data_pipeline.ipynb) in the Jupyter notebook.

---


## Dataset Overview

- The dataset is organized into **10 global shards**: `global-shard_01_of_10` … `global-shard_10_of_10`.  
- Each global shard contains **10 local shards**: `local-shard_0_of_10` … `local-shard_9_of_10`.  
- Each local shard contains ~280 compressed JSONL files: `*.jsonl.zst`.  
- Total dataset size: **~722 GB compressed**, **~2.1T decompressed**.


## Downloading Dataset

> **NOTE:**
This tutorial demonstrates preprocessing for a **single local shard**: global-shard_01_of_10/local-shard_0_of_10.

```bash
python3 download.py \
  --token HF_TOKEN \
  --num_workers 32 \
  --path_to_save /data/dclm \
  --patterns global-shard_01_of_10/local-shard_0_of_10/**
```

**Parameters:**
- `--token` — Hugging Face token for authentication.
- `--num_workers` — Number of parallel downloads; higher is faster.
- `--path_to_save` — Target directory for saving the dataset.
- `--patterns` — Subset of dataset to download. Ignore this param to download the full dataset.


## Decompressing Dataset

> **NOTE:**
Dependencies: parallel and zstd may need to be installed:

```bash
apt update
apt install parallel
apt install zstd
```

After downloading, decompress `.zst` files to `.jsonl`:

```bash
python3 decompress.py \
  --path_to_save /data/dclm/decompressed \
  --source_dir /data/dclm/global-shard_01_of_10/local-shard_0_of_10 \
  --num_workers 32
```


## Merging Files

This merges all decompressed `.jsonl` files from `/data/dclm/decompressed` into single `.jsonl` file to avoid hundreds of small `.jsonl` files before the preprocessing stage.

```bash
python3 merge.py \
  --path_to_save /data/dclm/decompressed/merged.jsonl \
  --source_dir /data/dclm/decompressed \
  --remove_small_files
```


## Data Shuffling

Script shuffles merged `.jsonl` file from previous data preparation step.

```bash
python3 shuffle.py \
  --path_to_save /data/dclm/decompressed/shuffled.jsonl \
  --source_file /data/dclm/decompressed/merged.jsonl \
  --num_workers 16
```


## Preprocessing to bin/idx format

This step will convert the merged `.jsonl` files into a `bin/idx` format for training. It requires Megatron-LM to be installed:

```bash
# Install Megatron Core with required dependencies
pip install megatron-core
pip install --no-build-isolation transformer-engine[pytorch]

# Clone repository for examples
git clone https://github.com/NVIDIA/Megatron-LM.git
```

Run data preprocessing script:

```bash
python3 Megatron-LM/tools/preprocess_data.py \
  --input /data/dclm/decompressed/shuffled.jsonl \
  --output-prefix /data/dclm/preprocessed \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model meta-llama/Meta-Llama-3-8B \
  --log-interval 10000 \
  --workers 32 \
  --append-eod
```
