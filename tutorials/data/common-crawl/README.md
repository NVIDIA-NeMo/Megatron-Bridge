# Common Crawl Data Preprocessing Tutorial W/ NeMo Curator

This guide explains how to prepare the Common Crawl dataset for language model pretraining.

**Common Crawl** is a non-profit organization that maintains an open repository of web crawl data. The dataset consists of petabytes of raw web page data, metadata extracts, and text extracts collected monthly since 2008. Each crawl contains billions of web pages from across the internet, making it one of the largest openly available datasets for training language models.

Dataset source: [Common Crawl](https://commoncrawl.org/)

## Dataset overview

The Common Crawl dataset is organized by **snapshots**, with each snapshot representing a complete web crawl:

- **Main crawls**: General purpose web crawl covering diverse websites across the internet. Released approximately monthly (format: `CC-MAIN-YYYY-WW`, e.g., `CC-MAIN-2025-30`)
- **News crawls**: Focused crawl of news websites and articles, useful for domain-specific training on journalism and current events. Released monthly (format: `YYYY-MM`, e.g., `2025-08`)

Each `CC-MAIN` snapshot contains multiple WARC (Web ARChive) files, with each file approximately **~1 GB compressed**, alongside a `warc.paths.gz` file that lists all WARC files in that snapshot. A typical main crawl snapshot includes around **~80,000 WARC files**, totaling approximately ~70-100 TB compressed. Each WARC file contains raw HTTP response data (HTML pages, headers, metadata) that requires extraction and filtering for language model training.

## Requirements

Install NeMo Curator directly from the GitHub repository with the CPU text processing extension:

```bash
pip install "nemo-curator[text_cpu] @ git+https://github.com/NVIDIA-NeMo/Curator.git@main"
```

For more information about NeMo Curator, visit the [official repository](https://github.com/NVIDIA-NeMo/Curator).
For detailed installation instructions, see the [installation guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html).

## Usage

To **download, uncompress, filter, and tokenize** CommonCrawl data, simply run the `prepare_commoncrawl.py` script.
All stages of the pipeline are handled within a single script thanks to **NeMo Curator**, which provides modular processing stages for downloading, cleaning, and tokenizing large-scale text datasets.

You will need to specify a few configuration options when running the script:

* **`--output_dir`**
  Path where processed data will be stored.

  * Raw files will be placed in: `output_dir/raw`
  * Tokenized files will be placed in: `output_dir/tokens`

* **`--tokenizer-model`**
  The tokenizer model identifier from the Hugging Face Hub used to tokenize the dataset.

* **`--append-eod`**
  When set, appends an End-of-Document (EOD) token to every sample.

* **`--filter-data`**
  Enables NeMo Curator’s filtering stages to remove short documents, documents with excessive URLs, and documents with highly repetitive n-grams.

* **`--use-aws-to-download`**
  Enables downloading CommonCrawl snapshots from S3.

* **`--start-snapshot`** and **`--end-snapshot`**
  Specify the range of snapshots to download.

* **`--url-limit`**
  Maximum number of files to download. Each file is approximately **1 GB**.

#### Example

```bash
python3 prepare_commoncrawl.py \
    --output_dir $DATASETS_PATH/CommonCrawl \
    --tokenizer-model nvidia/NVIDIA-Nemotron-Nano-12B-v2 \
    --append-eod \
    --filter-data \
    --url-limit 5
```

---

When the script completes, it will automatically generate a **`dataset-prefixes.txt`** file in the output directory.
This file contains the dataset file prefixes required by **Megatron-LM** and **Megatron-Bridge** via the `--data-args-path` configuration.

For more details about the new `MegatronTokenizerWriter` stage, refer to the ["megatron-tokenizer" tutorial in NeMo Curator](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/text/megatron-tokenizer).
