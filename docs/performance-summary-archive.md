# Performance Archive

NOTE: This page contains performance summaries for past releases. You can find the summary for the latest release at [performance-summary.md](performance-summary.md).

As part of the NVIDIA NeMo Framework, Megatron Bridge, provides optimal performance for training advanced generative AI models by incorporating the most recent training techniques, such as model parallelization, optimized attention mechanisms, and more, to achieve high training throughput.

This page provides performance benchmarks for large language models using Megatron-Bridge across different GPU systems and configurations.

## Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **FSDP**: Fully Sharded Data Parallel
  - FSDP > 0: use FSDP with sharding group size = #GPUs / (TP × PP)
  - FSDP = 0: use DDP (Distributed Data Parallel)
- **TP**: Tensor Parallel Size
- **PP**: Pipeline Parallel Size
- **CP**: Context Parallel Size
- **VP**: Virtual Pipeline Parallel Size
- **EP**: Expert Parallel Size
- **GA**: Number of Gradient Accumulations

## Performance Metrics

Performance is measured using:

- **Tokens/sec/GPU**: Throughput per GPU
- **Model TFLOP/sec/GPU**: Model floating-point operations per second per GPU

## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [here](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance).

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures[^moe-training-note]
- **System Configurations**: Results across different GPU systems (DGX-GB300, DGX-GB200, DGX-B300, DGX-B200, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8, NVFP4)

---

## 26.06 NeMo Container

### Pre-Training Performance

#### Model: LLAMA3.1_405B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | FP8 | 1536 | 1 | 8192 | 4 | 8 | 1 | 4 | n/a | 1048 | 2646 |
| DGX-GB300 | 256 | MXFP8 | 1536 | 1 | 8192 | 2 | 8 | 2 | 4 | n/a | 952 | 2403 |
| DGX-GB300 | 256 | NVFP4 | 1536 | 1 | 8192 | 4 | 8 | 1 | 4 | n/a | 1413 | 3575 |
| DGX-GB200 | 256 | FP8 | 1536 | 1 | 8192 | 4 | 16 | 1 | 4 | n/a | 843 | 2129 |
| DGX-GB200 | 256 | MXFP8 | 1536 | 1 | 8192 | 4 | 16 | 1 | 8 | n/a | 783 | 1976 |
| DGX-GB200 | 256 | NVFP4 | 1536 | 1 | 8192 | 4 | 16 | 1 | 8 | n/a | 1166 | 2944 |
| DGX-H100 | 1024 | FP8 | 1536 | 1 | 8192 | 8 | 8 | 2 | 8 | n/a | 326 | 822 |

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 2 | 1 | 8 | 32 | 6338 | 1648 |
| DGX-GB300 | 256 | MXFP8 | 15360 | 1 | 4096 | 1 | 2 | 1 | 8 | 32 | 6422 | 1670 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 4 | 1 | 4 | 64 | 4969 | 1292 |
| DGX-B300 | 256 | MXFP8 | 4096 | 2 | 4096 | 1 | 8 | 1 | n/a | 8 | 3541 | 920 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | MXFP8 | 1280 | 4 | 4096 | 1 | 1 | 1 | n/a | 16 | 33166 | 1081 |
| DGX-GB200 | 64 | MXFP8 | 1280 | 4 | 4096 | 1 | 1 | 1 | n/a | 64 | 28947 | 943 |
| DGX-B300 | 64 | MXFP8 | 1280 | 4 | 4096 | 1 | 1 | 1 | n/a | 8 | 18534 | 604 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 1 | 1 | 1 | n/a | 8 | 45275 | 1041 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 1 | 1 | 1 | n/a | 8 | 40706 | 936 |
| DGX-B300 | 8 | MXFP8 | 512 | 8 | 4096 | 1 | 1 | 1 | n/a | 8 | 40769 | 938 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 4096 | 1 | 1 | 1 | n/a | 16 | 8826 | 203 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 1 | 4 | 1 | 12 | 32 | 9015 | 1335 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 1 | 8 | 1 | 3 | 32 | 7376 | 1092 |

#### Model: Kimi_K2

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 1 | 4 | 1 | 4 | 64 | 5372 | 1099 |

-  Muon optimizer was used for pre-training Kimi-K2.

#### Model: Nemotron_3_Nano

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 1 | 1 | 1 | n/a | 8 | 39749 | 885 |
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 1 | 1 | 1 | n/a | 8 | 33522 | 747 |
| DGX-B300 | 8 | MXFP8 | 512 | 4 | 8192 | 1 | 1 | 1 | n/a | 8 | 37316 | 831 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 8192 | 1 | 1 | 1 | n/a | 8 | 14719 | 328 |

#### Model: Nemotron_3_Super

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | MXFP8 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 9652 | 817 |
| DGX-GB300 | 64 | NVFP4 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 9900 | 839 |
| DGX-GB200 | 64 | MXFP8 | 512 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 6742 | 571 |
| DGX-GB200 | 64 | NVFP4 | 512 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 6928 | 587 |
| DGX-B300 | 64 | MXFP8 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 8 | 7867 | 667 |
| DGX-B300 | 64 | NVFP4 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 8 | 8131 | 689 |

### SFT Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 32 | FP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 4819 | 2083 |
| DGX-GB300 | 32 | MXFP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 4312 | 1877 |
| DGX-GB200 | 32 | FP8 | 32 | 1 | 4096 | 1 | 8 | 1 | 10 | n/a | 3864 | 1671 |
| DGX-GB200 | 32 | MXFP8 | 32 | 1 | 4096 | 1 | 8 | 1 | 10 | n/a | 3593 | 1553 |
| DGX-H100 | 32 | FP8 | 32 | 1 | 4096 | 4 | 4 | 1 | 5 | n/a | 1638 | 710 |

### LoRA Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | FP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 7481 | 2086 |
| DGX-GB300 | 8 | MXFP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 7447 | 2072 |
| DGX-GB200 | 8 | FP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 6206 | 1731 |
| DGX-GB200 | 8 | MXFP8 | 32 | 1 | 4096 | 1 | 4 | 1 | 20 | n/a | 5958 | 1663 |
| DGX-H100 | 8 | FP8 | 32 | 1 | 4096 | 2 | 4 | 1 | 20 | n/a | 2643 | 735 |

## 26.04.01 NeMo Container

### Pre-Training Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 5248 | 2348 |
| DGX-GB300 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 4864 | 2186 |
| DGX-GB300 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 7296 | 3253 |
| DGX-GB200 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 4224 | 1892 |
| DGX-GB200 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 3712 | 1664 |
| DGX-GB200 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 4864 | 2202 |
| DGX-H100 | 64 | FP8 | 256 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | n/a | 1664 | 731 |

#### Model: LLAMA3.1_405B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1024 | 2617 |
| DGX-GB300 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | n/a | 960 | 2453 |
| DGX-GB300 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1440 | 3653 |
| DGX-GB200 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | n/a | 864 | 2144 |
| DGX-GB200 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 800 | 1994 |
| DGX-GB200 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 1184 | 2960 |
| DGX-H100 | 1024 | FP8 | 1536 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | n/a | 328 | 827 |

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 2 | 1 | 8 | 32 | 4992 | 1298 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 4256 | 1106 |
| DGX-B300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 3456 | 898 |
| DGX-B200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 32 | 3328 | 864 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 19200 | 523 |
| DGX-GB200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 16640 | 452 |
| DGX-B300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 15232 | 414 |
| DGX-B200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 13568 | 369 |
| DGX-H100 | 64 | BF16 | 1280 | 1 | 4096 | 0 | 1 | 4 | 1 | n/a | 8 | 5824 | 158 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 31744 | 729 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 26112 | 599 |
| DGX-B300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 30720 | 704 |
| DGX-B200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 27136 | 619 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 4096 | 0 | 1 | 1 | 1 | n/a | 16 | 8960 | 206 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | 12 | 32 | 6944 | 1029 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 3 | 32 | 5680 | 840 |
| DGX-B300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 5936 | 878 |
| DGX-B200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 3776 | 560 |
| DGX-H100 | 256 | FP8 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 1712 | 253 |

#### Model: Kimi_K2

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 5328 | 1088 |

-  Muon optimizer was used for pre-training Kimi-K2.

#### Model: Nemotron_3_Nano

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 37888 | 845 |
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 32768 | 725 |
| DGX-B300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 35840 | 794 |
| DGX-B200 | 8 | MXFP8 | 512 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 32768 | 726 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 14336 | 321 |

#### Model: Nemotron_3_Super

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 9344 | 795 |
| DGX-GB300 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 9600 | 817 |
| DGX-GB200 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 6656 | 564 |
| DGX-GB200 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 6784 | 574 |
| DGX-B300 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 7296 | 623 |
| DGX-B300 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 7424 | 634 |
| DGX-B200 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 6400 | 542 |
| DGX-B200 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 5632 | 475[^nemotron-3-super-b200-nvfp4-note] |

[^moe-training-note]: In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.
[^nemotron-3-super-b200-nvfp4-note]: Mapping used for MXFP8 precision could not fit for  NVFP4 precision for this model. We expect to achieve better performance for NVFP4 precision in future when NVFP4 param gather is supported.

## 26.04 NeMo Container

### Pre-Training Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 7397 | 3321 |
| DGX-GB200 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 4898 | 2201 |
| DGX-GB300 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 4742 | 2131 |
| DGX-GB200 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 3828 | 1721 |
| DGX-GB300 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 5243 | 2355 |
| DGX-GB200 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 4143 | 1861 |
| DGX-H100 | 64 | FP8 | 256 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | n/a | 1633 | 734 |

#### Model: LLAMA3.1_405B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1424 | 3594 |
| DGX-GB200 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 1136 | 2866 |
| DGX-GB300 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | n/a | 968 | 2443 |
| DGX-GB200 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 793 | 2002 |
| DGX-GB300 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1036 | 2615 |
| DGX-GB200 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | n/a | 842 | 2126 |

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 2 | 1 | 8 | 32 | 4965 | 1290 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 4193 | 1090 |
| DGX-B300 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 3131 | 814 |
| DGX-B200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 2815 | 732 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 19275 | 524 |
| DGX-GB200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 15938 | 433 |
| DGX-B300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 15114 | 410 |
| DGX-B200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 13045 | 355 |
| DGX-H100 | 64 | BF16 | 1280 | 1 | 4096 | 0 | 1 | 4 | 1 | n/a | 8 | 5810 | 158 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 31470 | 724 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 26032 | 599 |
| DGX-B300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 30553 | 703 |
| DGX-B200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 26859 | 618 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 4096 | 0 | 1 | 1 | 1 | n/a | 16 | 8901 | 205 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | 12 | 32 | 6994 | 1035 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 3 | 32 | 5672 | 840 |
| DGX-B300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 4865 | 720 |
| DGX-H100 | 256 | FP8 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 1686 | 250 |

#### Model: Nemotron_3_Nano

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 38102 | 849 |
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 32727 | 729 |
| DGX-B300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 35282 | 786 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 14507 | 323 |

#### Model: Nemotron_3_Super

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 9623 | 815 |
| DGX-GB200 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 6777 | 574 |
| DGX-B300 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 7047 | 597 |
| DGX-B200 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 5606 | 475 |
| DGX-GB300 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 9243 | 783 |
| DGX-GB200 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 6674 | 565 |
| DGX-B300 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 7009 | 594 |
| DGX-B200 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 6302 | 534 |

#### Model: Kimi_K2

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 5332 | 1090 |

-  Muon optimizer was used for pre-training Kimi-K2.

- In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## 26.02.01 NeMo Container

### Pre-Training Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | NVFP4 | 256 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | n/a | 7002 | 3147 |
| DGX-GB200 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 4557 | 2047 |
| DGX-GB300 | 64 | MXFP8 | 256 | 2 | 8192 | 0 | 1 | 4 | 1 | n/a | n/a | 4798 | 2157 |
| DGX-GB200 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 3837 | 1724 |
| DGX-GB300 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 5243 | 2353 |
| DGX-GB200 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 4357 | 1956 |
| DGX-H100 | 64 | FP8 | 256 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | n/a | 1639 | 736 |

#### Model: LLAMA3.1_405B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1358 | 3428 |
| DGX-GB200 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | n/a | 1083 | 2734 |
| DGX-GB300 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | n/a | 949 | 2394 |
| DGX-GB200 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 775 | 1957 |
| DGX-GB300 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | n/a | 1024 | 2585 |
| DGX-GB200 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | n/a | 818 | 2063 |

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 2 | 1 | 8 | 32 | 4691 | 1219 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 4021 | 1046 |
| DGX-B300 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 3099 | 806 |
| DGX-B200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 2790 | 725 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 19366 | 526 |
| DGX-GB200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 15754 | 428 |
| DGX-B300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 15031 | 412 |
| DGX-B200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 13722 | 373 |
| DGX-H100 | 64 | BF16 | 1280 | 1 | 4096 | 0 | 1 | 4 | 1 | n/a | 8 | 5984 | 163 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 30411 | 700 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 26373 | 607 |
| DGX-B300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 29454 | 678 |
| DGX-B200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 26695 | 614 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 4096 | 0 | 1 | 2 | 1 | 12 | 8 | 9058 | 208 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | n/a | 32 | 6583 | 974 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | n/a | 32 | 5530 | 819 |
| DGX-B300 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 4 | 8 | 2644 | 391 |
| DGX-H100 | 256 | FP8 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 1611 | 238 |

#### Model: Nemotron_3_Nano

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 37664 | 839 |
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 33934 | 756 |
| DGX-B300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 35861 | 798 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 14890 | 331 |

#### Model: Kimi_K2

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 5072 | 1037 |

-  Muon optimizer was used for pre-training Kimi-K2.

- In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## 26.02 NeMo Container

### Pre-Training Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 6798 | 3056 |
| DGX-GB200 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 4458 | 2004 |
| DGX-GB300 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 4596 | 2064 |
| DGX-GB200 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 3613 | 1623 |
| DGX-GB300 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 5003 | 2248 |
| DGX-GB200 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 4040 | 1815 |
| DGX-H100 | 64 | FP8 | 256 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | n/a | 1621 | 728 |

#### Model: LLAMA3.1_405B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1333 | 3365 |
| DGX-GB200 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 1076 | 2716 |
| DGX-GB300 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | n/a | 931 | 2349 |
| DGX-GB200 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 786 | 1983 |
| DGX-GB300 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 988 | 2495 |
| DGX-GB200 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | n/a | 793 | 2004 |
| DGX-H100 | 1024 | FP8 | 1536 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | n/a | 311 | 784 |

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 2 | 1 | 8 | 32 | 4612 | 1199 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 3955 | 1028 |
| DGX-B300 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 2983 | 776 |
| DGX-B200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 2689 | 699 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 19412 | 527 |
| DGX-GB200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 15784 | 428 |
| DGX-B300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 8359 | 228 |
| DGX-B200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 8047 | 219 |
| DGX-H100 | 64 | BF16 | 1280 | 1 | 4096 | 0 | 1 | 4 | 1 | n/a | 8 | 5993 | 163 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 30376 | 699 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 26084 | 600 |
| DGX-B300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 29521 | 679 |
| DGX-B200 | 8 | MXFP8 | 512 | 1 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 9691 | 223 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 4096 | 0 | 1 | 2 | 1 | 12 | 8 | 5113 | 118 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | n/a | 32 | 6583 | 974 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | n/a | 32 | 5448 | 806 |
| DGX-B300 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 4 | 8 | 2691 | 399 |
| DGX-B200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 3805 | 563 |
| DGX-H100 | 256 | FP8 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 1633 | 242 |

- In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## 25.11 NeMo Container

### Pre-Training Performance

#### System: DGX-GB300

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 37556 (36108) | 1933 (1858) |
| LLAMA3_70B | 64 | FP8-CS | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | 1 | 1 | 2 | 4520 | 2030 |
| LLAMA3.1_405B | 256 | FP8-CS | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | 1 | 192 | 999 | 2522 |
| DeepSeekV3 (w/o MTP) | 256 | BF16 | 4096 | 1 | 4096 | 0 | 1 | 2 | 1 | 4 | 32 | 32 | 3848 | 961 |
| DeepSeekV3 (w/o MTP)| 256 | FP8-MX | 4096 | 1 | 4096 | 0 | 1 | 2 | 1 | 4 | 32 | 32 | 4357 | 1088 |
| GPT OSS 120B | 64 | BF16 | 1280 | 2 | 8192 | 0 | 1 | 1 | 1 | 1 | 64 | 10 | 18347 | 565 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 28934 | 666 |
| Qwen3_235B_a22B | 256 | BF16 | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | 12 | 16 | 32 | 6131 | 907 |

#### System: DGX-GB200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 31508 (29789) | 1622 (1533) |
| LLAMA3_70B | 64 | FP8-CS | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | 1 | 1 | 2 | 4312 | 1937 |
| LLAMA3.1_405B | 256 | FP8-CS | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | 1 | 384 | 813 | 2053 |
| DeepSeekV3 (w/o MTP) | 256 | BF16 | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 64 | 3139 | 782 |
| DeepSeekV3 (w/o MTP) | 256 | FP8-MX | 4096 | 1 | 4096 | 0 | 1 | 8 | 1 | 4 | 32 | 128 | 4018 | 1003 |
| GPT OSS 120B | 64 | BF16 | 1280 | 1 | 8192 | 0 | 1 | 1 | 1 | 1 | 64 | 20 | 15876 | 488 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 23766 | 547 |
| Qwen3_235B_a22B | 256 | BF16 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 3 | 32 | 256 | 4916 | 728 |

#### System: DGX-B200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 30624 (29521) | 1576 (1519) |
| LLAMA3.1_405B | 128 | FP8-CS (FP8-MX) | 64 | 1 | 8192 | 0 | 4 | 8 | 2 | 8 | 1 | 32 | 661 (624) | 1667 (1576) |
| DeepSeekV3 (w/ MTP) | 256 | FP8-MX | 2048 | 1 | 4096 | 0 | 1 | 16 | 1 | 1 | 8 | 128 | 2139 | 557 |
| GPT OSS 120B | 64 | BF16 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 2 | 8213 | 223 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 1 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 64 | 9299 | 214 |
| Qwen3_235B_a22B | 64 | FP8-MX | 1024 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 8 | 128 | 3269 | 484 |

#### System: DGX-H100

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS | 128 | 1 | 8192 | 8 | 1 | 1 | 1 | n/a | 1 | 16 | 14451 | 744 |
| LLAMA3_70B | 64 | FP8-CS | 128 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | 1 | 64 | 1602 | 719 |
| LLAMA3.1_405B | 1024 | FP8-CS | 512 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | 1 | 64 | 292 | 737 |
| GPT OSS 120B | 64 | BF16 | 512 | 4 | 4096 | 0 | 1 | 4 | 1 | 1 | 8 | 2 | 5630 | 153 |
| Qwen3_30B_a3B | 16 | FP8-CS | 512 | 2 | 4096 | 0 | 1 | 2 | 1 | 24 | 8 | 32 | 5275 | 121 |
| Qwen3_235B_a22B | 256 | FP8-CS | 2048 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 128 | 1575 | 233 |

- The numbers in normal parentheses indicate the use of different quantization granularities: In case of GB200 and B200 systems, 32×32 for both weights and activations. For H100 system, 128×128 for weights and 1×128 for activations, which match those used in the original DeepSeekV3 pre-training.
- In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## 25.09 NeMo Container

### Pre-Training Performance

#### System: DGX-GB200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 31357 (29925) | 1614 (1540) |
| LLAMA3_70B | 64 | 128 | 2 | 8192 | 64 (0) | 1 (2) | 1 (4) | 1 | 1 (5) | 1 | 1 (16) | 3986 (3546) | 1791 (1593) |
| LLAMA3.1_405B | 128 | 64 | 1 | 8192 | 64 (0) | 2 (4) | 1 (8) | 1 (2) | 1 (8) | 1 | 1 (32) | 729 (578) | 1840 (1458) |
| DeepSeekV3 (tokendrop) | 256 | 2048 | 1 | 4096 | 0 | 1 | 4 (8) | 1 | 4 (2) | 64 | 32 (64) | 3454 (2835) | 899 (738) |
| Qwen3_30B_a3B (tokendrop) | 8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 22775 (23723) | 524 (546) |
| Qwen3_235B_a22B (tokendrop) | 64 | 1024 | 1 | 4096 | 0 | 2 | 1 | 1 | 1 | 64 | 32 | 4452 (4416) | 659 (654) |

#### System: DGX-B200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 29994 (29388) | 1544 (1513) |
| LLAMA3.1_405B | 128 | 64 | 1 | 8192 | 0 | 4 | 8 | 2 | 8 | 1 | 32 | 664 (622) | 1676 (1569) |
| DeepSeekV3 (tokendrop) | 256 | 2048 | 1 | 4096 | 0 | 1 | 16 | 1 | 1 | 8 | 128 | 2265 (2159) | 589 (562) |
| Qwen3_30B_a3B (tokendrop) | 8 | 512 | 1 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 64 | 18066 | 416 |
| Qwen3_235B_a22B (tokendrop) | 64 | 1024 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 8 | 128 | 4104 (4275) | 607 (633) |

#### System: DGX-H100

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 1 | 8192 | 8 | 1 | 1 | 1 | n/a | 1 | 16 | 14079 | 725 |
| LLAMA3_70B | 64 | 128 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | 1 | 64 | 1619 | 727 |
| LLAMA3.1_405B | 1024 | 512 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | 1 | 64 | 302 | 763 |
| DeepSeekV3 (dropless) | 1024 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 64 | 128 | 1297 | 338 (330) |
| Qwen3_30B_a3B (tokendrop) | 16 | 512 | 2 | 4096 | 0 | 1 | 2 | 1 | 24 | 8 | 32 | 10494 | 241 |
| Qwen3_235B_a22B (tokendrop) | 256 | 2048 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 128 | 1204 | 178 |

- The numbers in parentheses indicate the use of different quantization granularities: In case of GB200 and B200 systems, 32×32 for both weights and activations. For H100 system, 128×128 for weights and 1×128 for activations, which match those used in the original DeepSeekV3 pre-training.
- In token-dropless MoE training benchmarks, we force-balance the token distribution among experts.
