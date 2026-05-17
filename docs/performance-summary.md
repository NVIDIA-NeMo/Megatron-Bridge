# Performance

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

Below are performance benchmarks for various large language models. These results were obtained using performance recipes available [here](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance).

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures[^moe-training-note]
- **System Configurations**: Results across different GPU systems (DGX-GB300, DGX-GB200, DGX-B300, DGX-B200, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8, NVFP4)

---

## 26.04.01 NeMo Container

### Pre-Training Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 5376 | 2425 |
| DGX-GB300 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 4736 | 2156 |
| DGX-GB300 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 7296 | 3284 |
| DGX-GB200 | 64 | FP8 | 256 | 2 | 8192 | 64 | 1 | 1 | 1 | n/a | n/a | 4352 | 1970 |
| DGX-GB200 | 64 | MXFP8 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 3840 | 1699 |
| DGX-GB200 | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 4864 | 2209 |
| DGX-H100 | 64 | FP8 | 256 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | n/a | 1664 | 735 |

#### Model: LLAMA3.1_405B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1024 | 2599 |
| DGX-GB300 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | n/a | 960 | 2442 |
| DGX-GB300 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 1440 | 3617 |
| DGX-GB200 | 256 | FP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | n/a | 832 | 2065 |
| DGX-GB200 | 256 | MXFP8 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 800 | 2004 |
| DGX-GB200 | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 1184 | 3005 |
| DGX-H100 | 1024 | FP8 | 1536 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | n/a | 328 | 827 |

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 2 | 1 | 8 | 32 | 4976 | 1294 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 4256 | 1105 |
| DGX-B300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 3440 | 895 |
| DGX-B200 | 256 | MXFP8 | 4096 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 32 | 3328 | 864 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 19328 | 525 |
| DGX-GB200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 16640 | 451 |
| DGX-B300 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 15232 | 414 |
| DGX-B200 | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 13568 | 369 |
| DGX-H100 | 64 | BF16 | 1280 | 1 | 4096 | 0 | 1 | 4 | 1 | n/a | 8 | 5824 | 159 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 31232 | 723 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 26112 | 601 |
| DGX-B300 | 8 | MXFP8 | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 30720 | 704 |
| DGX-B200 | 8 | MXFP8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 27136 | 619 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 4096 | 0 | 1 | 1 | 1 | n/a | 16 | 8960 | 204 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | 12 | 32 | 6992 | 1035 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 3 | 32 | 5696 | 843 |
| DGX-B300 | 256 | MXFP8 | 8192 | 2 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 5984 | 885 |
| DGX-B200 | 256 | MXFP8 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 3776 | 560 |
| DGX-H100 | 256 | FP8 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 1696 | 252 |

#### Model: Kimi_K2

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 5344 | 1092 |

-  Muon optimizer was used for pre-training Kimi-K2.

#### Model: Nemotron_3_Nano

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 37888 | 849 |
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 32768 | 727 |
| DGX-B300 | 8 | MXFP8 | 512 | 4 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 35840 | 792 |
| DGX-B200 | 8 | MXFP8 | 512 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 32768 | 726 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 14336 | 323 |

#### Model: Nemotron_3_Super

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 9344 | 791 |
| DGX-GB300 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 9600 | 818 |
| DGX-GB200 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 6656 | 564 |
| DGX-GB200 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 6784 | 574 |
| DGX-B300 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 7424 | 628 |
| DGX-B300 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 8 | 7552 | 641 |
| DGX-B200 | 64 | MXFP8 | 512 | 1 | 8192 | 0 | 1 | 1 | 1 | n/a | 64 | 6400 | 542 |
| DGX-B200 | 64 | NVFP4 | 512 | 1 | 8192 | 0 | 2 | 1 | 1 | n/a | 64 | 5632 | 475[^nemotron-3-super-b200-nvfp4-note] |

[^moe-training-note]: In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.
[^nemotron-3-super-b200-nvfp4-note]: Mapping used for MXFP8 precision could not fit for  NVFP4 precision for this model. We expect to achieve better performance for NVFP4 precision in future when NVFP4 param gather is supported.

## Archive

Performance summary for past releases can be found in the [archive](performance-summary-archive.md).
