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

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures
- **System Configurations**: Results across different GPU systems (DGX-GB300, DGX-GB200, DGX-B300, DGX-B200, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8, NVFP4)

---

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

## Archive

Performance summary for past releases can be found in the [archive](performance-summary-archive.md).