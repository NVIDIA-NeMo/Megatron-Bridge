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
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8)

---

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

## Archive

Performance summary for past releases can be found in the [archive](performance-summary-archive.md).