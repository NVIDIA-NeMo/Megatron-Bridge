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
