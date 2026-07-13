# Performance

As part of the NVIDIA NeMo Framework, Megatron Bridge, provides optimal performance for training advanced generative AI models by incorporating the most recent training techniques, such as model parallelization, optimized attention mechanisms, and more, to achieve high training throughput.

This page provides performance benchmarks for large language models using Megatron-Bridge across different GPU systems and configurations.

## Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
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

- **Pre-training, SFT, and LoRA Performance**: Throughput metrics for various model sizes and architectures[^moe-training-note]
- **System Configurations**: Results across different GPU systems (DGX-GB300, DGX-GB200, DGX-B300, DGX-H100)
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

[^moe-training-note]: In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## Archive

Performance summary for past releases can be found in the [archive](performance-summary-archive.md).
