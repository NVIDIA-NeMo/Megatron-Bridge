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
- **System Configurations**: Results across different GPU systems (DGX-GB300, DGX-GB200, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8, NVFP4)

---

## 26.06 NeMo Container

### Pre-Training Performance

#### Model: LLAMA3.1_405B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | FP8 | 1536 | 1 | 8192 | 4 | 8 | 1 | 4 | n/a | 1024 | 2594 |
| DGX-GB300 | 256 | MXFP8 | 1536 | 1 | 8192 | 2 | 8 | 2 | 4 | n/a | 960 | 2401 |
| DGX-GB300 | 256 | NVFP4 | 1536 | 1 | 8192 | 4 | 8 | 1 | 4 | n/a | 1472 | 3693 |
| DGX-GB200 | 256 | FP8 | 1536 | 1 | 8192 | 4 | 16 | 1 | 4 | n/a | 864 | 2148 |
| DGX-GB200 | 256 | MXFP8 | 1536 | 1 | 8192 | 4 | 16 | 1 | 8 | n/a | 768 | 1971 |
| DGX-GB200 | 256 | NVFP4 | 1536 | 1 | 8192 | 4 | 16 | 1 | 8 | n/a | 1056 | 3047 |
| DGX-H100 | 1024 | FP8 | 1536 | 1 | 8192 | 8 | 8 | 2 | 8 | n/a | 328 | 822 |

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 2 | 1 | 8 | 32 | 6352 | 1650 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 4 | 1 | 4 | 64 | 5008 | 1301 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | BF16 | 1280 | 4 | 4096 | 1 | 1 | 1 | n/a | 64 | 20800 | 679 |
| DGX-GB200 | 64 | BF16 | 1280 | 4 | 4096 | 1 | 1 | 1 | n/a | 64 | 17728 | 579 |
| DGX-H100 | 64 | BF16 | 1280 | 1 | 4096 | 1 | 4 | 1 | n/a | 8 | 5824 | 191 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 1 | 1 | 1 | n/a | 8 | 45056 | 1037 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 1 | 1 | 1 | n/a | 8 | 40960 | 941 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 4096 | 1 | 1 | 1 | n/a | 16 | 8960 | 203 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 1 | 4 | 1 | 12 | 32 | 9040 | 1337 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 1 | 8 | 1 | 3 | 32 | 7216 | 1068 |

#### Model: Kimi_K2

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 2 | 4096 | 1 | 4 | 1 | 4 | 64 | 5360 | 1097 |

-  Muon optimizer was used for pre-training Kimi-K2.

#### Model: Nemotron_3_Nano

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 1 | 1 | 1 | n/a | 8 | 39936 | 881 |
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 1 | 1 | 1 | n/a | 8 | 31744 | 716 |
| DGX-H100 | 16 | FP8 | 1024 | 1 | 8192 | 1 | 1 | 1 | n/a | 8 | 14336 | 324 |

#### Model: Nemotron_3_Super

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | MXFP8 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 9984 | 847 |
| DGX-GB300 | 64 | NVFP4 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 9984 | 842 |
| DGX-GB200 | 64 | MXFP8 | 512 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 6912 | 589 |
| DGX-GB200 | 64 | NVFP4 | 512 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 7040 | 594 |

### SFT Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 32 | FP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 4864 | 2110 |
| DGX-GB300 | 32 | MXFP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 4352 | 1872 |
| DGX-GB200 | 32 | FP8 | 32 | 1 | 4096 | 1 | 8 | 1 | 10 | n/a | 3712 | 1613 |
| DGX-GB200 | 32 | MXFP8 | 32 | 1 | 4096 | 1 | 8 | 1 | 10 | n/a | 3584 | 1525 |
| DGX-H100 | 32 | FP8 | 32 | 1 | 4096 | 4 | 4 | 1 | 5 | n/a | 1664 | 710 |

### LoRA Performance

#### Model: LLAMA3_70B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | FP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 7680 | 2076 |
| DGX-GB300 | 8 | MXFP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 7680 | 2090 |
| DGX-GB200 | 8 | FP8 | 32 | 1 | 4096 | 1 | 2 | 1 | 20 | n/a | 6144 | 1753 |
| DGX-GB200 | 8 | MXFP8 | 32 | 1 | 4096 | 1 | 4 | 1 | 20 | n/a | 5632 | 1633 |
| DGX-H100 | 8 | FP8 | 32 | 1 | 4096 | 2 | 4 | 1 | 20 | n/a | 2560 | 738 |

[^moe-training-note]: In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## Archive

Performance summary for past releases can be found in the [archive](performance-summary-archive.md).
