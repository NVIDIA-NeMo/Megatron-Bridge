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

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures[^moe-training-note]
- **System Configurations**: Results across different GPU systems (DGX-GB300, DGX-GB200, DGX-B300, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8, NVFP4)

---

## 26.08 NeMo Container

### Pre-Training Performance

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 2 | 1 | 8 | 32 | 6288 | 1635 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 4 | 1 | 4 | 64 | 4912 | 1276 |

#### Model: DeepSeekV4 Flash

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 128 | MXFP8 | 2048 | 1 | 4096 | 1 | 1 | 1 | n/a | 64 | 8224 | 748 |
| DGX-GB200 | 128 | MXFP8 | 2048 | 1 | 4096 | 1 | 1 | 1 | n/a | 64 | 7776 | 706 |

#### Model: GPT OSS 120B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | MXFP8 | 1280 | 4 | 4096 | 1 | 1 | 1 | n/a | 16 | 33024 | 1077 |
| DGX-GB200 | 64 | MXFP8 | 1280 | 4 | 4096 | 1 | 1 | 1 | n/a | 64 | 28672 | 934 |

#### Model: Qwen3_30B_a3B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 8 | 4096 | 1 | 1 | 1 | n/a | 8 | 44544 | 1029 |
| DGX-GB200 | 8 | MXFP8 | 512 | 4 | 4096 | 1 | 1 | 1 | n/a | 8 | 39936 | 923 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 1 | 4 | 1 | 12 | 32 | 8816 | 1306 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 1 | 8 | 1 | 3 | 32 | 7280 | 1077 |

#### Model: Nemotron_3_Nano

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 1 | 1 | 1 | n/a | 8 | 39936 | 901 |

#### Model: Nemotron_3_5_Lightning

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 1 | 1 | 1 | n/a | 8 | 28672 | 807 |

#### Model: Nemotron_3_Super

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | MXFP8 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 9856 | 831 |
| DGX-GB300 | 64 | NVFP4 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 10240 | 864 |
| DGX-GB200 | 64 | MXFP8 | 512 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 7040 | 600 |
| DGX-GB200 | 64 | NVFP4 | 512 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 7168 | 610 |

#### Model: Nemotron_3_Ultra

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 256 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 3552 | 1281 |
| DGX-GB200 | 256 | MXFP8 | 256 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 2528 | 923 |

[^moe-training-note]: In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## Archive

Performance summary for past releases can be found in the [archive](performance-summary-archive.md).
