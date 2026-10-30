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

## 26.10 NeMo Container

### Pre-Training Performance

#### Model: DeepSeekV3

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 2 | 1 | 8 | 32 | 6432 | 1675 |
| DGX-GB200 | 256 | MXFP8 | 4096 | 1 | 4096 | 1 | 4 | 1 | 4 | 64 | 4896 | 1271 |

#### Model: DeepSeekV4 Flash

> **Note:** DeepSeek V4 requires checking out the Megatron-Core `dev` branch. See the [DeepSeek V4 reference directory](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/r0.6.0/examples/models/deepseek_v4) for setup instructions.

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 128 | MXFP8 | 2048 | 2 | 4096 | 1 | 1 | 1 | n/a | 32 | 8992 | 818 |
| DGX-GB200 | 128 | MXFP8 | 2048 | 1 | 4096 | 1 | 1 | 1 | n/a | 32 | 8096 | 737 |

#### Model: Qwen3_235B_a22B

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | MXFP8 | 8192 | 2 | 4096 | 1 | 4 | 1 | 12 | 32 | 9152 | 1355 |
| DGX-GB200 | 256 | MXFP8 | 8192 | 1 | 4096 | 1 | 8 | 1 | 3 | 32 | 7408 | 1096 |

#### Model: Nemotron_3_5_Lightning

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 8 | MXFP8 | 512 | 4 | 8192 | 1 | 1 | 1 | n/a | 8 | 37888 | 1058 |
| DGX-GB200 | 8 | MXFP8 | 512 | 2 | 8192 | 1 | 1 | 1 | n/a | 8 | 30720 | 853 |

#### Model: Nemotron_3_Super

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 64 | NVFP4 | 512 | 1 | 8192 | 1 | 1 | 1 | n/a | 16 | 11776 | 993 |
| DGX-GB200 | 64 | NVFP4 | 512 | 1 | 8192 | 2 | 1 | 1 | n/a | 16 | 8064 | 686 |

#### Model: Nemotron_3_Ultra

| System | #-GPUs | Precision | GBS | MBS | Sequence Length | TP | PP | CP | VP | EP | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|--------|--------|-----------|-----|-----|-----------------|----|----|----|----|----|-----------------------|-------------------------|
| DGX-GB300 | 256 | NVFP4 | 256 | 1 | 8192 | 1 | 1 | 1 | n/a | 64 | 4000 | 1440 |
| DGX-GB200 | 256 | NVFP4 | 256 | 1 | 8192 | 2 | 1 | 1 | n/a | 64 | 3200 | 1155 |

[^moe-training-note]: In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## Archive

Performance summary for past releases can be found in the [archive](performance-summary-archive.md).
