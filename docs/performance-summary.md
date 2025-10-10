# Performance

As part of the NVIDIA NeMo Framework, Megatron Bridge, provides optimal performance for training advanced generative AI models by incorporating the most recent training techniques, such as model parallelization, optimized attention mechanisms, and more, to achieve high training throughput.

This page provides performance benchmarks for large language models using Megatron-Bridge across different GPU systems and configurations.

## Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **FSDP**: Fully Sharded Data Parallel
  - FSDP = 1: use FSDP
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

```{contents}
:local:
:depth: 2
```

## Performance Summary for Large Language Models

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [here](https://github.com/NVIDIA/Megatron-Bridge/tree/main/scripts/performance).

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures
- **System Configurations**: Results across different GPU systems (DGX-GB200, DGX-B200, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8)

---

## 25.09 NeMo Container

### Pre-Training Performance

#### System: DGX-GB200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 31357 (29925) | 1614 (1540) |
| LLAMA3_70B | 64 | 128 | 2 | 8192 | 1 (0) | 1 (2) | 1 (4) | 1 | 1 (5) | 1 | 1 (16) | 3986 (3546) | 1791 (1593) |
| LLAMA3.1_405B | 128 | 64 | 1 | 8192 | 1 (0) | 2 (4) | 1 (8) | 1 (2) | 1 (8) | 1 | 1 (32) | 729 (578) | 1840 (1458) |
| DeepSeekV3 | 256 | 2048 | 1 | 4096 | 0 | 1 | 4 (8) | 1 | 4 (2) | 64 | 32 (64) | 3454 (2835) | 899 (738) |
| Qwen3_30B_a3B | 8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 22775 (23723) | 524 (546) |
| Qwen3_235B_a22B | 64 | 1024 | 1 | 4096 | 0 | 2 | 1 | 1 | 1 | 64 | 32 | 4452 (4416) | 659 (654) |

#### System: DGX-B200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 29994 (29388) | 1544 (1513) |
| LLAMA3.1_405B | 128 | 64 | 1 | 8192 | 0 | 4 | 8 | 2 | 8 | 1 | 32 | 664 (622) | 1676 (1569) |
| DeepSeekV3 | 256 | 2048 | 1 | 4096 | 0 | 1 | 16 | 1 | 1 | 8 | 128 | 2265 (2159) | 589 (562) |
| Qwen3_30B_a3B | 8 | 512 | 1 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 64 | 18066 | 416 |
| Qwen3_235B_a22B | 64 | 1024 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 8 | 128 | 4104 (4275) | 607 (633) |

#### System: DGX-H100

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 1 | 8192 | 1 | 1 | 1 | 1 | n/a | 1 | 16 | 14079 | 725 |
| LLAMA3_70B | 64 | 128 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | 1 | 64 | 1619 | 727 |
| LLAMA3.1_405B | 1024 | 512 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | 1 | 64 | 302 | 763 |
| DeepSeekV3 | 1024 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 64 | 128 | 1297 | 338 (330) |
| Qwen3_30B_a3B | 16 | 512 | 2 | 4096 | 0 | 1 | 2 | 1 | 24 | 8 | 32 | 10494 | 241 |
| Qwen3_235B_a22B | 256 | 2048 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 128 | 1204 | 178 |

- The numbers in parentheses for DeepSeekV3 indicate the use of different quantization granularities: 128×128 for weights and 1×128 for activations, which match those used in the original DeepSeekV3 pre-training.
