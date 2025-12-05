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

Below are performance benchmarks for various large language models organized by release version. These results were obtained using performance recipes available [here](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance).

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures
- **System Configurations**: Results across different GPU systems (DGX-GB200, DGX-B200, DGX-H100)
- **Precision Options**: Performance comparisons between different precision modes (BF16, FP8, MXFP8)

---

## 25.11 NeMo Container

### Pre-Training Performance

#### System: DGX-GB300

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 37556 (36108) | 1933 (1858) |
| LLAMA3_70B | 64 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 1 (0) | 1 (2) | 1 (4) | 1 | 1 (5) | 1 | 1 (16) | 4440 (4346) | 1995 (1952) |
| LLAMA3.1_405B | 128 | FP8-CS (FP8-MX) | 64 | 1 | 8192 | 1 (0) | 2 (4) | 1 (8) | 1 (2) | 1 (8) | 1 | 1 (32) | 850 (638) | 2145 (1610) |
| DeepSeekV3 | 256 | FP8-MX | 2048 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 64 | 64 | 3943 | 1026 |
| GPT OSS 120B | 64 | BF16 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 64 | 2 | 18618 | 506 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 28934 | 666 |
| Qwen3_235B_a22B | 64 | FP8-MX | 1024 | 1 | 4096 | 0 | 2 | 1 | 1 | 1 | 64 | 32 | 5350 | 792 |


#### System: DGX-GB200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 31508 (29789) | 1622 (1533) |
| LLAMA3_70B | 64 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 1 (0) | 1 (2) | 1 (4) | 1 | 1 (5) | 1 | 1 (16) | 4312 (3617) | 1937 (1625) |
| LLAMA3.1_405B | 128 | FP8-CS (FP8-MX) | 64 | 1 | 8192 | 1 (0) | 2 (4) | 1 (8) | 1 (2) | 1 (8) | 1 | 1 (32) | 706 (563) | 1782 (1420) |
| DeepSeekV3 | 256 | FP8-MX | 2048 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 32 64 | 3653 | 949 |
| GPT OSS 120B | 64 | BF16 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 64 | 2 | 15754 | 428 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 23766 | 547 |
| Qwen3_235B_a22B | 64 | FP8-MX | 1024 | 1 | 4096 | 0 | 2 | 1 | 1 | 1 | 64 | 32 | 4366 | 646 |

#### System: DGX-B200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 30624 (29521) | 1576 (1519) |
| LLAMA3.1_405B | 128 | FP8-CS (FP8-MX) | 64 | 1 | 8192 | 0 | 4 | 8 | 2 | 8 | 1 | 32 | 661 (624) | 1667 (1576) |
| DeepSeekV3 | 256 | FP8-MX | 2048 | 1 | 4096 | 0 | 1 | 16 | 1 | 1 | 8 | 128 | 2139 | 557 |
| GPT OSS 120B | 64 | BF16 |  512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 2 | 8213 | 223 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 1 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 64 | 9299 | 214 |
| Qwen3_235B_a22B | 64 | FP8-MX | 1024 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 8 | 128 | 3269 | 484 |

#### System: DGX-H100

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS | 128 | 1 | 8192 | 1 | 1 | 1 | 1 | n/a | 1 | 16 | 14451 | 744 |
| LLAMA3_70B | 64 | FP8-CS | 128 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | 1 | 64 | 1602 | 719 |
| LLAMA3.1_405B | 1024 | FP8-CS | 512 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | 1 | 64 | 292 | 737 |
| GPT OSS 120B | 64 | BF16 | 512 | 4 | 4096 | 0 | 1 | 4 | 1 | 1 | 8 | 2 | 5630 | 153 |
| Qwen3_30B_a3B | 16 | FP8-CS | 512 | 2 | 4096 | 0 | 1 | 2 | 1 | 24 | 8 | 32 | 5275 | 121 |
| Qwen3_235B_a22B | 256 | FP8-CS | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 1575 | 1575 | 233 |

- The numbers in normal parentheses indicate the use of different quantization granularities: In case of Gb200 and B200 systems, 32×32 for both weights and activations. For H100 system, 128×128 for weights and 1×128 for activations, which match those used in the original DeepSeekV3 pre-training.
- In MoE trianing benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.
