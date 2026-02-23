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

## 26.02 NeMo Container

### Pre-Training Performance

#### System: DGX-GB300

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_70B | 64 | FP8-CS | 256 | 2 | 8192 | 1 | 1 | 1 | 1 | n/a | n/a | 2 | 5003 | 2248 |
| LLAMA3_70B | 64 | FP8-MX | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 16 | 4596 | 2064 |
| LLAMA3_70B | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 1 | 4 | 1 | 5 | n/a | 16 | 6798 | 3056 |
| LLAMA3.1_405B | 256 | FP8-CS | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 192 | 988 | 2495 |
| LLAMA3.1_405B | 256 | FP8-MX | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | n/a | 192 | 931 | 2349 |
| LLAMA3.1_405B | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 8 | 1 | 4 | n/a | 192 | 1347 | 3401 |
| DeepSeekV3 | 256 | FP8-MX | 4096 | 2 | 4096 | 0 | 1 | 2 | 1 | 8 | 32 | 16 | 4612 | 1199 |
| GPT OSS 120B | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 5 | 19412 | 527 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 8 | 30376 | 699 |
| Qwen3_235B_a22B | 256 | FP8-MX | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | n/a | 32 | 64 | 6583 | 974 |

#### System: DGX-GB200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_70B | 64 | FP8-CS | 256 | 2 | 8192 | 1 | 1 | 1 | 1 | n/a | n/a | 2 | 4040 | 1815 |
| LLAMA3_70B | 64 | FP8-MX | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 32 | 3887 | 1746 |
| LLAMA3_70B | 64 | NVFP4 | 256 | 1 | 8192 | 0 | 2 | 4 | 1 | 5 | n/a | 32 | 5041 | 2264 |
| LLAMA3.1_405B | 256 | FP8-CS | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | n/a | 384 | 793 | 2004 |
| LLAMA3.1_405B | 256 | FP8-MX | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 384 | 786 | 1983 |
| LLAMA3.1_405B | 256 | NVFP4 | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 8 | n/a | 384 | 1145 | 2891 |
| DeepSeekV3 | 256 | FP8-MX | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 64 | 3955 | 1028 |
| GPT OSS 120B | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 5 | 15784 | 428 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 16 | 26084 | 600 |
| Qwen3_235B_a22B | 256 | FP8-MX | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | n/a | 32 | 256 | 5448 | 806 |

#### System: DGX-B300

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| DeepSeekV3 | 256 | FP8-MX | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 256 | 2983 | 776 |
| GPT OSS 120B | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 5 | 8359 | 228 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 8 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 8 | 29521 | 679 |
| Qwen3_235B_a22B | 256 | FP8-MX | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 4 | 8 | 256 | 2691 | 399 |

#### System: DGX-B200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| DeepSeekV3 | 256 | FP8-MX | 4096 | 1 | 4096 | 0 | 1 | 16 | 1 | n/a | 8 | 256 | 2689 | 699 |
| GPT OSS 120B | 64 | BF16 | 1280 | 4 | 4096 | 0 | 1 | 1 | 1 | n/a | 64 | 5 | 8047 | 219 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 1 | 4096 | 0 | 1 | 1 | 1 | n/a | 8 | 64 | 9691 | 223 |
| Qwen3_235B_a22B | 256 | FP8-MX | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | n/a | 8 | 256 | 3805 | 563 |

#### System: DGX-H100

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_70B | 64 | FP8-CS | 256 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | n/a | 128 | 1621 | 728 |
| LLAMA3.1_405B | 1024 | FP8-CS | 1536 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | n/a | 192 | 311 | 784 |
| GPT OSS 120B | 64 | BF16 | 1280 | 1 | 4096 | 0 | 1 | 4 | 1 | n/a | 8 | 80 | 5993 | 163 |
| Qwen3_30B_a3B | 16 | FP8-CS | 1024 | 1 | 4096 | 0 | 1 | 2 | 1 | 12 | 8 | 128 | 2557 | 118 |
| Qwen3_235B_a22B | 256 | FP8-CS | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 512 | 1633 | 242 |

- In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## 25.11 NeMo Container

### Pre-Training Performance

#### System: DGX-GB300

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 37556 (36108) | 1933 (1858) |
| LLAMA3_70B | 64 | FP8-CS | 256 | 2 | 8192 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 4520 | 2030 |
| LLAMA3.1_405B | 256 | FP8-CS | 1536 | 1 | 8192 | 0 | 2 | 8 | 2 | 4 | 1 | 192 | 999 | 2522 |
| DeepSeekV3 (w/o MTP) | 256 | BF16 | 4096 | 1 | 4096 | 0 | 1 | 2 | 1 | 4 | 32 | 32 | 3848 | 961 |
| DeepSeekV3 (w/o MTP)| 256 | FP8-MX | 4096 | 1 | 4096 | 0 | 1 | 2 | 1 | 4 | 32 | 32 | 4357 | 1088 |
| GPT OSS 120B | 64 | BF16 | 1280 | 2 | 8192 | 0 | 1 | 1 | 1 | 1 | 64 | 10 | 18347 | 565 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 28934 | 666 |
| Qwen3_235B_a22B | 256 | BF16 | 8192 | 2 | 4096 | 0 | 1 | 4 | 1 | 12 | 16 | 32 | 6131 | 907 |

#### System: DGX-GB200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 31508 (29789) | 1622 (1533) |
| LLAMA3_70B | 64 | FP8-CS | 256 | 2 | 8192 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 4312 | 1937 |
| LLAMA3.1_405B | 256 | FP8-CS | 1536 | 1 | 8192 | 0 | 4 | 16 | 1 | 4 | 1 | 384 | 813 | 2053 |
| DeepSeekV3 (w/o MTP) | 256 | BF16 | 4096 | 1 | 4096 | 0 | 1 | 4 | 1 | 4 | 64 | 64 | 3139 | 782 |
| DeepSeekV3 (w/o MTP) | 256 | FP8-MX | 4096 | 1 | 4096 | 0 | 1 | 8 | 1 | 4 | 32 | 128 | 4018 | 1003 |
| GPT OSS 120B | 64 | BF16 | 1280 | 1 | 8192 | 0 | 1 | 1 | 1 | 1 | 64 | 20 | 15876 | 488 |
| Qwen3_30B_a3B | 8 | FP8-MX | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 23766 | 547 |
| Qwen3_235B_a22B | 256 | BF16 | 8192 | 1 | 4096 | 0 | 1 | 8 | 1 | 3 | 32 | 256 | 4916 | 728 |

#### System: DGX-B200

| Model | #-GPUs | Precision | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | FP8-CS (FP8-MX) | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 30624 (29521) | 1576 (1519) |
| LLAMA3.1_405B | 128 | FP8-CS (FP8-MX) | 64 | 1 | 8192 | 0 | 4 | 8 | 2 | 8 | 1 | 32 | 661 (624) | 1667 (1576) |
| DeepSeekV3 (w/ MTP) | 256 | FP8-MX | 2048 | 1 | 4096 | 0 | 1 | 16 | 1 | 1 | 8 | 128 | 2139 | 557 |
| GPT OSS 120B | 64 | BF16 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 2 | 8213 | 223 |
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
| Qwen3_235B_a22B | 256 | FP8-CS | 2048 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 128 | 1575 | 233 |

- The numbers in normal parentheses indicate the use of different quantization granularities: In case of GB200 and B200 systems, 32×32 for both weights and activations. For H100 system, 128×128 for weights and 1×128 for activations, which match those used in the original DeepSeekV3 pre-training.
- In MoE training benchmarks, we force-balance the token distribution among experts and all benchmarks are token-dropless.

## 25.09 NeMo Container

### Pre-Training Performance

#### System: DGX-GB200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 31357 (29925) | 1614 (1540) |
| LLAMA3_70B | 64 | 128 | 2 | 8192 | 1 (0) | 1 (2) | 1 (4) | 1 | 1 (5) | 1 | 1 (16) | 3986 (3546) | 1791 (1593) |
| LLAMA3.1_405B | 128 | 64 | 1 | 8192 | 1 (0) | 2 (4) | 1 (8) | 1 (2) | 1 (8) | 1 | 1 (32) | 729 (578) | 1840 (1458) |
| DeepSeekV3 (tokendrop) | 256 | 2048 | 1 | 4096 | 0 | 1 | 4 (8) | 1 | 4 (2) | 64 | 32 (64) | 3454 (2835) | 899 (738) |
| Qwen3_30B_a3B (tokendrop) | 8 | 512 | 4 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 16 | 22775 (23723) | 524 (546) |
| Qwen3_235B_a22B (tokendrop) | 64 | 1024 | 1 | 4096 | 0 | 2 | 1 | 1 | 1 | 64 | 32 | 4452 (4416) | 659 (654) |

#### System: DGX-B200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 2 | 8192 | 0 | 1 | 1 | 1 | n/a | 1 | 8 | 29994 (29388) | 1544 (1513) |
| LLAMA3.1_405B | 128 | 64 | 1 | 8192 | 0 | 4 | 8 | 2 | 8 | 1 | 32 | 664 (622) | 1676 (1569) |
| DeepSeekV3 (tokendrop) | 256 | 2048 | 1 | 4096 | 0 | 1 | 16 | 1 | 1 | 8 | 128 | 2265 (2159) | 589 (562) |
| Qwen3_30B_a3B (tokendrop) | 8 | 512 | 1 | 4096 | 0 | 1 | 1 | 1 | 1 | 8 | 64 | 18066 | 416 |
| Qwen3_235B_a22B (tokendrop) | 64 | 1024 | 1 | 4096 | 0 | 1 | 8 | 1 | 2 | 8 | 128 | 4104 (4275) | 607 (633) |

#### System: DGX-H100

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | PP | CP | VP | EP | GA | Tokens / sec / GPU | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-----------------------|-------------------------|
| LLAMA3_8B | 8 | 128 | 1 | 8192 | 1 | 1 | 1 | 1 | n/a | 1 | 16 | 14079 | 725 |
| LLAMA3_70B | 64 | 128 | 1 | 8192 | 0 | 4 | 8 | 1 | 5 | 1 | 64 | 1619 | 727 |
| LLAMA3.1_405B | 1024 | 512 | 1 | 8192 | 0 | 8 | 8 | 2 | 8 | 1 | 64 | 302 | 763 |
| DeepSeekV3 (dropless) | 1024 | 8192 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 64 | 128 | 1297 | 338 (330) |
| Qwen3_30B_a3B (tokendrop) | 16 | 512 | 2 | 4096 | 0 | 1 | 2 | 1 | 24 | 8 | 32 | 10494 | 241 |
| Qwen3_235B_a22B (tokendrop) | 256 | 2048 | 1 | 4096 | 0 | 2 | 8 | 1 | 4 | 32 | 128 | 1204 | 178 |

- The numbers in parentheses indicate the use of different quantization granularities: In case of GB200 and B200 systems, 32×32 for both weights and activations. For H100 system, 128×128 for weights and 1×128 for activations, which match those used in the original DeepSeekV3 pre-training.
- In token-dropless MoE training benchmarks, we force-balance the token distribution among experts.
