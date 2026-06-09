# Performance

This page provides the current performance benchmarks for Wan model across different GPU systems and configurations as we continue to optimize the model for optimal performance.

## Nomenclature

- **GBS**: Global Batch Size
- **MBS**: Micro Batch Size
- **FSDP**: Fully Sharded Data Parallel
  - FSDP = 1: use FSDP
  - FSDP = 0: use DDP (Distributed Data Parallel)
- **TP**: Tensor Parallel Size
- **SP**: Sequence Parallel
- **PP**: Pipeline Parallel Size
- **CP**: Context Parallel Size
- **VP**: Virtual Pipeline Parallel Size
- **EP**: Expert Parallel Size

## Performance Metrics

Performance is measured using:
- **Tokens/sec/GPU**: Throughput per GPU
- **Model TFLOP/sec/GPU**: Model floating-point operations per second per GPU

## Performance Summary for Models

Below are performance benchmarks for various models using DFM framework.

The performance data includes:

- **Pre-training Performance**: Throughput metrics for various model sizes and architectures
- **System Configurations**: Results across different GPU systems (DGX-GB200, DGX-GB300, DGX-H100)

Note: The GB200/B200 results were measured using Nemo container version 25.11 (`nvcr.io/nvidia/nemo:25.11`). The H100/GB300 results were measured using Nemo container version 25.09 (`nvcr.io/nvidia/nemo:25.09.00`).

---

## Megatron-Core Pre-Training Performance

### Wan 2.1 14B

| System     | GPUs | GBS | Seq Len | Parallelism (TP/SP/PP/CP) | FSDP | TFLOP/s/GPU |
|:-----------|-----:|----:|--------:|:--------------------------|-----:|-------------:|
| DGX-GB300  |   32 |  64 |   37440 | 1 / 0 / 1 / 2            | 0    | 1,030.67 |
| DGX-GB200  |   32 |  64 |   37440 | 1 / 0 / 1 / 2            | 1    | 899.62       |
| DGX-B200   |   32 |  64 |   37440 | 1 / 0 / 1 / 2            | 1    | 804.02       |
| DGX-H100   |  128 | 128 |   37440 | 2 / 1 / 1 / 4            | 0    | 325.77       |

> TP = Tensor Parallelism, SP = Sequence Parallelism, PP = Pipeline Parallelism, CP = Context Parallelism
