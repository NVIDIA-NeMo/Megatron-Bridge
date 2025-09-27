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

*Performance tables will be added here*

#### System: DGX-B200

*Performance tables will be added here*

#### System: DGX-H100

*Performance tables will be added here*
