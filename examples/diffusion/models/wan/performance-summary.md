# Performance

This page provides the current performance benchmarks for models using DFM across different GPU systems and configurations as we continue to optimize the model for optimal performance. Please refer to `examples/megatron/recipes/wan/conf` for updated YAML configurations.

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

---

## Megatron-Core Pre-Training Performance

#### System: DGX-GB200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-------------------------|
|Wan 2.1 14B|32|64|1|37440|1|1|0|1|2|0|0|899.62|


#### System: DGX-GB300

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-------------------------|
|Wan 2.1 14B|32|64|1|37440|0|1|0|1|2|0|0|1,030.67|

#### System: DGX-B200

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-------------------------|
|Wan 2.1 14B|32|64|1|37440|1|1|0|1|2|0|0|804.02|

#### System: DGX-H100

| Model | #-GPUs | GBS | MBS | Sequence Length | FSDP | TP | SP | PP | CP | VP | EP | Model TFLOP / sec / GPU |
|-------|--------|-----|-----|-----------------|------|----|----|----|----|----|----|-------------------------|
|Wan 2.1 14B|128|128|1|37440|0|2|1|1|4|0|0|325.77|
