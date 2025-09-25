# Communication Overlap

Megatron Bridge supports overlapping communication with computation in distributed training to improve performance and throughput. This optimization technique reduces the impact of inter-GPU communication overhead by executing communication operations concurrently with computational operations whenever possible.

Communication overlap is managed through the {py:class}`bridge.training.comm_overlap.CommOverlapConfig` class and can be applied to different types of parallelism: tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP), and context parallelism (CP).

## Data-parallel Communication Overlap

Megatron Bridge supports the overlap of data-parallel (DP) communications with computations in LLM training. The framework features a Distributed Optimizer that distributes optimizer states and high-precision master parameters across GPUs. This introduces two types of data-parallel communications: reduce-scatter of gradients and all-gather of updated parameters.

The DP communication is chunked by the granularity of a Transformer layer and overlaps each communication chunk with computation. This overlap method exposes only one DP communication chunk ensuring efficient large-scale LLM training. When training with pipeline parallelism, the granularity of DP communication becomes the Transformer layers per virtual pipeline stage.

### Configuration

DP communication overlap settings can be inspected in Megatron Core via the `DistributedDataParallelConfig` class. DP gradient reduce-scatter and parameter all-gather overlaps are enabled when setting `overlap_grad_reduce=True` and `overlap_param_gather=True`, respectively. The precision of gradient reduce-scatter is controlled by `grad_reduce_in_fp32`. When `grad_reduce_in_fp32=False`, gradients are reduced in bf16, leading to improved performance in large-scale training compared to the default fp32 precision. When training in fp8 computing precision, setting `fp8_param_gather=True` conducts the parameter all-gather in fp8, reducing the all-gather overhead by half.

Data parallel communication overlap settings are controlled through the distributed data parallel and communication overlap configurations.

```{note}
Data-parallel overlap relies on attributes such as `grad_reduce_in_fp32` and `fp8_param_gather`. When a mixed-precision recipe (for example `bf16_mixed`, `fp16_mixed`, `bf16_with_fp8_delayed_scaling_mixed`, etc.) is provided, those attributes are sourced from the recipe stored in the `MixedPrecisionConfig`. Set the desired values inside the mixed-precision configuration rather than overriding them directly on the optimizer or DDP configs. This ensures the communication overlap settings and the selected precision recipe remain consistent.
```

For example:

```python
from megatron.bridge.training.config import ConfigContainer, OptimizerConfig
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.mixed_precision import get_mixed_precision_config

# Configure communication overlap
comm_overlap_config = CommOverlapConfig(
    tp_comm_overlap=False,  # Tensor parallel overlap
    overlap_grad_reduce=True,  # Gradient reduce-scatter overlap
    overlap_param_gather=True,  # Parameter all-gather overlap
    overlap_param_gather_with_optimizer_step=False,  # Advanced optimization
    bucket_size=128 * 1024 * 1024,  # 128MB bucket size
)

# Configure distributed optimizer
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=3e-4,
    use_distributed_optimizer=True,  # Required for DP overlap
    # ... other optimizer parameters
)

# Mixed precision configuration controls overlap-related attributes
mixed_precision_config = get_mixed_precision_config("bf16_mixed")
mixed_precision_config.grad_reduce_in_fp32 = False  # Use bf16 for gradient reduction
mixed_precision_config.fp8_param_gather = False

config = ConfigContainer(
    comm_overlap=comm_overlap_config,
    optimizer=optimizer_config,
    mixed_precision=mixed_precision_config,
    # ... other config parameters
)
```

Key data parallel overlap options:

- `overlap_grad_reduce`: Overlaps gradient reduce-scatter with computation (default: True)
- `overlap_param_gather`: Overlaps parameter all-gather with computation (default: True)  
- `overlap_param_gather_with_optimizer_step`: Advanced optimization for pipeline parallelism
- `bucket_size`: Controls the granularity of communication chunking (default: 128MB)
- `grad_reduce_in_fp32`: Controls gradient reduction precision (False for bf16, True for fp32)
- `fp8_param_gather`: Enables fp8 parameter all-gather for reduced communication overhead

## Tensor-parallel Communication Overlap

Tensor parallelism, used with sequence-parallel activation sharding (`sequence_parallel=True`), introduces activation (gradient) all-gather and reduce-scatter operations. Megatron Bridge provides various options to overlap the tensor-parallel (TP) communications with computation.

![Tensor-parallel Communication Overlap](images/tp_comm_overlap.png)
*Figure: Tensor-parallel communication overlap showing bulk and pipelined overlap strategies.*

The TP communication without direct computation dependency are overlapped with the computation in bulk (the linear layer and TP communication pairs in the yellow boxes). The bulk TP communication is enabled by default. The other TP communications with direct computation dependency are overlapped in pipelined fashion (the linear layer and TP communication pairs in the red boxes).

In the pipelined overlap, the activation (gradient) tensor all-gather is replaced with multiple steps of input P2P ring exchanges, and reduce-scatter is replaced with multiple steps of GEMM output P2P ring exchanges followed by a reduction of the received outputs.

### Configuration

```python
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig, 
    TransformerLayerTPOverlapCfg,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
)

# Configure tensor parallel overlap
comm_overlap_config = CommOverlapConfig(
    tp_comm_overlap=True,  # Enable TP communication overlap
    tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,  # Predefined config
    tp_comm_bootstrap_backend="nccl",  # Communication backend
)
```

Requirements for TP communication overlap:
- `tensor_model_parallel_size >= 2`
- `sequence_parallel=True`
- Appropriate hardware configuration

### Advanced Configuration

For most use cases, setting `tp_comm_overlap=True` with `tp_comm_overlap_cfg=None` (the default) will automatically configure appropriate overlap settings. For advanced users requiring custom optimization, Megatron Bridge includes predefined configurations optimized for specific hardware and model combinations. These configurations are available in the `comm_overlap` module but require expert knowledge to use effectively.

## Pipeline-parallel Communication Overlap

Pipeline parallelism introduces P2P activation (gradient) sends and receives between pipeline-parallel (PP) GPUs. The PP communication frequency increases when increasing the virtual-pipeline-parallel size because the number of Transformer layers executed per micro-batch decreases.

![Pipeline-parallel Communication Overlap](images/pp_comm_overlap.png)
*Figure: Pipeline-parallel communication overlap in 1F1B pipelining phase.*

Megatron Bridge supports the overlap of PP communications with non-dependent computations in the 1F1B stage (the body of pipelining, where 1 forward and 1 backward micro-batch executions are interleaved). The PP communications in pipeline fill and flush stages are still exposed.

### Configuration

```python
comm_overlap_config = CommOverlapConfig(
    tp_comm_overlap=False,
    overlap_p2p_comm=True,  # Enable PP communication overlap
    batch_p2p_comm=False,   # Use separate send/receive kernels
)
```

PP communication overlap settings:
- `overlap_p2p_comm`: Enables overlap of P2P communications (default: auto-configured)
- `batch_p2p_comm`: Uses batched vs separate kernels (default: auto-configured based on virtual PP)

The overlap is automatically enabled when:
- `pipeline_model_parallel_size > 1`
- `virtual_pipeline_model_parallel_size > 1` (for optimal performance)

## Context-parallel Communication Overlap

Context parallelism partitions activations (gradients) on all layers in the sequence domain. This introduces all-gather and reduce-scatter of activations (gradients) in self-attention forward- and back-propagations.

Megatron Bridge hides the context-parallel (CP) communications under the self-attention computation. Like the TP communication overlaps, the CP communications are chunked then pipeline-overlapped with the self-attention computation, where the all-gather and the reduce-scatter of activations (gradients) are replaced with P2P ring exchanges of data.

### Automatic Configuration

The CP communication overlap is automatically enabled when context parallelism is used (`context_parallel_size > 1`). No additional configuration is required as the overlap is built into the context parallelism implementation.

## MoE Expert Parallel Communication Overlap

For Mixture of Experts (MoE) models, Megatron Bridge supports overlapping expert parallel all-to-all communications with computation.

### Configuration

```python
comm_overlap_config = CommOverlapConfig(
    tp_comm_overlap=False,
    overlap_moe_expert_parallel_comm=True,  # Enable MoE EP overlap
    delay_wgrad_compute=True,  # Advanced MoE optimization
)
```

Requirements for MoE expert parallel overlap:
- `expert_model_parallel_size > 1`
- `num_moe_experts > 1`
- `moe_token_dispatcher_type` in ["alltoall", "flex"]
- BF16 or FP16 precision
- PyTorch >= 2.6.0
- Specific recomputation settings

## Complete Configuration Example

Here's a comprehensive example combining multiple communication overlap strategies:

```python
from megatron.bridge.training.config import ConfigContainer, OptimizerConfig
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
)
from megatron.bridge.models import GPTModelProvider

# Model configuration with parallelism
model_config = GPTModelProvider(
    # Parallelism settings
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=2,
    context_parallel_size=2,
    sequence_parallel=True,
    
    # Model parameters
    hidden_size=8192,
    num_layers=32,
    # ... other model parameters
)

# Communication overlap configuration
comm_overlap_config = CommOverlapConfig(
    # Tensor parallel overlap
    tp_comm_overlap=True,
    tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    
    # Pipeline parallel overlap
    overlap_p2p_comm=True,
    batch_p2p_comm=False,
    
    # Data parallel overlap
    overlap_grad_reduce=True,
    overlap_param_gather=True,
    bucket_size=128 * 1024 * 1024,
)

# Optimizer with distributed settings
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=3e-4,
    use_distributed_optimizer=True,
)

# Complete configuration
config = ConfigContainer(
    model=model_config,
    comm_overlap=comm_overlap_config,
    optimizer=optimizer_config,
)
```


## API Reference

For detailed API documentation, see:
- {py:class}`bridge.training.comm_overlap.CommOverlapConfig` - Main configuration class
- {py:class}`bridge.training.comm_overlap.TransformerLayerTPOverlapCfg` - Tensor parallel overlap configuration
- {py:class}`bridge.training.comm_overlap.BulkOverlapCfg` - Bulk overlap configuration
- {py:class}`bridge.training.comm_overlap.PipelineOverlapCfg` - Pipeline overlap configuration
- {py:class}`bridge.training.comm_overlap.RingExchangeOverlapCfg` - Ring exchange overlap configuration
megatron-core/developer-guide/latest/api-guide/tensor_parallel.html) - Underlying implementation details
