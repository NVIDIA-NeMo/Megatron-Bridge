# Performance Tuning Guide

Megatron-Bridge provides a wide range of features for performant and memory-efficient LLM training on GPUs, and comes pre-configured with optimal settings. However, factors such as model architecture, hyperparameters, GPU count, and GPU type can affect the available options, and additional tuning may be necessary to achieve optimal performance. This document explores the factors that affect training performance, highlights common issues, and outlines techniques for performance tuning that lead to higher MFU (Model FLOPS Utilization) and TCO.

```{Note}
This guide makes references to several configuration settings. These settings will be referenced relative to the the config class that contains them, e.g. `OptimizerConfig.lr`. Please see <project:apidocs/index.rst> for more details on configuration settings.
```

```{Note}
This guide references several configuration settings from `TransformerConfig`. Please apply these to the appropriate ModelProvider for your model, e.g. `GPTModelProvider`, as the `ConfigContainer` does not accept a raw `TransformerConfig`.
```

## Low Precision Training

1. Expected speedup of FP8 training compared to BF16 training

   > 1. The default low-precision LLM training recipe applies FP8 computation exclusively to the linear layers within the Transformer block, typically achieving a speedup of 1.2–1.5X.
   > 2. However, the actual speedup depends on the proportion of training time spent on these linear layers. For instance, smaller LLMs with a limited hidden size exhibit lower FP8 speedup, as linear layers scale with O(sequence_length × hidden_size²) complexity, whereas the other element-wise computation layers (e.g., layer norms, dropouts, RoPE, and simple math functions) scale with O(sequence_length × hidden_size), and dot-product attention scales with O(sequence_length² × hidden_size). Consequently, the contribution of linear layers to the overall training time is smaller in such models.
   > 3. Different FP8 recipes use varying quantization block sizes, affecting performance. Smaller quantization blocks generally incur higher overhead in both quantization and GEMM execution. For example, MXFP8 with a 1×32 quantization block performs less efficiently than full tensor-wise FP8 scaling.

2. Common issues of low FP8 training speedup

   > 1. Host performance boundness when LLM uses small GPU kernels (see [Lowering Host Overhead and Jitters](#lowering-overhead-jitter)).
   > 2. A low proportion of linear layers in training step time that use FP8 computation.

## Parallel Mapping Strategies

1. Data Parallelism using Distributed Optimizer

   > 1. You should begin with data-parallel (DP) mapping. As long as the model and activation memory fit within the GPUs, data parallelism generally offers optimal performance, minimizes communication overhead, and maximizes per-GPU tensor sizes (compared to per-tensor sharding).
   >
   > 2. Megatron-Bridge uses the distributed optimizer as the default method for data-parallel training. It shards master parameters and optimizer states across data-parallel ranks, reducing model state memory usage without increasing communication overhead compared to traditional data-parallel training.
   >
   >    > 1. `OptimizerConfig.use_distributed_optimizer=true`

2. Per-tensor Sharding (Tensor-parallel or Context-parallel mappings)

   > 1. Tensor parallelism (TP) is the primary recommendation when a model exceeds GPU memory capacity under data-parallel mapping. However, since it involves higher communication overhead, the tensor-parallel size should ideally be confined to the high-bandwidth intra-node network (NVLink domain).
   >
   >    > 1. `TransformerConfig.tensor_model_parallel_size=<int>`
   >
   > 2. When the sequence length in a training run is significantly larger than the hidden size, activation memory can overflow. In such cases, context parallelism (CP) helps by sharding tensors along the sequence dimension, allowing the workload to fit within limited GPU memory and improving performance. Like tensor parallelism (TP), CP requires inter-GPU communication of activations. However, for the same tensor sizes, CP generally results in lower communication volume.

That said, CP’s effectiveness depends on the relative sizes of the sequence length and hidden size. When the sequence length is smaller than the hidden size, CP produces narrow (or "skinny") tensor shards on each GPU. This reduces data reuse and can degrade performance.

Additionally, because CP shards activations, it also partitions optimizer states in distributed training. As a result, optimizer state partitioning spans both the data parallel (DP) and context parallel (CP) dimensions.

> > 1. `TransformerConfig.context_parallel_size=<int>`
>
> 1. Performance tips:
>
>    > 1. A large tensor-parallel or context-parallel size is not recommended unless the hidden size or sequence length is large enough to maintain sufficient per-GPU parallelism and avoid excessive communication overhead. For example, using a tensor-parallel size of 8 for LLAMA 3 70B could lead to low GPU utilization and make training host-performance bound.
>    > 2. You can combine TP and CP to optimize performance by balancing communication overhead. For example, using TP=2 along with CP=2 can give better performance than TP=4 when the sequence size is larger than the hidden size.
>    > 3. For additional tips, see [Long Sequence Training](#long-sequence-train).

1. Pipeline Parallelism

   > 1. Pipeline parallelism (PP) is necessary when a model cannot fit within GPU memory using tensor parallelism. Also, virtual pipeline parallelism (VPP) should be used in conjunction with pipeline parallelism to reduce the overhead caused by pipeline warm-up and flush bubbles.
   >
   >    > 1. `TransformerConfig.pipeline_model_parallel_size=<int>`
   >    > 2. `TransformerConfig.virtual_pipeline_model_parallel_size=<int>`
   >
   > 2. Performance tips in PP and VPP sizing:
   >
   >    > 1. PP can also be combined with per-tensor sharding methods to mitigate the impact of sharding inefficiencies and pipeline bubbles. For instance, TP4 + PP2 may outperform TP8 when both mappings fit into memory because using a large TP reduces per-GPU tensor sizes but increases the communication cost, increasing the exposed communication.
   >    > 2. VPP increases inter-stage communication overhead. When a global batch contains many micro-batches, using a smaller VPP size can improve performance, as the exposed communication cost outweighs the reduction in pipeline bubbles.
   >
   > 3. Asymmetric Transformer layer allocation across pipeline stages
   >
   >    > 1. An LLM with a large vocabulary size has computationally heavy embedding lookup and projection operations, leading to load imbalance across pipeline stages. To address this, Megatron-Bridge provides an option to allocate one fewer Transformer layer in the first and last pipeline stages, which handle embedding lookup and projection, to better balance workloads.
   >    >
   >    >    > 1. `GPTProvider.account_for_embedding_in_pipeline_split=true`
   >    >    > 2. `GPTProvider.account_for_loss_in_pipeline_split=true`

2. Expert Parallelism

   > 1. Expert Parallelism (EP) is designed specifically for Mixture-of-Experts (MoE) models to efficiently distribute sparse MLP weights across multiple chips. It can be used in combination with other parallelism strategies such as Tensor Parallelism (TP), Context Parallelism (CP), Pipeline Parallelism (PP), Data Parallelism (DP), and Fully Sharded Data Parallel (FSDP). In the current design, the dense attention part and the sparse MLP part are fully decoupled in terms of their TP, CP, and DP parallelism configurations. Expert Tensor Parallelism (ETP) is introduced to specifically control the tensor parallelism for the sparse MLP part. ETP uses TP for dense layers for the ranks allocated for EP in sparse layers. On the other hand, the baseline is DEP, which folds DP in dense layers for EP in sparse layers.
   >
   >    > 1. `TransformerConfig.expert_model_parallel_size=<int>`
   >    > 2. `TransformerConfig.expert_tensor_parallel_size=<int>`
   >
   > 2. Performance tips in hybrid folding options and EP sizing:
   >
   >    > 1. Typically, EP is kept within the high-bandwidth intra-node network (NVLink domain) to minimize the communication overhead it can introduce. However, using communication overlap techniques—such as pipeline overlap or 1F1B overlap—along with PP (e.g., DualPipe) might make it possible to expand EP into the inter-node networks.
   >    >
   >    > 2. Within the sparse MLP block, DP replaces CP because it has no impact on the computation pattern based on the dispatched tokens in each EP rank.
   >    >
   >    > 3. Usually, ETP is set to 1 to avoid significant communication overhead that comes with applying TP to MLP GEMMs.
   >    >
   >    > 4. When multiple experts are placed on a single chip after applying Expert Parallelism, enabling grouped GEMM can significantly improve computation efficiency.
   >    >
   >    >    > 1. `TransformerConfig.moe_grouped_gemm=True`

3. Fully Sharded Data Parallelism

   > 1. Megatron-Bridge supports PyTorch-native FSDP. FSDP can be used in combination with per-tensor sharding methods.
   >
   >    > 1. To use PyTorch FSDP2:
   >    >
   >    >    > 1. `DistributedInitConfig.use_torch_fsdp2=True`
   >
   > 2. FSDP can be preferred over TP+PP+DP mappings in the following scenarios:
   >
   >    > 1. Small models with a large sequence, thus the parameter AllGather and gradient ReduceScatter can effectively be hidden under computation and the short communication overlap causes minor interference to the computation under overlap.
   >    > 2. In FSDP training, activation storage remains as the main memory bottleneck because FSDP only shards model state memory, and a large per-GPU activation is needed to hide the costly FSDP communication. On GB200 GPUs, Megatron-Bridge offers an option to offload activations to the host memory via a high-speed chip-to-chip interconnect.
   >    > 3. Baseline training is host performance-bound, but FSDP allows for larger per-GPU tensor sizes by eliminating TP or enabling a larger micro-batch size.

   <!-- TODO: support megatron custom fsdp -->
   <!-- > 1. Megatron-Bridge supports two Fully Sharded Data Parallelism (FSDP) implementations: PyTorch-native FSDP and a custom Megatron FSDP built within Megatron Core. While both follow the same sharding principles, the custom implementation is further optimized for performance. The performance gain of the custom FSDP comes primarily from minimizing the data movement to the communication tensors and reusing communication buffers. Both FSDP methods can be used in combination with per-tensor sharding methods. -->
   <!-- > -->
   <!-- >    > 1. To use PyTorch FSDP2: -->
   <!-- >    > -->
   <!-- >    >    > 1. `DistributedInitConfig.use_torch_fsdp2=True` -->
   <!-- >    > -->
   <!-- >    > 2. To use Custom Megatron FSDP: -->
   <!-- >    > -->
   <!-- >    >    > 1. `recipe.trainer.strategy.fsdp="megatron"` -->
   <!-- >    >    > 2. `recipe.trainer.strategy.ddp.data_parallel_sharding_strategy="optim_grads_params"` -->
   <!-- > -->
   <!-- > 2. FSDP can be preferred over TP+PP+DP mappings in the following scenarios: -->
   <!-- > -->
   <!-- >    > 1. Small models with a large sequence, thus the parameter AllGather and gradient ReduceScatter can effectively be hidden under computation and the short communication overlap causes minor interference to the computation under overlap. -->
   <!-- >    > 2. In FSDP training, activation storage remains as the main memory bottleneck because FSDP only shards model state memory, and a large per-GPU activation is needed to hide the costly FSDP communication. On GB200 GPUs, Megatron-Bridge offers an option to offload activations to the host memory via a high-speed chip-to-chip interconnect. -->
   <!-- >    > 3. Baseline training is host performance-bound, but FSDP allows for larger per-GPU tensor sizes by eliminating TP or enabling a larger micro-batch size. -->

4. Heterogeneous Encoder Parallelism

   > 1. Encoder Pipeline Parallel
   >
   >    > 1. Use `T5ModelProvider.encoder_pipeline_model_parallel_size`.
   >    > 2. In an Encoder-Decoder architecture like Multimodal models (VLMs like NeVA etc.), Encoder Pipeline Parallel can be used to add pipeline parallelism to the encoder.
   >    > 3. Pipeline parallelism controls the amount of pipelining in the decoder part.
   >    > 4. Encoder Pipeline Parallel is limited to 1 at the moment, i.e., the encoder can occupy a maximum of 1 PP stage.
   >    > 5. By default, Encoder Pipeline Parallel is 0 and Decoder Pipeline Parallel is 1.
   >    > 6. When the Encoder Pipeline Parallel size is 0, it shares the first PP stage of the Decoder.
   >
   > 2. Encoder Tensor Parallel
   >
   >    > 1. Use `T5ModelProvider.encoder_tensor_model_parallel_size`.
   >    > 2. Since encoders tend to be much smaller than decoders, we also provide the ability to set a different amount of tensor parallelism to the encoder than the decoder.
   >    > 3. By default, encoder tensor parallel is set to 0, i.e., the amount of tensor parallelism in the encoder is equal to tensor parallelism in the decoder.
   >    > 4. To use this option, Encoder Pipeline Parallel must be greater than 0 as we need the encoder to be on its own pipeline stage.
   >    > 5. Encoder Tensor Parallel size is limited to be less than or equal to Tensor parallel size.
   >
   > 3. Total number of GPUs required when these features are used is:
   >
   >    > 1. Data Parallel size * Context Parallel size * ((Encoder TP * Encoder PP) + (Decoder TP * Decoder PP))
   >
   > 4. These features are experimental and may still have bugs. There are critical bug fixes that will be made in a future release.

5. Parallel mapping strategies with NVL72

   > 1. Training with only data parallelism or FSDP makes it straightforward to fully utilize the bandwidth of an NVL72 system. However, when combining multiple parallelism strategies, it's important to ensure that high-volume communicators remain confined within each NVL72 domain. For example, with TP=4, DP=16, and PP=4, the GPUs in the first TP group of DP1/PP1 spans both NVLink and network domains, causing communication performance to be bottlenecked by the slower network link. To avoid this, you may choose TP and DP sizes such that the product of TP × DP divides evenly into the NVL72 configuration. If the model-parallel size does not align naturally, padding may be required to support non-divisible group sizes.
   > 2. To avoid this partitioning complexity, you can just use 64 GPUs out of the 72 GPUs.

## Communication Overlaps and Tuning

1. Data-parallel communication of Distributed Optimizer

   > 1. Distributed optimizer overlaps parameter AllGathers with the forward computation of the first micro-batch and gradient ReduceScatters with the backward computation of the last micro-batch.
   >
   >    > 1. `DistributedDataParallelConfig.overlap_param_gather=true`
   >    > 2. `DistributedDataParallelConfig.overlap_grad_reduce=true`
   >
   > 2. When using the distributed optimizer with pipeline parallelism (PP) + virtual pipeline parallelism (VPP), DP communications overlap with multiple micro-batches, increasing the opportunity for effective overlap. Also, Megatron-Bridge aligns the execution timing of DP communications across pipeline-parallel ranks to synchronize the computing kernel slowdown from the overlap.
   >
   >    > 1. `DistributedDataParallelConfig.align_param_gather=true`
   >
   > 3. Slow DP communication at large scaling training:
   >
   >    > 1. Distributing optimizer states across a partial DP domain reduces communication costs over high-latency Ethernet networks. Model states remain replicated outside the distributed domain. During the final micro-batch backpropagation, gradient ReduceScatters occur within the distributed domain, followed by AllReduce in the non-distributed domain. Parameter AllGathers are performed only within the distributed domain.
   >    >
   >    >    > 1. `DistributedDataParallelConfig.num_distributed_optimizer_instances= <int>`
   >    >
   >    > 2. A large message size for DP communication is recommended to maximize network bandwidth utilization. You can achieve this by increasing the communication bucket size.
   >    >
   >    >    > 1. `DistributedDataParallelConfig.bucket_size=<number_of_elements: int>`
   >
   > 4. A common reason for DP communication overlap failure:
   >
   >    > 1. Persistent Layer Normalization (LN) kernels from Transformer Engine use spin-waiting for all SMs in the GPU, causing the LN kernel and subsequent computation kernels to be scheduled only after DP communication. To prevent this, an appropriate SM margin should be configured using the following environment variables.
   >    >
   >    >    > 1. `NVTE_FWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives = 16>`
   >    >    > 2. `NVTE_BWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives = 16>`

<!-- 2. Custom Megatron FSDP -->

<!--    > 1. Unless you specify the communication bucket size, MCORE FSDP uses fixed communication overlap that overlaps the parameter AllGather and gradient ReduceScatter of each Transformer layer with its associated forward and backward computations. -->

3. Tensor-parallel (TP) communication (with sequence parallelism)

   > 1. Megatron-Bridge currently uses the userbuffer backend in Transformer Engine for TP communication overlaps. This offers the pipelined overlap of the TP communication with dependent computation.
   >
   >    > 1. `CommOverlapConfig.tp_comm_overlap`
   >
   > 2. The overlap method, resource, and precision of the TP communication overlaps are configurable, and the most performant configurations are set in the Megatron-Bridge training recipes by default. Also, you can set a custom TP communication overlap configuration via the below interface following the structure of TransformerLayerTPOverlapCfg class.
   >
   >    > 1. `CommOverlapConfig.tp_comm_overlap_cfg=<TransformerLayerTPOverlapCfg>`
   >
   > 3. TP communication overlap setting tips
   >
   >    > 1. Balancing the number of SMs between communication and GEMM
   >    >
   >    >    > 1. For AllGather/ReduceScatter bulk and ReduceScatter pipelined overlap, you can adjust the number of SMs to balance communication and GEMM execution. Allocating too many SMs to communication may degrade GEMM performance, while too few may expose communication overhead. The default SM allocation for communication is 16, but you can fine-tune it based on profiling results.
   >    >    > 2. `TPOverlapCfg.num_sm=<int>`
   >    >
   >    > 2. CGA sizing to improve SM utilization
   >    >
   >    >    > 1. The CGA size can be set between 1 and 4, but it should not exceed the number of SMs allocated for communication. We recommend using CGA ≤ 2 to prevent potential SM rasterization that could impact GEMM performance.
   >    >    > 2. `TPOverlapCfg.cga_size=<int≤4>`
   >    >
   >    > 3. Use 4× splits for ReduceScatter and GEMM overlap to optimize the balance between GEMM efficiency and communication exposure.
   >    >
   >    >    > 1. In GEMM-then-ReduceScatter pipeline overlap, a 1× ReduceScatter chunk remains exposed. A small split size increases communication exposure, while a large split size may degrade performance due to aggregated GEMM wave quantization. We find that num_splits = 4 generally provides the best performance.
   >    >    > 2. `TPOverlapCfg.num_split=<int>`
   >
   > 4. Common reason for TP comm overlap failure at Hopper
   >
   >    > 1. At H100 GPU, an environment variable `CUDA_DEVICE_MAX_CONNECTIONS=1` should be set. Otherwise, TP communication kernels can be scheduled at the end of GEMM to overlap with.
   >    > 2. Pipelined TP communication overlap is used by a static userbuffer registered upon model initialization. Therefore, it doesn't support activation tensors dynamically changing between steps or between Transformer layers.

4. Context-parallel (CP) communication

   > 1. CP communication is configurable via "cp_comm_type", which can be "p2p", "all_gather", "a2a", or "a2a+p2p". Communications of "p2p" are implemented as ring-exchange send/receive operations, and they are hard-coded to overlap with the attention compute of sequence chunks. See [Long Sequence Training](#long-sequence-train) for more details.

5. Expert-parallel communication

   > 1. To hide the A2A/AG communication introduced by EP, pipeline split overlap or 1F1B overlap alongside Pipeline Parallelism could be possible. It will be added to Megatron-Bridge in future releases.

6. Pipeline-parallel (PP) send/receive communication

   > 1. PP send/recv in steady 1F1B states are set to be overlapped with computes by default.
   > 2. The PP send/recv in warmup and flush are exposed by default.

(comm-data-types)=
## Communication Data Types

1. FP8 data-parallel parameter AllGather in Distributed Optimizer and FSDP

   > 1. Megatron-Bridge supports FP8 parameter AllGather for per-tensor FP8 scaling recipes. This operation is lossless, enhancing performance while reducing memory usage.
   >
   >    > 1. `MixedPrecisionConfig.fp8_param=true`

2. BF16 (instead of FP32) data-parallel reduction in Distributed Optimizer and FSDP

   > 1. We have validated that BF16 reduction is numerically safe across numerous model training runs. However, BF16 reduction with a large data-parallel size (e.g., DP ≥ 128), especially the Ring reduction algorithm—which accumulates copies sequentially—may impact numerical stability. When using SHARP with NVIDIA InfiniBand, BF16 reduction is more robust, as it performs binary additions with higher precision for intermediate partial reductions.
   >
   >    > 1. `DistributedDataParallelConfig.grad_reduce_in_fp32=false`

3. FP8 tensor-parallel ReduceScatter

   > 1. When communication latency exceeds GEMM execution time, using FP8 input ReduceScatter can better hide communication overhead. This approach has low numerical impact, as the GEMM output must be cast to FP8 and then converted back to high precision during reduction.
   >
   >    > 1. `TPOverlapCfg.fp8_buf=true`

4. FP8 A2A Dispatch for expert parallel communication

   > 1. Megatron-Bridge is working on supporting FP8 A2A dispatch (before expert FC1), but still keeps BF16 A2A combine (after expert FC2).

## Performance at Scale

1. Scaling a training job is typically achieved by increasing the size of the data-parallel domain. In large-scale training, this often results in a small number of micro-batches per global batch—or even a single micro-batch—causing most computations to overlap with data-parallel communication. To maintain high performance in such scenarios, you should focus on minimizing the overhead of data-parallel communication and reducing host-driven inter-GPU jitter.

2. You can lower the overhead of data-parallel communication by (1) reducing the communication precision e.g., BF16 for gradient reduction and FP8 parameter gathering, (2) improving the efficiency of communication by increasing the data-parallel communication message size or using the hierarchical data-parallel reduction, or (3) using multi-cast and switch reduction with SHARP in case of InfiniBand network.

   > 1. Using BF16 gradient reduction and FP8 parameter gather are described in [Communication Data Types](#comm-data-types)
   >
   > 2. For non-pipeline-parallel training, the data-parallel communication bucket size can be adjusted using the knobs below. In pipeline-parallel training, however, the bucket size is fixed and determined by the number of parameters assigned to each virtual pipeline rank.
   >
   >    > 1. `DistributedDataParallelConfig.bucket_size=<int: bytes>`
   >
   > 3. Setting the knob below splits the data-parallel domain of the distributed optimizer into a sharding domain and a replication domain. Gradient reduction then occurs in two stages—one within each domain—avoiding the use of a single large flat ring for collective operations that have high latency.
   >
   >    > 1. `DistributedDataParallelConfig.num_distributed_optimizer_instances=<int: ≤dp_size>`

3. Ideas to reduce the host-driven inter-GPU jitters are discussed in [Lowering Host Overhead and Jitters](#lowering-overhead-jitter).

(lowering-overhead-jitter)=
## Lowering Host Overhead and Jitters

1. Common observation associated with host overhead

   > 1. Significantly low GPU FLOPS.
   > 2. Small performance gain of low-precision (FP8) training.
   > 3. Small LLMs with small hidden size or sequence length or fine-tuning without sequence packing
   > 4. High multi-GPU communication variation.

2. Increasing micro-batch size and reduce per-tensor sharding

   > 1. The most common way to increase per-GPU tensor size is by increasing the micro-batch size or minimizing unnecessary per-tensor sharding (e.g., TP or CP) when GPU memory permits.

3. Manual garbage collection to align the host interruption across GPUs

   > 1. Megatron-Bridge manually aligns the timing of garbage collection across GPUs that significantly mitigate the host overhead compared to the baseline automatic garbage collection.
   >
   >    > 1. `TrainingConfig.manual_gc_interval=<int>`

4. CUDA graph to eliminate repeated static host code execution

   > 1. Megatron-Bridge supports graph capture, significantly reducing host overhead. CUDA Graph is applicable only to LLMs with a static tensor shape across training steps. For example, it supports fixed-size packed sequences but does not handle sequences with varying lengths at each step. Also, MoE models with token-dropless propagation have limited CUDA graph support, restricted to the dense modules only.
   > 2. CUDA graph requires additional memory for static buffer management, typically adding a few gigabytes for static buffers, while models with PP size > 1 may consume over 10GB. We are actively working to reduce this memory overhead.
   > 3. `TransformerConfig.enable_cuda_graph=true`

5. Bind CPU memory for GPU processes

   > 1. Binding CPU cores to GPU processes helps mitigate long latency issues and ensures minimal variation in GPU queuing latency across GPUs. This optimization significantly impacts, particularly when the communication domain size is large.
   > 2. Example command line for a X86-based GPU system: `numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4)) <run script>`
   > 3. Example command line for a Grace-based GPU system: `numactl --cpunodebind=$((SLURM_LOCALID/2)) --membind=$((SLURM_LOCALID/2)) <run script>`

(reducing-memory-overflow)=
## Techniques for Reducing Memory to Avoid Memory Overflow and Enhance Training Efficiency

1. Activation recomputation

   > 1. Megatron-Bridge LLMs default to dot-product attention-only recomputation using Flash Attention, efficiently regenerating large intermediate activations from the attention operation with minimal computational overhead.
   >
   > 2. Megatron-Bridge also supports recomputing the full intermediate activations of a Transformer block, significantly reducing activation memory usage at the cost of approximately 30% additional computation. The number of Transformer blocks to recompute can be adjusted using a configurable setting.
   >
   >    > 1. `TransformerConfig.recompute_granuality=full`
   >    > 2. `TransformerConfig.recompute_method=block`
   >    > 3. `TransformerConfig.recompute_num_layers=<int:≤num_layers_in_the_model>`

2. Activation offloading to host memory

   > 1. Megatron-Bridge supports offloading activation memory to host memory, essential for training tasks constrained by activation memory. This is particularly useful for scenarios like (1) FSDP, where model state memory is minimized through sharding but activation memory remains high, (2) LoRA, which has frozen parameters but significant activation memory demands, and (3) the training with a large sequence length. The efficiency of activation offloading depends on both the interconnect bandwidth between the GPU and host and the host memory bandwidth. From this perspective, Grace-based systems like the GB200 enhance offloading performance by optimizing these bandwidths.
   >
   > 2. The following knobs should be configured to enable offloading and specify the number of Transformer layers to offload to host memory. The maximum number of layers that can be offloaded depends on host memory capacity, which may be lower when the CPU is shared among multiple GPUs.
   >
   >    > 1. `TransformerConfig.cpu_offloading=True`
   >    > 2. `TransformerConfig.cpu_offloading_weights=False`
   >    > 3. `TransformerConfig.cpu_offloading_num_layers= <int:≤activation_offload_layers>`
   >
   > 3. Environment variable settings to avoid resource conflict between CPU memory offloading and network communication
   >
   >    > 1. `NCCL_NET_GDR_LEVEL=PHB # NCCL <=2.25`
   >    > 2. `NCCL_NET_GDR_C2C=1     # NCCL >=2.26`
   >
   > 4. Optimization tips
   >
   >    > 1. Given the ratio between activation volume and computational operations, offloading all layer activations naively can become a performance bottleneck. Optimizing performance requires tuning the number of layers to offload while balancing it with recomputation.

3. Weight memory-optimized BF16 training

   > 1. In BF16 training, Megatron-Bridge optimizes memory usage by storing only the BF16 remainder of the master weight copies for the next optimizer update. This is possible because BF16 data can be represented using a subset of FP32 bits, allowing Megatron-Bridge to avoid redundant storage of the FP32 portion used for BF16 representation. This is default enabled when using precision-aware optimizer in Megatron Core.
   >
   >    > 1. `OptimizerConfig.use_precision_aware_optimizer=True`

4. Common memory usage hikes from environment variable setting

   > 1. The below environment variables will (1) avoid preserving the buffers for NCCL communication and (2) disable NVLSharp when not used. Both these options lower the GPU memory usage.
   >
   >    > 1. `TORCH_NCCL_AVOID_RECORD_STREAMS=1`
   >    > 2. `NCCL_NVLS_ENABLE=0`
   >
   > 2. While not enabled by default, you can further reduce memory usage caused by segmentation penalties by setting the env var shown below.
   >
   >    > 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

5. Keep parameters in FP8 at FP8 training

   > 1. In FP8 training, after optimizer step execution, we can keep the parameters in FP8. Compared to the baseline that keeps the intermediate weight values in BF16, FP8 parameters lower memory usage and improve communication performance. The below knob enables keeping the parameters in FP8.
   >
   >    > 1. `MixedPrecisionConfig.fp8_param_gather=True`

## Operator Fusion

1. You can control specific fusion behaviors using the following configuration knobs:

   > 1. `TransformerConfig.masked_softmax_fusion=true`
   > 2. `GPTProvider.cross_entropy_loss_fusion=true`
   > 3. `GPTProvider.gradient_accumulation_fusion=true`
   > 4. `TransformerConfig.bias_activation_fusion=true`
   > 5. `TransformerConfig.bias_dropout_fusion=true`
   > 6. `TransformerConfig.apply_rope_fusion=true`

2. Megatron-Bridge offers different Flash Attention options, which can be chosen through the model config:

   > 1. Let Transformer Engine decide (default): `TransformerConfig.attention_backend=AttnBackend.auto`
   > 2. FlashAttention2: `TransformerConfig.attention_backend=AttnBackend.flash`
   > 3. cuDNN fused attention: `TransformerConfig.attention_backend=AttnBackend.fused`

(long-sequence-train)=
## Long Sequence Training

1. Problem of long sequence training

   > 1. Training with long sequence length can lead to memory overflow due to the huge memory cost of activations. The problem could be solved by recomputing activations in backward, but it can impose up to ~30% overheads in each training step. Context parallelism is a better solution which splits the sequence dimension across multiple GPUs, so that each GPU only computes and saves activations of a sequence chunk. In this way, memory overflow is addressed without introducing any redundant compute.

2. CP to shard activation (knob)

   > 1. `TransformerConfig.context_parallel_size=<int>`
   >
   >    > 1. Both TP and CP can reduce activation memory overheads. It's not wise to be biased to either of them. Communications of TP and CP are overlapped by GEMM and Attention respectively. Blindly enlarging their sizes can make some communications hard to overlap. It's recommended to sweep a combination of TP+CP configs. The optimal config is expected to make full use of all related compute and do best overlapping, thereby achieving best end-to-end performance.
   >
   > 2. `TransformerConfig.cp_comm_type=<str> or <list of str>`
   >
   >    > 1. Megatron-Core provides multiple implementation variants of CP and allows you to make choices based on your specific use cases by configuring "cp_comm_type". The configuration value can be `p2p`, `all_gather`, `a2a`, or `a2a+p2p`. These communication types are compatible with each other, so they can be flexibly interleaved between transformer layers. You only need to provide a list, where each element corresponds to a layer.
   >    > 2. `p2p`: exchanges KV sequence chunks in ring-topology. The P2P communications can be fully overlapped.
   >    > 3. `all_gather`: inserts an all-gather before attention to get a full sequence of KV. The all-gather is exposed, but it should not impose big overheads if GQA/MQA are used, as they have very few KV heads.
   >    > 4. `a2a`: is an implementation of DeepSpeed Ulysses. A2A communications are added before and after the attention module to gather full sequence length and further scatter heads in CP domain. A2A cannot be overlapped.
   >    > 5. `a2a+p2p`: is a middle ground between `a2a` and `p2p`. This is useful for cases of big CP sizes, where each sequence chunk is too short to overlap P2P communications. It first does A2A in partial CP groups to gather relatively longer sequence chunks, then applies P2P implementation to the gathered chunks. It also can be helpful for hierarchical CP communications, for example A2A and P2P happen in NVLink and IBLink domains respectively.
   >    > 6. With small and medium CP size, `p2p` is the recommended configuration because communications can be fully overlapped; "all_gather" also should work fine with GQA/MQA. As for strongly-scaling a sequence length with big CP sizes, the short chunk length can barely overlap the `p2p` communications, so `a2a+p2p` ought to be the preferred choice. `a2a` could be adopted in some cases for its simplicity. However, CP size can be restricted with "a2a" because it requires the number of attention heads to be divisible by CP size. Restricted CP size will finally limit the sequence length that can be run.

3. Activation recomputation (in [Techniques for Reducing Memory to Avoid Memory Overflow and Enhance Training Efficiency](#reducing-memory-overflow))

4. Activation offloading to host memory (in [Techniques for Reducing Memory to Avoid Memory Overflow and Enhance Training Efficiency](#reducing-memory-overflow))

## Sequence Packing for Performant Fine-Tuning

1. Dataset preparation

   > 1. Fine-tuning datasets with shorter sequences of variable length can be packed into longer sequences, up to a set maximum length, for best efficiency.

2. To use this feature, the microbatch size must be set to 1. In place of increasing the micro batch size, the maximum sequence length can be increased, which will effectively increase the number of individual sequences per packed sequence.

3. Enabled with:

   > 1. `FinetuningDatasetConfig.packed_sequence_specs.packed_sequence_size=<max sequence length>`
   > 2. `TrainingConfig.micro_batch_size=1`

4. Performance benefits also include:

   > 1. Inconsistent lengths between sequences in the fine-tuning dataset would reduce the computation efficiency. With a micro-batch size over 1, all sequences must be padded with empty tokens to the length of the longest one in the micro-batch. Similarly, some optimizations like CUDA graphs require uniform sequence lengths between micro-batches. Packed sequences are arranged so that the total number of tokens per packed sequence is as close to the maximum length as possible, making most processed tokens useful.
   > 2. Likewise, when using data parallel, variance in time needed to process different batches can result in all batches needing to wait for the longest to finish-- and this variance is reduced with packed sequence.

## GPU Core Clock Optimization

1. Increase the clock ratio of GPU core over off-chip memory system

   > 1. NVIDIA GPUs support a CPU core clock boost mode, which increases the core clock rate by reducing the off-chip memory clock rate. This is particularly beneficial for LLMs, which are typically compute throughput-bound.
   >
   >    > 1. `sudo nvidia-smi boost-slider --vboost 1 <run commandline>`

## Profiling Options for Analysis-based Performance Tuning

1. Nsight system profile

   > 1. Megatron-Bridge provides an interface to enable the NVIDIA Nsight Systems profiler, which displays the GPU execution trace of all CUDA streams. You can check whether communication kernels overlap with computation kernels and adjust resource allocation to balance communication and computation. The Nsight Systems profile can be enabled using ProfilingConfig, as shown below.
   > 2. `ProfilingConfig(use_nsys_profiler=True, profile_start_step=<int>, profile_end_step=<int>, profile_ranks=<[0,...]>)`

2. Memory snapshot

   > 1. Megatron-Bridge provides an interface to extract the memory snapshot that shows the memory allocation bytes, the allocation lifespan, and the function call stack. Extracting the memory snapshot can be enabled by ProfilingConfig as shown below.
   > 2. `ProfilingConfig(record_memory_history=True, memory_snapshot_path=</path/to/store/the/output/file, profile_ranks=<[0,...]>)`

## Index - List of Tuning Knobs

- `CommOverlapConfig.tp_comm_overlap`
- `CommOverlapConfig.tp_comm_overlap_cfg`
- `CUDA_DEVICE_MAX_CONNECTIONS`
- `TrainingConfig.manual_gc_interval`
- `MixedPrecisionConfig.fp8_param`
- `ProfilingConfig`
- `NCCL_NET_GDR_C2C`
- `NCCL_NET_GDR_LEVEL`
- `NCCL_NVLS_ENABLE`
- `NVTE_BWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives`
- `TransformerConfig.attention_backend`
- `AttnBackend`
- `NVTE_FWD_LAYERNORM_SM_MARGIN=<#SM for DP collectives`
- `PYTORCH_CUDA_ALLOC_CONF`
- `TrainingConfig.micro_batch_size`
- `FinetuningDatasetConfig.packed_sequence_specs.packed_sequence_size`
- `TransformerConfig.apply_rope_fusion`
- `TransformerConfig.bias_activation_fusion`
- `TransformerConfig.bias_dropout_fusion`
- `TransformerConfig.cp_comm_type`
- `TransformerConfig.cpu_offloading`
- `TransformerConfig.cpu_offloading_num_layers`
- `TransformerConfig.cpu_offloading_weights`
- `GPTProvider.cross_entropy_loss_fusion`
- `TransformerConfig.enable_cuda_graph`
- `MixedPrecisionConfig.fp8_param_gather`
- `GPTProvider.gradient_accumulation_fusion`
- `TransformerConfig.masked_softmax_fusion`
- `TransformerConfig.recompute_granuality`
- `TransformerConfig.recompute_method`
- `TransformerConfig.recompute_num_layers`
- `OptimizerConfig.use_precision_aware_optimizer`
- `GPTProvider.account_for_embedding_in_pipeline_split`
- `GPTProvider.account_for_loss_in_pipeline_split`
- `TransformerConfig.context_parallel_size`
- `DistributedDataParallelConfig.align_param_gather`
- `DistributedDataParallelConfig.bucket_size`
- `DistributedDataParallelConfig.bucket_size`
- `DistributedDataParallelConfig.data_parallel_sharding_strategy`
- `DistributedDataParallelConfig.grad_reduce_in_fp32`
- `DistributedDataParallelConfig.num_distributed_optimizer_instances`
- `DistributedDataParallelConfig.overlap_grad_reduce`
- `DistributedDataParallelConfig.overlap_param_gather`
- `T5ModelProvider.encoder_pipeline_model_parallel_size`
- `T5ModelProvider.encoder_tensor_model_parallel_size`
- `TransformerConfig.expert_model_parallel_size=<int>`
- `TransformerConfig.expert_tensor_parallel_size=<int>`
- `TransformerConfig.moe_grouped_gemm`
- `DistributedInitConfig.use_torch_fsdp2`
- `TransformerConfig.pipeline_model_parallel_size`
- `TransformerConfig.tensor_model_parallel_size`
- `TransformerConfig.virtual_pipeline_model_parallel_size`
- `OptimizerConfig.use_distributed_optimizer`
- `TORCH_NCCL_AVOID_RECORD_STREAMS`
- `TPOverlapCfg.cga_size`
- `TPOverlapCfg.fp8_buf`
- `TPOverlapCfg.num_sm`
- `TPOverlapCfg.num_split`
<!-- - `garbageCollectionCallback.gc_interval_val` -->
<!-- - `NsysPlugin` -->
