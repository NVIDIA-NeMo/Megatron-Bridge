# Changelog

## NVIDIA Megatron-Bridge 0.1.0rc3

* Model Collection Support
  * Llama
  * Qwen 2, Qwen 3, Qwen 3 MoE
  * DeepSeek
  * Mamba
* [Migration guide from NeMo 2 to Megatron-Bridge](https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/nemo2-migration-guide.html)
* [Contribution guide for adding a new model](https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/adding-new-models.html)
* [Checkpoint conversion from Hugging Face to Megatron](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/conversion)
* [Performance](https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/performance-summary.html)
  * MoE LLM
    * Change the model to dropless with balanced gating
    * Fusion of operators in router function
    * Global permutation fusion with A2A dispatcher
    * EP A2A communication overlap with computation in both 1F1B pipelining and non-pipelined training
    * Precision-aware optimizer update to support BF16 states
  * Megatron FSDP
    * Migration from mcore FSDP to megatron FSDP
    * Fusion of weight gradient copy to reduce-scatter communication buffer to WGRAD GEMM
    * Removed redundant optimizer operations
    * Use Zero1 (opt and master param sharding) in the replica domain of hybrid FSDP to further lower memory usage
    * IB-SHARP support for the IB AllReduce of hybrid FSDP in a patch with NCCL2.28
  * MXFP8
    * Improved act grad all-gather overlap performance via userbuffer
    * Parameter all-gather overlap with computation while the communication buffer sharing with reduce-scatter
    * Fusion of MXFP8 scaling factor swizzling kernels
    * Use PDL (Programmatic Dependent Launch) for quantization kernels to lower CPU overhead
  * Others
    * Full iteration cuda graph for dense model without pipelining
    * Fusion of activation and cast fusion (currently tensor-wise scaling only)
    * Store SwiGLU input in FP8 to save activation memory

## NVIDIA Megatron-Bridge 0.1.0a0

* Llama and Qwen
* Pretrain/SFT
* PeFT  
* Recipe structure with examples for plain python & NeMo Run usage
