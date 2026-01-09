# Changelog

## NVIDIA Megatron-Bridge 0.2.2

* This release addresses known security issues. For the latest NVIDIA Vulnerability Disclosure Information visit <https://www.nvidia.com/en-us/security/>, for acknowledgement please reach out to the NVIDIA PSIRT team at <PSIRT@nvidia.com>

## NVIDIA Megatron-Bridge 0.2.1

* Performance  
  * Activation offloading to host memory support with pipelining  
    * Supports the high activation memory needs of MoE models training with dynamic shapes  
    * Fixed Nemotron FLOPS calculation model  
* Model Collection Support  
  * Ministral 3  
* Enhanced LoRA support  
  * LoRA support for Mamba layers (for Nemotron Nano V2 and NemotronH finetuning)

## NVIDIA Megatron-Bridge 0.2.0

* [Model Collection Support](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models)

  * LLM
    * HuggingFace Conversion + training recipes:
      * GPT-OSS
      * Qwen3 Next
      * Nemotron-H
      * Nemotron Nano v2
      * Moonlight
      * OlMoE
      * GLM 4.5
      * Gemma 3
    * HuggingFace conversion support:
      * Llama Nemotron
      * Mistral
      * Gemma
      * Gemma 2
  * VLM
    * Nemotron Nano v2 VL
    * Qwen 3 VL
    * Qwen2.5 VL
    * Gemma3 VL

* [Performance](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance)
  * Megatron-Bridge support for new benchmarks
      * Benchmarks (same workloads as GB200 system) for GB300 system
      * GPT-OSS 120B
      * Qwen3-Next 80B_A3B
      * Support for linear attention on Blackwell - Gated Delta Networks
      * Pre-training with NVFP4 precision: Llama3 8B, Lama3 70B, Llama3.1 405B
  * Megatron-Bridge support for benchmarks previously existing only for NeMo 2.0
    * Nemotron-H 56B
    * Fine-tuning (SFT and LoRA): Llama3 8B and Llama3 70B
  * HybridEP: DeepSeek V3 benchmarks on GB200 and GB300 systems now use HybridEP
  * CUDA Graphs
    * Full-model iteration CUDA graph used for dense models- Llama3 8B, Llama3 70B, Llama3.1 405B
    * Fine-grained Transformer component specific CUDA Graphs used for MoE models

* [NVIDIA Model Optimization Integration](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/quantization)
  * Knowledge Distillation
  * Post training quantization export
  * Quantization aware training

* [Enhanced LoRA support](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/peft)
  * Support for expert layers
  * Supported merging adapters for export to HuggingFace @HollowMan6

* [Finetuning dataset improvements: OpenAI messages format conversion, chat template support](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/data)
* [Integration with Tensor NVIDIA-DLFW-Inspect for tensor statistic collection & monitoring](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/tensor_inspect.py)
* [Support for sample-based training](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/config.py)
* Broader Community Adoption: Integrate the Megatron-Bridge into the training pipelines of VeRL ([PR](https://github.com/volcengine/verl/pull/4063/files)), Slime ([PR](https://github.com/THUDM/slime/pull/894/)), and Sky-RL ([PR](https://github.com/NovaSky-AI/SkyRL/pull/453)).
* Special thanks to the community contributors for this release: @HollowMan6, @fzyzcjy, @erictang000, @hawkoli1987.

## NVIDIA Megatron-Bridge 0.1.0rc4

* Fix docs build
* Update performance scripts

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
