# ERNIE 5.5

**ERNIE 5.5 Flash** is an efficiency-focused model in the ERNIE 5.5 family: a sparse Mixture-of-Experts (MoE) language model with **280B total parameters and 14B actived parameters per token**, pre-trained from scratch on **33.5T tokens** with a training context of up to **128K**; fine-tuning and inference support a **1M-token** context. Different domains, reasoning tasks and agentic applications call for different data, training objectives and behavioral policies; by releasing the weights at the end of pre-training, we let developers build on the model's general capabilities and choose for themselves where to take it next. Because further training and agentic use keep adding to compute and memory cost, we made efficiency a key design goal from the start, revisiting each component, from attention and residual connections to the MoE layers and the output head, and treating activated compute, KV cache and long-context cost as part of one overall design. The same long-term view shaped the data and the training system: pre-training data emphasizes capabilities that are hard to build in afterwards, such as repository-scale code and long-context understanding, and the training system was co-designed with the architecture so that its efficiency choices also train efficiently at scale.

**ERNIE-5.5-Flash-Base** is the checkpoint at the end of pre-training, before any mid- or post-training, and can serve as a common starting point for domain adaptation, instruction tuning and reinforcement-learning research. Under a unified evaluation pipeline it delivers strong results across knowledge, reasoning, math, code, multilingual and long-context benchmarks.

Megatron Bridge supports ERNIE 5.5 (`model_type: ernie5_5_moe`) with Hugging Face ↔ Megatron checkpoint conversion
and supervised fine-tuning on NVIDIA GB300. For requirements, data preparation, training and export to Hugging Face
format, see the
[ERNIE 5.5 examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/ernie55/README.md).
The examples also cover a fine-tuning recipe for NVIDIA GB200.

## Performance Options (Opt-In)

The recipes leave the following model options off. Each one keeps the training math unless noted, up to
floating-point summation order: `ernie_fused_mhc`, `ernie_muon_ns_batch_size` above 1 and `ernie_vha_batched_mix`
are not bitwise identical to the default path, and `ernie_cp_key_gather_inverse_grad` keeps `-0.0` (and, on CUDA,
subnormal) key-gradient elements that the default path turns into `+0.0`. Set them on the command line, for example
`model.ernie_vha_batched_mix=true`. The config docstrings in `src/megatron/bridge/models/ernie/ernie55_config.py` give
the details and requirements.

- `ernie_fused_mhc`: fused fp32 Triton kernels for the hyper-connection (mHC) mappings and mixing.
- `cuda_graph_impl=transformer_engine` with `cuda_graph_modules=["attn"]`: Transformer Engine CUDA graphs of the
  H attention layers. `ernie_graph_d_layers` and `ernie_graph_mtp_layers` also graph the D attention layers of the
  decoder and of the MTP depth (one document per row, `thd_max_packed_sequences=1`).
- `ernie_muon_ns_batch_size`: batch Muon's Newton-Schulz over up to this many same-shape blocks (1 keeps
  Megatron-Core's Muon).
- `ernie_cp_key_gather_inverse_grad`: under context parallelism, gather the key gradients by the inverse
  permutation instead of `index_select`'s backward.
- `ernie_reuse_checkpoint_topk`: for packed D layers trained with the cuDNN indexer, context parallelism and full
  recomputation, retain the first forward's selected key IDs, lengths and FP32 logits on that
  particular checkpoint. The replay runs the original top-k-only sparse KL and its gradients
  over the saved choice. Other layers and microbatches have independent selections.
- `ernie_overlap_latent_kv_gather`: on D layers with context parallelism, issue the latent KV all-gather
  after the indexer K gather. The remote top-k selection can run while latent KV is in flight. The
  top-k-only teacher loss and the main sparse attention wait before they read the gathered rows.
  With `ernie_dsa_contiguous_backward`, the gradient-enabled replay uses the gather that preserves
  contiguous gradient ownership; the original checkpoint forward can still overlap its gather.
- `ernie_vha_batched_mix`: apply the VHA head mix as one strided-batched GEMM.
- `ernie_fused_situ_prob`: fuse the routed SiTU-GLU with the routing-probability multiply (needs
  `ernie_fused_ops` and Megatron-Core dev).
- `ernie_router_force_balance`: benchmark only. It changes the training math: the routers pick uniformly random
  experts, as Megatron-Core's `moe_router_force_load_balancing` does.
- `ERNIE55_LOG_EXPERT_LOAD=1` (environment variable): diagnostic only. It writes the tokens each rank's experts
  receive, per MoE layer and micro-batch, to JSON files (see `modeling_ernie55/expert_load.py`).

Checkpoint import also reads the block-scaled FP8 release of ERNIE-5.5-Flash-Base and dequantizes it on load.
