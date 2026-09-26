# Inkling

The Inkling bridge converts the text backbone of the unquantized
[`thinkingmachines/Inkling-Small`](https://huggingface.co/thinkingmachines/Inkling-Small)
checkpoint (276B total / 12B active parameters). It uses native Megatron tensor
and expert parallel projections, Transformer Engine normalization and grouped
experts, and PyTorch FlexAttention for the learned relative-position bias.

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("/shared/models/Inkling-Small")
provider = bridge.to_megatron_provider()
provider.expert_model_parallel_size = 8
provider.expert_tensor_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

Download a fixed checkpoint revision to the shared path before launching all
ranks. Conversion reads the published `model.llm.*` safetensors layout directly.
Transformers' materialized model uses different names and is not a conversion
source. Dense, routed, and shared `w13` weights contain interleaved gate/up rows.
Full checkpoint export preserves the source audio, vision, and auxiliary MTP
tensors unchanged; these components are not trained.

For attention-only LoRA, target `linear_qkv`, `linear_r`, and `linear_proj`.
Native adapter export produces Inkling's published `wq_du`, `wk_dv`, `wv_dv`,
`wr_du`, and `wo_ud` names. No downstream adapter-name translation is required.

Supported execution is unpacked text training and full-sequence scoring with
zero dropout and context parallel size 1. Packed sequences, cached generation,
CUDA graphs, quantized checkpoint import, and multimodal/MTP training are not
implemented. FlexAttention backward through the relative bias is tested on
PyTorch 2.13; older releases have not been verified. Use a serving engine for cached
generation.

The recorded two-layer FP32 functional suite passed at TP1/EP1, TP2/EP1, and TP1/EP2
on H200 with full-layer recomputation. It checks all 42 checkpoint tensors,
forward/backward parity against Transformers, nonzero LoRA export (20 tensors),
merged export, and native base/adapter checkpoint restore. Operator tests also
compare relative-attention and convolution gradients against Transformers.
The BF16 dense-activation test checks native fused SwiGLU outputs and gradients
against FP32 activation followed by one output cast.
FP32 internal router/convolution weights retain the imported BF16 values but
export in FP32 unless an export dtype is requested.

Full Inkling-Small integration with SkyRL completed ten LoRA updates over 5,120
trajectories on 16 H200s (eight learner, eight sampler), with checkpoints saved
at steps 0, 5, and 10. On the same 100 heldout tasks, mean reward was
0.31 / 0.38 / 0.43 and full successes were 23 / 32 / 38 at those steps.
These are descriptive results from one run. Separate full-model checks covered
433-position trainer/sampler update and restore agreement and one sequence
with 32,768 supervised positions.

The full-model runs used Bridge `7cd9a887` and SkyRL `99bf425d`; later review
revisions have not been rerun on GPUs. The tiny-model tests exercise conversion
and numerical behavior separately from these full-model integration tests.
See the [validation results and reproducible plots](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6170)
for source revisions, test scope, and remaining CI coverage.

Related implementations include Miles' [Inkling model](https://github.com/radixark/miles/pull/1683)
and [native LoRA](https://github.com/radixark/miles/pull/2122) support. Those use
ISEEKYAN/mbridge and SGLang; this integration uses NVIDIA Megatron-Bridge's
conversion and PEFT interfaces.
