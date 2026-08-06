# HY V3

Megatron Bridge registers Hugging Face checkpoint conversion for the
`tencent/Hy3-preview-Base` MoE causal language model. The bridge maps HY V3's
GQA with QK layer normalization, dense-first layer schedule, sigmoid router,
shared experts, and grouped-GEMM routed experts to Megatron Core.

Import the checkpoint with the shared conversion CLI:

```bash
./scripts/conversion/convert.sh import \
  --hf-model tencent/Hy3-preview-Base \
  --megatron-path /workspace/models/hy3-preview-base \
  --trust-remote-code
```

Export a converted checkpoint back to Hugging Face format:

```bash
./scripts/conversion/convert.sh export \
  --hf-model tencent/Hy3-preview-Base \
  --megatron-path /workspace/models/hy3-preview-base/iter_0000000 \
  --hf-path /workspace/models/hy3-preview-base-hf \
  --trust-remote-code
```

No library training recipe is currently exported for HY V3.
