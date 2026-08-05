# ERNIE 4.5

Megatron Bridge registers Hugging Face conversion support for the text-only
`baidu/ERNIE-4.5-0.3B-PT` MoE model and the
`baidu/ERNIE-4.5-VL-28B-A3B-Instruct` and `baidu/ERNIE-4.5-VL-28B-A3B-Thinking`
vision-language MoE models.

Import a text checkpoint with the shared conversion CLI:

```bash
./scripts/conversion/convert.sh import \
  --hf-model baidu/ERNIE-4.5-0.3B-PT \
  --megatron-path /workspace/models/ernie-4.5-0.3b \
  --trust-remote-code
```

For ERNIE 4.5 VL import, export, and inference commands, see the
[checked-in examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/vlm/ernie_vl).
The VL bridge supports its modality-specific dual-pool MoE layout; use the
parallelism settings documented with those examples.
