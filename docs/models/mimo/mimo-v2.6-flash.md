# MiMo-V2.6-Flash

MiMo-V2.6-Flash (Xiaomi) is a 309B-total / ~15B-active sparse MoE model. Its language model is the same as
MiMo-V2-Flash: hybrid full and sliding-window attention, 256 routed experts with top-8 routing, asymmetric head
dimensions (`head_dim=192` for Q and K, `v_head_dim=128` for V), dual-base RoPE and Multi-Token Prediction (MTP).
The checkpoint also contains vision, audio and speech-embedding weights.

## Supported Variants

Megatron Bridge supports Hugging Face checkpoints with the `MiMoV2ForCausalLM` architecture and the `mimo_v2`
model type, such as [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), through
`MiMoV26Bridge`. The bridge converts the language model, MTP layers included. The checkpoint depends on custom
modeling code, so pass `--trust-remote-code`.

## Architecture Notes

`MiMoV26Bridge` extends `MiMoV2FlashBridge` and reuses its provider, attention, MoE and MTP mappings. Only the
checkpoint format differs from MiMo-V2-Flash:

- **Fused QKV.** `self_attn.qkv_proj` is one FP8 tensor stored as `num_key_value_heads` slices of
  `[Q | K | V]`, and the 128x128 block scales restart at each slice. Import dequantizes each slice with its own
  scales and splits it into Q, K and V. MTP layers use the sliding-window attention sizes.
- **MXFP4 experts.** Routed experts store two E2M1 values per byte with one E8M0 scale per 32 values. Import
  decodes them to BF16 exactly.
- **Other FP8 weights.** The dense and MTP MLPs use 128x128 block-scaled FP8, as in MiMo-V2-Flash.
- **Export in the source format.** Weights that are MXFP4 or FP8 in the source checkpoint are requantized with its
  scale layout, and `qkv_proj` is rebuilt slice by slice. The vision, audio and speech-embedding tensors are copied
  unchanged from the source checkpoint, and the source `quantization_config` is kept.

## Conversion

Import and export commands for the published checkpoint, the hardware requirements and the expected export
accuracy are in the
[MiMo-V2-Flash examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/mimo_v2_flash/README.md#mimo-v26-flash).

## Training

Training recipes (SFT, PEFT) are not provided. Training with an imported checkpoint uses the MiMo-V2-Flash provider
and model.

## Known Limitations

- The vision, audio and speech-embedding weights are not converted. Export copies them from the source checkpoint,
  so it needs that checkpoint (`--hf-model`).
- Export requantizes to the source format and does not support `--export-weight-dtype`.
- Requantization keeps MXFP4 values and FP8 codes exactly. FP8 scales are recomputed from BF16 weights, so decoded
  FP8 values can differ by up to about 0.4% of a tensor's largest value.
- As for MiMo-V2-Flash, context parallelism is not supported, and the tensor-parallel size must be at most
  `min(num_key_value_heads, swa_num_key_value_heads)`.

## Related Implementation

- Bridge: [`src/megatron/bridge/models/mimo_v2_flash/mimo_v26_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/mimo_v2_flash/mimo_v26_bridge.py)
- MiMo-V2-Flash bridge and provider: [`src/megatron/bridge/models/mimo_v2_flash`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/mimo_v2_flash)
