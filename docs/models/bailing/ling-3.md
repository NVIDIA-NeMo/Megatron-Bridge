# Ling 3.0

[Ling 3.0](https://huggingface.co/collections/inclusionAI/ling-30) is a hybrid Mixture-of-Experts language model family from inclusionAI. Megatron Bridge supports the public Tiny, Tiny Base, and Flash checkpoints through the Bailing bridge with automatic Hugging Face configuration and bidirectional weight conversion.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| Ling 3.0 Tiny Base | [`inclusionAI/Ling-3.0-tiny-base`](https://huggingface.co/inclusionAI/Ling-3.0-tiny-base) | 24 logical blocks, 128 routed experts, low-rank-Q MLA, one MTP layer |
| Ling 3.0 Tiny | [`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny) | 24 logical blocks, 128 routed experts, low-rank-Q MLA |
| Ling 3.0 Flash | [`inclusionAI/Ling-3.0-flash`](https://huggingface.co/inclusionAI/Ling-3.0-flash) | 42 logical blocks, 512 routed experts, direct-Q MLA and one MTP layer |

## Architecture Notes

- The decoder alternates KDA and MLA attention with dense and MoE feed-forward stages. Each Hugging Face logical block maps to two Megatron Core `HybridModel` positions.
- The models use sigmoid top-8 expert routing, shared experts, and head-wise gated MLA output.
- Tiny and Tiny Base use low-rank-Q MLA. Tiny Base includes one low-rank-Q MLA MTP layer; the post-trained Tiny checkpoint does not include MTP weights. Flash uses direct-Q MLA and one MTP layer.
- Custom Hugging Face model code is required, so conversion commands use `--trust-remote-code`.

## Examples

For checkpoint import/export and round-trip validation, see the [Bailing examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/bailing/README.md).

## SFT Recipe

`ling_v3_tiny_base_sft_8gpu_h100_bf16_config` is an eight-GPU BF16 full-parameter SFT recipe for `Ling-3.0-tiny-base`. It reads the model architecture, including the MTP layer and its loss-scaling setting, from the public Hugging Face configuration. The recipe uses the matching public tokenizer, SQuAD prompt/completion data, offline packing, sequence length 2048, and TP=1/PP=1/EP=8/CP=1.

The 2048 sequence length is the default SFT workload, not the model's 262144-token context limit. The recipe includes SQuAD as its packed public functional default; omit `--dataset` to retain that recipe-level packing policy. Supply the local Hugging Face Base directory through `--pretrained_checkpoint`; no Python recipe changes are required:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --recipe ling_v3_tiny_base_sft_8gpu_h100_bf16_config \
    --mode sft \
    --pretrained_checkpoint /path/to/Ling-3.0-tiny-base
```

For a different supported dataset, pass `--dataset` explicitly; that dataset preset owns its own packing policy. For offline recipe construction, set `LING_V3_TINY_BASE_HF_PATH` to the same trusted local Base directory. The default public model ID and revision are used when the variable is unset.

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/bailing`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/bailing)
- Tiny Base SFT recipe: [`src/megatron/bridge/recipes/bailing/h100/ling_v3_tiny_base.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/bailing/h100/ling_v3_tiny_base.py)
- Tiny Base verification card: [`examples/model_verification_cards/ling-3.0-tiny-base/card.yaml`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/ling-3.0-tiny-base/card.yaml)
- Examples: [`examples/models/bailing`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/bailing)
