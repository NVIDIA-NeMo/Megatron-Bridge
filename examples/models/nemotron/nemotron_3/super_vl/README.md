# Nemotron 3.5 Super VL

> [!NOTE]
> This is a documentation draft for model support that is still under
> development. Complete conversion, training, and inference commands will be
> added before publication.

Nemotron 3.5 Super VL combines the Nemotron 3 Super language model, including
its MTP layer, with the Nemotron Omni RADIO vision tower and separate temporal
video embedder. The model does not include an audio encoder.

The functional recipes are defined under
[`src/megatron/bridge/recipes/nemotron_omni/`](../../../../../src/megatron/bridge/recipes/nemotron_omni/):

- `nemotron_35_super_vl_pretrain_config`
- `nemotron_35_super_vl_sft_config`
- `nemotron_35_super_vl_peft_config`
- `nemotron_35_super_vl_sft_long_context_128gpu_gb200_bf16_config`

## Fine-Tuning Freeze Controls

The model provider exposes independent controls for the language model, vision
encoder, and vision-to-language projector. They can be set in a recipe or
passed as command-line overrides.

| Setting | Component | `true` means |
| --- | --- | --- |
| `model.freeze_language_model` | Language decoder and its language-side weights | Keep the pretrained language component frozen |
| `model.freeze_vision_model` | RADIO vision tower, including its image and temporal-video embedding paths | Keep the pretrained vision component frozen |
| `model.freeze_vision_projection` | Projector from vision features to the language hidden size | Keep the multimodal projector frozen |

The standard SFT recipe intentionally uses:

```text
model.freeze_language_model=false
model.freeze_vision_model=true
model.freeze_vision_projection=false
```

This freezes the vision encoder while updating the language model and vision
projector on image-text or video-text data.

Other useful configurations are:

| Fine-tuning policy | Language model | Vision model | Vision projection |
| --- | --- | --- | --- |
| Language only | train | freeze | freeze |
| Language and projector (SFT default) | train | freeze | train |
| Projector only | freeze | freeze | train |
| Full multimodal model | train | train | train |

For example, language-only SFT can be selected without defining another model
provider:

```bash
./scripts/training/train.sh \
  --nodes <nodes> --gpus-per-node <gpus-per-node> \
  --recipe nemotron_35_super_vl_sft_config \
  --mode sft \
  model.freeze_language_model=false \
  model.freeze_vision_model=true \
  model.freeze_vision_projection=true
```

These settings control whether the pretrained base parameters receive
gradients. PEFT adapter placement is a separate concern: `peft.target_modules`
determines which matching modules receive adapters. Freezing a base component
does not, by itself, prevent an unscoped LoRA target name from adding a
trainable adapter inside that component. The H100 and GB200 PEFT recipes use
language-model-qualified target patterns, covering the decoder and MTP while
excluding `vision_model` and `vision_projection`. Custom PEFT configurations
must preserve that qualification if language-only adapters are intended.

## Draft Completion Checklist

- Add verified HF-to-Megatron and Megatron-to-HF conversion commands.
- Add deterministic image and video inference examples.
- Add SFT and PEFT dataset requirements and launch examples.
- Document dynamic-resolution image and temporal-video preprocessing.
- Link the completed page from the Nemotron 3 example index.
