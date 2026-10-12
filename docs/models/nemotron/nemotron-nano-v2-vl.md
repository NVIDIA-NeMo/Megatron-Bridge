# Nemotron Nano V2 VL

> **Deprecation notice:** Nemotron Nano v2 VL 12B support is no longer actively
> maintained or tested against current upstream checkpoints and will be removed
> in Megatron Bridge 0.7.0.

NVIDIA Nemotron Nano v2 VL is an open 12B multimodal reasoning model for document intelligence and video understanding.
It enables [AI assistants](https://www.nvidia.com/en-us/use-cases/ai-assistants) to extract, interpret, and act on
information across text, images, tables, and videos. This makes the model valuable for agents focused on data analysis,
document processing and visual understanding in applications like generating reports, curating videos, and dense
captioning for media asset management, and retrieval-augmented search.

NeMo Megatron Bridge supports finetuning this model (including LoRA finetuning) on single-image, multi-image, and video
datasets.
The finetuned model can be converted back to the 🤗 Hugging Face format for downstream evaluation.

```{important}
Please use the custom container `nvcr.io/nvidia/nemo:25.09.nemotron_nano_v2_vl` when working with this model.

Run all commands from `/opt/Megatron-Bridge` (e.g. `docker run -w /opt/Megatron-Bridge ...`)
```

```{tip}
We use the following environment variables throughout this page
- `HF_MODEL_PATH=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`
- `MEGATRON_MODEL_PATH=/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` (feel free to set your own path)

Unless explicitly stated, any megatron model path in the commands below should NOT contain the iteration number
`iter_xxxxxx`. For more details on checkpointing, please see
[here](../../training/checkpointing.md#checkpoint-contents)
```

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF model to your desired `$MEGATRON_MODEL_PATH`, run the following command.
```bash
./scripts/conversion/convert.sh import \
--hf-model $HF_MODEL_PATH \
--megatron-path $MEGATRON_MODEL_PATH \
--trust-remote-code
```

### Export Megatron → HF
You can export a trained model with the following command.
```bash
./scripts/conversion/convert.sh export \
--hf-model $HF_MODEL_PATH \
--megatron-path <trained megatron model path> \
--hf-path <output hf model path> \
--not-strict
```

Note: it is normal to see a warning that `vision_model.radio_model.input_conditioner.norm_mean` and `vision_model.radio_model.input_conditioner.norm_std` from source are not in the exported checkpoint. These two weights are not needed in the checkpoint.


### Run In-Framework Inference on Converted Checkpoint
You can run a quick sanity check on the converted checkpoint with the following command.
```bash
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
--hf_model_path $HF_MODEL_PATH \
--megatron_model_path $MEGATRON_MODEL_PATH \
--image_path <example image path> \
--prompt "Describe this image." \
--max_new_tokens 100 \
--use_llava_model
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model and then run forward. If
  specified, the script will just load the megatron model
- `--max_new_tokens` controls the number of tokens to generate.
- For inference with multiple images, pass in a comma-separated list, e.g.
  `--image_path="/path/to/example1.jpeg,/path/to/example2.jpeg"`.
  Use a suitable prompt, e.g. `--prompt="Describe the two images in detail."`.
- For inference with video, pass in video path instead, e.g. `--video_path="/path/to/demo.mp4"`. Use a suitable prompt,
  e.g. `--prompt="Describe what you see."`.


## Finetuning Recipes

The deprecated Nano v2 VL example script and its YAML override files have
been removed from `main`. The public recipe entry points are unchanged:

```python
from megatron.bridge.recipes.nemotron_vl import (
    nemotron_nano_v2_vl_12b_sft_config,
    nemotron_nano_v2_vl_12b_peft_config,
)
```

For the historical full-finetuning, component-specific LoRA, and video
examples, use the [Bridge 0.6.0 guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/v0.6.0/docs/models/nemotron/nemotron-nano-v2-vl.md#finetuning-recipes)
with a `v0.6.0` checkout. Those commands and YAML files belong to that release
and are no longer available on `main`.
