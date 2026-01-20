# Nemotron Nano V2 VL

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
[here](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html#checkpoint-contents) 
```

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF model to your desired `$MEGATRON_MODEL_PATH`, run the following command.
```bash
python examples/conversion/convert_checkpoints.py import \
--hf-model $HF_MODEL_PATH \
--megatron-path $MEGATRON_MODEL_PATH \
--trust-remote-code
```

### Export Megatron → HF
You can export a trained model with the following command.
```bash
python examples/conversion/convert_checkpoints.py export \
--hf-model $HF_MODEL_PATH \
--megatron-path <trained megatron model path> \
--hf-path <output hf model path> \
--not-strict
```

Note: it is normal to see a warning that `vision_model.radio_model.input_conditioner.norm_mean` and `vision_model.radio_model.input_conditioner.norm_std` from source are not in the exported checkpoint. These two weights are not needed in the checkpoint.


### Run In-Framework Inference on Converted Checkpoint
You can run a quick sanity check on the converted checkpoint with the following command.
```bash
python examples/conversion/hf_to_megatron_generate_vlm.py \
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
Before training, ensure the following environment variables are set.
1. `SAVE_DIR`: to specify a checkpoint and log saving directory, used in the commands below. 
2. `HF_TOKEN`: to download models from HF Hub.
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets every time.
4. `WANDB_API_KEY`: (optional) to enable WandB logging.

### Full Finetuning
Example usage for full parameter finetuning using the 
[Raven dataset](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron/viewer/raven):

```bash
torchrun --nproc-per-node=8 examples/recipes/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint <megatron model path> \
dataset.maker_name=make_raven_dataset \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name>
```

Note:
- The config file `examples/recipes/nemotron_vl/conf/nemotron_nano_v2_vl_override_example.yaml` contains a list of arguments 
  that can be overridden in the command. For example, you can set `train.global_batch_size=<batch size>` in the command. 
- To change the dataset, you only need to change `dataset.maker_name`. See the dataset section below for details.
- After training, you can run inference with `hf_to_megatron_generate_vlm.py` by supplying the trained megatron checkpoint. 
  You can also export the trained checkpoint to Hugging Face format.
- This full finetuning recipe requires at least 4xH100 (80G) GPUs.

### Parameter-Efficient Finetuning (PEFT)
Parameter-efficient finetuning (PEFT) using LoRA is supported. 
LoRA can be independently applied to the vision model, vision projection, and language model. We support two commonly used
settings out of the box in the example script:
1. Apply LoRA to the language model, and fully finetune the vision model and projection (used when the visual 
   distribution is substantially different from pretrained.)

```bash
torchrun --nproc-per-node=8 examples/recipes/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--lora-on-language-model \
dataset.maker_name=make_raven_dataset \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name> \
model.freeze_language_model=True \
model.freeze_vision_model=False \
model.freeze_vision_projection=False
```

2. Apply LoRA to all linear layers in attention and MLP modules of the vision model, vision projection, and the language model.

```bash
torchrun --nproc-per-node=8 examples/recipes/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--lora-on-language-model \
—-lora-on-vision-model \
dataset.maker_name=make_raven_dataset \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name> \
model.freeze_language_model=True \
model.freeze_vision_model=True \
model.freeze_vision_projection=True
```

These LoRA finetuning recipe requires at least 2xH100 (80G) GPUs.

A LoRA checkpoint only contains the learnable adapter weights. In order to convert the LoRA checkpoint to Hugging Face
format for downstream evaluation, it is necessary to merge the LoRA adapters back to the base model. 

```bash
python examples/peft/merge_lora.py \
--hf-model-path $HF_MODEL_PATH \
--lora-checkpoint <trained LoRA checkpoint>/iter_N \
--output <LoRA checkpoint merged>
```
You can now run in-framework inference with `hf_to_megatron_generate_vlm.py` by supplying the merged LoRA checkpoint. 
You can also export the merged LoRA checkpoint to Hugging Face format.


## Example Datasets

Megatron Bridge supports various vision-language dataset examples which can be used to finetune Nemotron Nano V2 VL:
| Dataset | Maker Name | Description |
|---------|------------|-------------|
| [cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | `make_cord_v2_dataset` | OCR receipts: Single-image-text dataset for receipt understanding, outputs xml-like annotated text. |
| [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) | `make_medpix_dataset` | Medical VQA: Single-image question-answer dataset covering clinical medical images and free-form answers. |
| [The Cauldron (Raven subset)](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | `make_raven_dataset` | Visual reasoning: Multi-image, vision reasoning dataset for analogical reasoning in different visual layouts. |
| [LLaVA-Video-178K (0_30_s_nextqa subset)](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | `make_llava_video_178k_dataset` | Video understanding: video question-answer dataset covering every-day scenarios. |

`cord-v2` is a tiny dataset and is included for demonstration only. This dataset is not recommended for PEFT tuning for this model since the XML output format interacts with the special tokens, leading to unexpected results. 

Note on video training example:
- We provide a video config yaml file instead of the default config yaml file that overwrites a few commands. Please
  pass in `--config-file "examples/recipes/nemotron_vl/conf/nemotron_nano_v2_vl_video.yaml"`.
- The LLaVA video dataset requires manual download beforehand. Please place the downloaded and extracted video files
  in a folder `VIDEO_ROOT` and pass it in to the maker with `dataset.maker_kwargs={"video_root_path":$VIDEO_ROOT}`. 
  In the nextqa subset example, `VIDEO_ROOT` should look like
  ```
  $VIDEO_ROOT/
  ├── NextQA/
  │   └── NExTVideo/
  │       └── 0000/
  │           └── 2440175990.mp4
  │       └── 0001/
  │           └── ...
  └── ...
  ```

Full video training example command:
```bash
torchrun --nproc-per-node=8 examples/recipes/nemotron_vl/finetune_nemotron_nano_v2_vl.py \
--hf-model-path $HF_MODEL_PATH \
--pretrained-checkpoint $MEGATRON_MODEL_PATH \
--config-file "examples/recipes/nemotron_vl/conf/nemotron_nano_v2_vl_video.yaml" \
logger.wandb_project=<optional wandb project name> \
logger.wandb_save_dir=$SAVE_DIR \
checkpoint.save=$SAVE_DIR/<experiment name> \
dataset.maker_kwargs={"video_root_path":$VIDEO_ROOT}
```
