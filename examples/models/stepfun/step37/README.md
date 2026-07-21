# Step 3.7 Flash Examples

Step 3.7 Flash is a multimodal LLM. Its text backbone is
byte-identical Step-3.5 (45 layers, hybrid full/sliding 64/96 GQA, MoE
288×top-8, head-wise attention gate, MTP 3 layers). The vision tower is
Perception-Encoder G/14 followed by two Conv2d downsamplers
(52×52 → 26×26 → 13×13 = 169 tokens per image) and an `align_projector`
linear (`encoder.output_dim → hidden_size`, 6144 → 4096) inside
`ImageInsertEmbedding`.

The Megatron-Bridge implementation works end-to-end as follows:

* Data side: `Step37Flickr8kSFTDataProvider` implements the Flickr8k-packed
  pipeline.
* Model side: `Step37Model.forward(images: list[ImageForInsert],
  cu_seqlens, ...)` takes pre-built image-insert payloads.

## MCore Dev Branch Requirement

Step 3.7 Flash imports require MCore changes that are not yet on a tagged release: PR [#4473](https://github.com/NVIDIA/Megatron-LM/pull/4473), PR [#4841](https://github.com/NVIDIA/Megatron-LM/pull/4841). Until these merge to Megatron-LM `main` and the bridge submodule pin advances, point `3rdparty/Megatron-LM` at the Megatron-LM `dev` branch:

```bash
./scripts/switch_mcore.sh dev
uv sync
```

Use `./scripts/switch_mcore.sh main` and `uv sync --locked` to return to the pinned main-branch submodule.

## Convert: Import HF → Megatron

```bash
./scripts/conversion/convert.sh import \
    --executor local --device gpu --gpus-per-node 4 \
    --hf-model stepfun-ai/Step-3.7-Flash \
    --megatron-path /path/to/step37_megatron_ckpt \
    --tp 1 --pp 1 --ep 4
```

## Train: `step37_flickr8k_step` (SFT)

Before launching, ensure the following environment variables are set:

1. `WORKSPACE`: base directory for checkpoints / logs
2. `HF_TOKEN`: to download the model from HF Hub (or pre-populate `HF_HOME`)
3. `HF_HOME` *(optional)*: shared cache to avoid re-downloading
4. `WANDB_API_KEY` *(optional)*: enable WandB logging

Then launches one srun task with [slurm_pretrain.sh](slurm_pretrain.sh).

```bash
sbatch --requeue --parsable slurm_pretrain.sh
```

Tune the recipe directly via CLI overrides; the script forwards everything past
`run_recipe.py` to the recipe. Typical knobs:

```bash
# Adjust PP / TP / EP / CP
model.pipeline_model_parallel_size=8
model.tensor_model_parallel_size=1
model.expert_model_parallel_size=8
model.context_parallel_size=1
model.num_layers_in_last_pipeline_stage=3

# Sequence / batch
model.seq_length=4096
train.micro_batch_size=1
train.global_batch_size=1024
```

## Convert: Export Megatron → HF

```bash
./scripts/conversion/convert.sh export \
    --executor local --device gpu --gpus-per-node 4 \
    --hf-model stepfun-ai/Step-3.7-Flash \
    --megatron-path /path/to/step37_megatron_ckpt \
    --hf-path /path/to/step37_hf_export \
    --tp 1 --pp 1 --ep 4
```
