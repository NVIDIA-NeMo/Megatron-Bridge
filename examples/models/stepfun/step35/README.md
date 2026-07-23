# Step-3.5-Flash Examples

This directory contains example scripts for the Step-3.5-Flash MoE language model
(`stepfun-ai/Step-3.5-Flash`). The model uses a hybrid attention pattern (full
attention interleaved with sliding attention), a fused per-head attention gate
(`g_proj` merged into `linear_qkv`), MoE layers (3-44) with dense layers at the
top/bottom (0-2 and MTP layers 45-47), and Multi-Token Prediction.

## MCore Dev Branch Requirement

Step-3.5-Flash imports require MCore changes that are not yet on a tagged release: PR [#4473](https://github.com/NVIDIA/Megatron-LM/pull/4473), PR [#4841](https://github.com/NVIDIA/Megatron-LM/pull/4841). Until these merge to Megatron-LM `main` and the bridge submodule pin advances, point `3rdparty/Megatron-LM` at the Megatron-LM `dev` branch:

```bash
./scripts/switch_mcore.sh dev
uv sync
```

Use `./scripts/switch_mcore.sh main` and `uv sync --locked` to return to the pinned main-branch submodule.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for
checkpoints and results. By default, this is set to `/workspace`. You can override
it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

### Import HF → Megatron

The Step-3.5 bridge fuses the per-head `g_proj` rows into `linear_qkv.weight`
(see `QKVGMapping`), expands the per-layer `rotary_base_per_layer` list to
`num_layers + mtp_num_layers`, and surfaces `layer_types` /
`attention_other_setting` / `sliding_attention_setting` so that the custom
`Step35DecoderLayer` can size sliding-attention layers (96 heads, 8 KV groups)
differently from full-attention layers (64 heads).

To import the HF checkpoint to a Megatron path:

```bash
./scripts/conversion/convert.sh import \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${MEGATRON_CKPT_PATH}"
```

See [conversion.sh](conversion.sh) for a complete single-process CPU import example with
logging redirected to `${WORKSPACE}/logs/`.

### Export Megatron → HF

```bash
./scripts/conversion/convert.sh export \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${MEGATRON_CKPT_PATH}" \
    --hf-path "${MEGATRON_CKPT_PATH}-hf-export"
```

## Inference

[inference.sh](inference.sh) runs greedy generation directly with
`torch.distributed.run` on 1 node / 8 GPUs with `TP=1`, `PP=1`, and `EP=8`.
Run it from an interactive 8-GPU allocation or equivalent single-node
environment:

```bash
bash examples/models/stepfun/step35/inference.sh
```

By default it loads `stepfun-ai/Step-3.5-Flash` and converts in memory. To
generate from a checkpoint produced by [conversion.sh](conversion.sh), pass:

```bash
MEGATRON_MODEL_PATH="${MEGATRON_CKPT_PATH}/iter_0000000" \
    bash examples/models/stepfun/step35/inference.sh
```

### Expected output

The following smoke-test output was produced with `TP=1`, `PP=1`, `EP=8`, and
`MAX_NEW_TOKENS=4`:

```text
======== GENERATED TEXT OUTPUT ========
Prompt: Write one concise sentence about Megatron Bridge.
Generated: <｜begin▁of▁sentence｜>Write one concise sentence about Megatron Bridge. Write one concise sentence
=======================================
```

## Pretraining / Resume

The recipe `step35_196b_a11b_pretrain_config` (in
`src/megatron/bridge/recipes/stepfun/step35.py`) ships with TP=1, PP=8, CP=8, EP=8
and inherits the published checkpoint's MTP layer count. It is meant as a
**resume / continued pretraining** entry point; `cfg.dataset.blend = None`
uses mock data by default. Set a real dataset blend before running real
pretraining.

Before launching, ensure the following environment variables are set:

1. `WORKSPACE`: base directory for checkpoints / logs
2. `HF_TOKEN`: to download the model from HF Hub (or pre-populate `HF_HOME`)
3. `HF_HOME` *(optional)*: shared cache to avoid re-downloading
4. `WANDB_API_KEY` *(optional)*: enable WandB logging

### Multi-Node Pretrain via Slurm (PP=8, EP=8, 8 nodes, 64 gpus)

[slurm_pretrain.sh](slurm_pretrain.sh)

* launches one srun task per GPU and translates Slurm env vars to
  `torch.distributed` env vars inside the container,
* overrides the recipe to PP=8 with the tail stages thinned out via
  `model.num_layers_in_last_pipeline_stage=3`,
* turns on the Step-3.5-Flash sliding-attention skip pattern
  (`model.window_size=[512,0]` and the `model.window_attn_skip_freq` mask
  matching `layer_types`), and
* loads the pre-converted Megatron checkpoint as a fine-tune source
  (`checkpoint.finetune=true`, `load_optim=false`, `load_rng=false`,
  `exit_on_missing_checkpoint=true`).

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

### Hybrid Attention (Full / Sliding)

`Step35Bridge.provider_bridge` populates `provider.layer_types` from the HF
`layer_types` list and writes the sliding-layer shape overrides into
`provider.sliding_attention_setting`:

```python
provider.sliding_attention_setting = {
    "rotary_percent": 1.0,
    "num_attention_heads": 96,
    "num_query_groups": 8,
    "head_dim": 128,
}
```

`Step35DecoderLayer` deep-copies the config and overrides
`rotary_percent` / `num_attention_heads` / `num_query_groups` / `kv_channels`
when the resolved global layer index is marked `"sliding_attention"` in
`layer_types`. The training-time sliding-window mask is controlled separately
through `model.window_size` and `model.window_attn_skip_freq` (see the
slurm_pretrain.sh defaults).
