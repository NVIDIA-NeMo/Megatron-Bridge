# Kimi-K2.5-VL Full-Model Guide

Step-by-step guide to run the full Kimi-K2.5-VL pipeline (conversion,
inference, comparison) using the full-size model (~1T params,
384 MoE experts, FP8 expert weights). Multi-node SLURM required.

## Prerequisites

```bash
export WORKSPACE=/your/custom/path
```

Ensure the following are available:
- `HF_TOKEN`: to download `moonshotai/Kimi-K2.5` from HuggingFace Hub
- `HF_HOME`: (optional) to cache downloaded models and datasets
- `WANDB_API_KEY`: (optional) to enable WandB logging

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion (HF → Megatron → HF)

The full model requires multi-node Slurm for conversion.

**Import** the full HF checkpoint into Megatron format with NeMo Run. This
example uses the same 96-GPU parallelism as the conversion verification job:

```bash
./scripts/conversion/convert.sh import \
    --executor slurm --device gpu \
    --nodes 12 --gpus-per-node 8 \
    --account <YOUR_ACCOUNT> --partition batch --time 4:00:00 \
    --container-image <CONTAINER_IMAGE> \
    --mount <HOST_WORKSPACE>:${WORKSPACE} \
    --mount <BRIDGE_CHECKOUT>:/opt/Megatron-Bridge \
    --env HF_TOKEN \
    --hf-model moonshotai/Kimi-K2.5 \
    --megatron-path ${WORKSPACE}/models/Kimi-K2.5-megatron \
    --tp 2 --pp 1 --ep 48
```

### Round-Trip Verification

Use [slurm_conversion.sh](slurm_conversion.sh) to run the recommended
parallelism config through `convert.sh roundtrip` and verify HF ↔ Megatron
round-trip conversion. Run it from a Slurm login node; the wrapper submits and
waits for the job by default. Validation stays in memory and does not write
another copy of the approximately 1T-parameter checkpoint:

```bash
export CONTAINER_IMAGE=/path/to/container.sqsh
export SLURM_ACCOUNT=your_account
export CONTAINER_MOUNTS=/host/workspace:${WORKSPACE}
bash examples/models/kimi/kimi_k25_vl/slurm_conversion.sh
```

The script uses 12 nodes (96 GPUs) with `TP=2`, `PP=1`, and `EP=48`.
The current checkout is mounted automatically at `/opt/Megatron-Bridge` and
must be visible from the compute nodes. Extra launcher arguments are forwarded;
for example, add `--srun-arg=--mpi=pmix` only if your cluster requires it.

## Inference

The full model requires multi-node inference. Recommended parallelism:
TP=2, EP=48, PP=1 (48 GPUs, 6 nodes).

Uses the shared VLM generation script
(`examples/conversion/hf_to_megatron_generate_vlm.py`), which auto-detects
Kimi models and handles processor patching, image-token pre-expansion for PP,
and TP sequence padding for MoE.

```bash
sbatch examples/models/kimi/kimi_k25_vl/slurm_inference.sh
```

See [slurm_inference.sh](slurm_inference.sh) for configuration details.

Note:
- `--trust_remote_code` is required for Kimi-K2.5 models.
- Use `--pp_layout` to specify custom pipeline layouts (e.g.
  `--pp_layout "Et*15|t*15|t*16|t*15L"` for PP=4).
- You can optionally pass `--megatron_model_path` to use a pre-converted
  checkpoint (faster startup).

### Expected Inference Output

With the [Qwen-VL demo image](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg)
and prompt `"Describe this image."`, the model enters `<think>` reasoning mode
before producing the final answer. The first 100 generated tokens look like:

```
<think>The user wants me to describe the image. Let me analyze what I see in the image:

1. **Setting**: A beach scene during what appears to be sunset or sunrise
   (golden hour lighting). The ocean is visible in the background with waves.

2. **Main subjects**:
   - A woman sitting on the sand
   - A large dog (looks like a yellow Labrador or Golden Retriever)

3. **The woman**:
   - Long dark hair
```

The model correctly identifies the beach scene, golden hour lighting, the
woman, and the dog breed. Kimi-K2.5 is a thinking model, so the initial output
is always the internal `<think>` reasoning chain before the final response.
