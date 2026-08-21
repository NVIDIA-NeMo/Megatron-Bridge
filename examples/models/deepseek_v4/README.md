# DeepSeek V4

Megatron Bridge supports checkpoint conversion and training for the DeepSeek V4
family. See the
[DeepSeek-V4-Flash verification card](../../model_verification_cards/deepseek-v4-flash/card.yaml)
for the exact commands, revisions, metrics, and current verification status.

## Supported Variants

All variants use the same bridge. Import dequantizes the published checkpoint to
the requested Megatron training dtype.

| Variant | Hugging Face model | Published weight format |
|---------|--------------------|-------------------------|
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash` | FP8 attention and MXFP4 experts |
| DeepSeek-V4-Flash-Base | `deepseek-ai/DeepSeek-V4-Flash-Base` | FP8 with float32 scales |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` | FP8 attention and MXFP4 experts |
| DeepSeek-V4-Pro-Base | `deepseek-ai/DeepSeek-V4-Pro-Base` | FP8 with float32 scales |

## Runtime Requirements

DeepSeek V4 training requires a compatible Megatron-LM `dev` revision; the
Megatron-LM revision pinned for the Megatron Bridge `main` branch is not a
supported training runtime. Switch the submodule before syncing the environment:

```bash
./scripts/switch_mcore.sh dev
uv sync
```

Grouped-FP8 checkpoint saves also require the stateless grouped extra-state fix
merged in Megatron-LM
[#5997](https://github.com/NVIDIA/Megatron-LM/pull/5997). If the selected `dev`
revision does not contain that fix yet, apply the equivalent signed commit
`d05cb07ff618` from
[#6509](https://github.com/NVIDIA/Megatron-LM/pull/6509) to the compatible `dev`
checkout before enabling checkpoint saves.

`fast-hadamard-transform` is required by DSA and is installed from the pinned
source dependency by `uv sync`. Run the examples in a CUDA-enabled Megatron
Bridge container; see the
[repository installation instructions](../../../README.md#-installation).

The Slurm launchers require an account, partition, and container image. Set
them before running any `--executor slurm` conversion or `train.sh` command:

```bash
export SLURM_ACCOUNT=your-account
export SLURM_PARTITION=your-partition
export CONTAINER_IMAGE=your-container-image
```

Pass `--mount HOST` or `--mount HOST:CONTAINER` for each dataset, checkpoint,
or output path that is not already visible inside the container.

## Conversion

The shared conversion CLI performs dtype-aware FP8/MXFP4 dequantization during
import; no preprocessing script is required. A DeepSeek-V4-Flash GPU import is:

```bash
./scripts/conversion/convert.sh import \
  --executor local --device gpu --gpus-per-node 4 \
  --hf-model deepseek-ai/DeepSeek-V4-Flash \
  --megatron-path work/models/deepseek-v4-flash \
  --tp 1 --pp 1 --ep 4 \
  --torch-dtype bfloat16 --trust-remote-code
```

Export accepts a different parallel layout because distributed checkpoints can
be resharded while loading:

```bash
./scripts/conversion/convert.sh export \
  --executor slurm --device gpu --nodes 8 --gpus-per-node 4 \
  --hf-model deepseek-ai/DeepSeek-V4-Flash \
  --megatron-path work/models/deepseek-v4-flash/iter_0000000 \
  --hf-path work/models/deepseek-v4-flash-hf \
  --tp 1 --pp 4 --ep 8 \
  --torch-dtype bfloat16 --export-weight-dtype bfloat16 \
  --trust-remote-code
```

GPU import is verified. GPU export, CPU conversion, and manual HF/Megatron
forward correlation remain unverified at the revision recorded in the card.
Do not infer verification from the presence of a command.

[`conversion.sh`](conversion.sh) is a convenience wrapper for import, export,
and optional round-trip checks. Set `MODEL_VARIANT`, `WORKSPACE`, and the
parallelism and executor variables documented in its header before running it.

## Pretraining

Use the public [`scripts/training/train.sh`](../../../scripts/training/train.sh)
launcher. Hardware-qualified library recipes are defined under
[`recipes/deepseek`](../../../src/megatron/bridge/recipes/deepseek):

| Hardware | Recipe | Precision and optimizer |
|----------|--------|-------------------------|
| H100, 32 GPUs | `deepseek_v4_flash_pretrain_32gpu_h100_bf16_config` | BF16 Adam |
| H100, 32 GPUs | `deepseek_v4_flash_pretrain_32gpu_h100_fp8mx_config` | MXFP8 Adam |
| H100, 32 GPUs | `deepseek_v4_flash_pretrain_32gpu_h100_bf16_muon_config` | BF16 Muon |
| GB200, 64 GPUs | `deepseek_v4_flash_pretrain_64gpu_gb200_bf16_config` | BF16 Adam |
| GB200, 64 GPUs | `deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config` | MXFP8 Adam |
| GB200, 64 GPUs | `deepseek_v4_flash_pretrain_64gpu_gb200_bf16_muon_config` | BF16 Muon |
| GB200, 128 GPUs | `deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_library_config` | MXFP8 Adam |
| GB300, 32 GPUs | `deepseek_v4_pro_pretrain_32gpu_gb300_bf16_config` | BF16 Adam |
| GB300, 32 GPUs | `deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config` | MXFP8 Adam |

For example, a short generated-data run of the 128-GPU GB200 library recipe is:

```bash
./scripts/training/train.sh --nodes 32 --gpus-per-node 4 \
  --recipe deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_library_config \
  --mode pretrain --dataset mock --max_steps 10
```

That recipe owns its TP/PP/EP/CP topology, global batch size, natural-routing
semantics, recompute, offload, and checkpoint settings. Keep those defaults for
the qualified configuration and override only the dataset, run length, logging,
and output paths. Its 100-step real-data validation and checkpoint evidence are
recorded in the verification card.

Compatibility aliases such as `deepseek_v4_flash_pretrain_mxfp8_config` remain
exported, but new launches should use the hardware-qualified names above.
Canonical benchmark recipes under `src/megatron/bridge/perf_recipes/` are
performance references, not substitutes for the library recipes.

## Supervised Fine-Tuning

DeepSeek-V4-Flash provides BF16 Adam full-parameter SFT recipes:

| Recipe | Sequence format | MTP | Target |
|--------|-----------------|-----|--------|
| `deepseek_v4_flash_sft_config` | Unpacked SBHD | On | Hopper or Blackwell |
| `deepseek_v4_flash_no_mtp_sft_config` | Unpacked SBHD | Off | Hopper or Blackwell |
| `deepseek_v4_flash_sft_openmath_thinking_packed_config` | Offline-packed THD | On | Portable base |
| `deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config` | Offline-packed THD | On | 32-GPU GB200 |

The recipes select fused mHC only when the runtime supports the Blackwell
kernel; Hopper uses the unfused fallback. The GB200 packed recipe additionally
enables HybridEP, uneven-dispatch padding, DSA fusion, grouped GEMM, selective
recompute, and attention activation offload.

Launch the verified GB200 packed recipe from an imported BF16 checkpoint:

```bash
./scripts/training/train.sh --nodes 8 --gpus-per-node 4 \
  --recipe deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config \
  --mode sft --step-func dsv4_step \
  --pretrained_checkpoint work/models/deepseek-v4-flash \
  --save_dir work/results/deepseek-v4-flash-sft \
  --save_interval 100 --max_steps 100 --seq_length 1024 \
  dist.distributed_timeout_minutes=180
```

The card records 100 finite steps and a fresh-process checkpoint reload for
this configuration at CP=1. Long-context and CP=2 SFT, post-SFT export and HF
inference, and PEFT remain unverified or unsupported as indicated there. MXFP8
and Muon SFT recipes are intentionally not shipped because full-model tests did
not establish a stable supported configuration.

## Legacy Slurm Templates

[`slurm_pretrain.sh`](slurm_pretrain.sh) and
[`slurm_sft.sh`](slurm_sft.sh) are customizable 32-GPU site templates retained
for compatibility. They call the lower-level `run_recipe.py` entry point and
do not reproduce the current hardware-qualified validation commands.
`slurm_sft.sh` uses the unpacked SFT recipes; it does not launch the GB200 packed
recipe. Prefer `scripts/training/train.sh` for new runs.

## Storage

DeepSeek-V4-Flash materializes approximately 570 GB of BF16 model weights.
Plan persistent storage before importing or saving checkpoints:

| Artifact | Approximate size |
|----------|------------------|
| Quantized Hugging Face cache | 150-200 GB |
| Imported BF16 Megatron checkpoint | 570 GB |
| Each BF16 model-only SFT checkpoint | 570 GB |

Optimizer state can add multiple terabytes. Disable optimizer-state saves only
when the workflow does not require resumable training; otherwise provision the
required storage and validate checkpoint resume explicitly.

## Known Limitations

- DeepSeek V4 currently uses TP=1 with hybrid attention; scale with PP, EP, and
  DP. Use only recipe-qualified topology changes.
- Standard Megatron KV-cache autoregressive inference is unsupported because
  the hybrid-attention path does not accept an inference context. The
  [`inference.sh`](inference.sh) example uses legacy full-prefix generation and
  is not a verified KV-cache path. HF-native inference is the target after a
  verified Megatron-to-HF export.
- Fused mHC requires Blackwell (`sm_100`). Hopper uses the unfused path.
- CPU import and export require enough RAM for the full BF16 model plus
  conversion workspace and remain unverified in the current card.
