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

DeepSeek V4 training uses the native HybridModel, hash routing, and compressed
sparse attention support in Megatron-LM `main`. These features are included in
the repository's main-branch MCore pin for eager PP=1 training and checkpoint
conversion. Select that pin and sync the environment:

```bash
./scripts/switch_mcore.sh main
uv sync
```

Pipeline recipes also require the native mHC PP/VPP support in
[Megatron-LM #7336](https://github.com/NVIDIA/Megatron-LM/pull/7336).
Recipes using Transformer Engine or full-iteration CUDA graphs require
[Megatron-LM #7337](https://github.com/NVIDIA/Megatron-LM/pull/7337).
These capabilities are not included in the current main-branch pin. Use a
compatible MCore revision containing the required changes for those recipes;
the older `dev` revision uses different mHC configuration names and is not a
direct replacement for this native-main bridge.

Packed THD support, including packed SFT and LoRA, is still a work in progress
and is **not supported on Megatron-LM `main` or the current main-branch MCore
pin**. The packed recipes and commands below are development references.
Historical results in the verification card apply to their recorded revisions
and do not establish THD support for the native main-branch HybridModel.

Pipeline layouts split complete attention/MoE pairs in `hybrid_layer_pattern`,
with matching physical-layer metadata for embedding and MTP/loss placement.
The Flash library recipe uses PP4/VPP4; the 128-GPU Flash benchmark uses PP1.
The Pro benchmark also combines mHC CUDA graphs with recompute and activation
offload, which remains unsupported by the main-based runtime above.

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
  --executor slurm --device gpu --nodes 2 --gpus-per-node 4 \
  --hf-model deepseek-ai/DeepSeek-V4-Flash \
  --megatron-path work/models/deepseek-v4-flash/iter_0000000 \
  --hf-path work/models/deepseek-v4-flash-hf \
  --tp 1 --pp 1 --ep 8 \
  --torch-dtype bfloat16 --export-weight-dtype bfloat16 \
  --trust-remote-code --not-strict
```

BF16 export intentionally omits the source checkpoint's quantization-scale
companions, so it uses `--not-strict`. The verification card pairs that option
with exact key, shard, dtype, integer-routing-table, and strict HF reload checks.
GPU import and GPU export are verified. CPU conversion and manual HF/Megatron
forward correlation remain unverified at the revisions recorded in the card.

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
| GB200, 64 GPUs | `deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_library_config` | MXFP8 Adam |
| GB300, 32 GPUs | `deepseek_v4_pro_pretrain_32gpu_gb300_bf16_config` | BF16 Adam |
| GB300, 32 GPUs | `deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config` | MXFP8 Adam |

For example, a short generated-data run of the GB200 library recipe is:

```bash
./scripts/training/train.sh --nodes 16 --gpus-per-node 4 \
  --recipe deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_library_config \
  --mode pretrain --dataset mock --max_steps 10
```

The historical `128gpu` name is retained for compatibility. The recipe owns its
TP1/PP4/VPP4/EP16/CP1 topology, global batch size, natural-routing semantics,
recompute, offload, and checkpoint settings. Its current qualification is at 64
GPUs; larger data-parallel scales require separate performance validation. Keep
the recipe defaults and override only the dataset, run length, logging, pretrained
checkpoint, and output paths. The latest 100-step real-data candidate and its
checkpoint evidence are recorded in the verification card.

Compatibility aliases such as `deepseek_v4_flash_pretrain_mxfp8_config` remain
exported, but new launches should use the hardware-qualified names above.
Canonical benchmark recipes under `src/megatron/bridge/perf_recipes/` are
performance references, not substitutes for the library recipes.

## Supervised Fine-Tuning

DeepSeek-V4-Flash provides BF16 Adam full-parameter SFT recipes. Packed THD
support remains a work in progress and is not supported on `main`:

| Recipe | Sequence format | MTP | Target |
|--------|-----------------|-----|--------|
| `deepseek_v4_flash_sft_config` | Unpacked SBHD | On | Hopper or Blackwell |
| `deepseek_v4_flash_no_mtp_sft_config` | Unpacked SBHD | Off | Hopper or Blackwell |
| `deepseek_v4_flash_sft_openmath_thinking_packed_config` | Offline-packed THD (WIP) | On | Portable base |
| `deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config` | Offline-packed THD (WIP) | On | 32-GPU GB200 |

The recipes select fused mHC only when the runtime supports the Blackwell
kernel; Hopper uses the unfused fallback. The GB200 packed recipe additionally
enables HybridEP, uneven-dispatch padding, DSA fusion, grouped GEMM, selective
recompute, and attention activation offload.

The following packed SFT command is a development reference using an imported
BF16 checkpoint. It requires THD support outside the current main-branch path:

```bash
./scripts/training/train.sh --nodes 8 --gpus-per-node 4 \
  --recipe deepseek_v4_flash_sft_openmath_thinking_packed_gb200_config \
  --mode sft --step-func dsv4_step \
  --pretrained_checkpoint work/models/deepseek-v4-flash \
  --save_dir work/results/deepseek-v4-flash-sft \
  --save_interval 100 --max_steps 100 --seq_length 1024 \
  dist.distributed_timeout_minutes=180
```

The card records historical runs with 100 finite steps, a fresh-process checkpoint reload, and
post-SFT GPU export with deterministic HF inference for this configuration at
CP=1. It also records a separate 100-step CP=2 packed-SFT validation at
sequence length 1024. These results apply to the revisions listed in the card;
they do not establish THD support on `main`. Longer sequence lengths remain
unverified. MXFP8 and Muon SFT recipes are intentionally not shipped because full-model tests did
not establish a stable supported configuration.

## Parameter-Efficient Fine-Tuning

Packed OpenMath LoRA recipes preserve the SFT data and objective contract.
They use THD, which is still a work in progress and is not supported on `main`.
These recipes are retained as development references:

| Recipe | Target |
|--------|--------|
| `deepseek_v4_flash_peft_openmath_thinking_packed_config` | Portable base |
| `deepseek_v4_flash_peft_openmath_thinking_packed_gb200_config` | 32-GPU GB200 |

The LoRA rank and alpha are both 32. Adapters cover the MLA query down/up, KV,
and output projections plus shared and routed expert FC1/FC2 projections.
Routed experts use separate adapters (`share_expert_adapters=False`) rather
than sharing one adapter across the experts local to an EP rank.

The following development example uses the same imported BF16 checkpoint and
packed data as SFT. It requires THD support outside the current main-branch path:

```bash
./scripts/training/train.sh --nodes 8 --gpus-per-node 4 \
  --recipe deepseek_v4_flash_peft_openmath_thinking_packed_gb200_config \
  --mode lora --step-func dsv4_step \
  --pretrained_checkpoint work/models/deepseek-v4-flash \
  --save_dir work/results/deepseek-v4-flash-peft \
  --save_interval 100 --max_steps 100 --seq_length 1024 \
  dist.distributed_timeout_minutes=180
```

The verification card records loss, checkpoint, and performance results for
this recipe at the listed revisions. Those historical results do not establish
THD support on `main`.

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
| Each exported BF16 HF checkpoint | 570 GB |

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
