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
The Pro benchmark uses the separate runtime described below, including mHC
recompute with scoped attention CUDA graphs.

When `train.sh` overrides `model.pipeline_model_parallel_size` or
`model.virtual_pipeline_model_parallel_size`, the pattern segments and layout
are rebuilt with the default split of attention/MoE pairs: even, except that the
first stage grows to hold the hash-routed MoE layers. This replaces a recipe's
uneven split and keeps the MTP suffix. If the command also sets
`model.pipeline_model_parallel_layout`, nothing is rebuilt, so pass a matching
`model.hybrid_layer_pattern` as well. Python code that changes PP or VPP on a
recipe should call `set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)`.
Training rejects `|` segments that do not match PP x VPP or the explicit layout.
With VPP unset, an explicit layout sets it, as in Megatron Core: an eight-stage
PP4 layout means VPP2.

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
| GB200, 8 GPUs | `deepseek_v4_flash_proxy_pretrain_8gpu_gb200_bf16_config` | BF16 Adam, six-layer proxy |
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

### GB300 Pro performance reproduction and debugging proxy

These benchmark recipes retain the model geometry, parallelism, batch sizes,
Muon optimizer, model MXFP8, MoK/HybridEP, and mHC attention-graph settings from
the supplied full and proxy MCore logs. Runtime-specific differences are listed
below; matching these settings does not establish numerical or timing parity.
The earlier 50-step GB300 runs used the private `dsv4` revision
`d211e50e888fcb8640ddc9923e98940800afcfd3` and its compatible kernel container.
They completed with finite reported losses and gradient norms and no skipped or
NaN iterations; this is execution evidence, not convergence or numerical parity.

Both eager and attention-graph benchmarks run on MCore `main` with
[MCore #7337](https://github.com/NVIDIA/Megatron-LM/pull/7337) merged in. #7337
adds attention graph support around the mHC wrappers and makes mHC's direct
projection-weight reads wait for an overlapped parameter gather. The recipes
reject an MCore without it. A conflict-free merge of main
`c1ad819d07cb946620341a317b8d06e74207e0f1` with #7337 at
`12880406160688146b19733996d0ede8bd3a2726`, with no other changes, is
[`28c439772712240602a28123b462d8e374bec67f`](https://github.com/malay-nagda/Megatron-LM/commit/28c439772712240602a28123b462d8e374bec67f).
Use its full SHA for nemo-ci's `--mcore-ref`, and mount this Bridge revision and
that MCore source explicitly; changing the Bridge source alone does not replace
the image's baked-in MCore.

Open [MCore #5479](https://github.com/NVIDIA/Megatron-LM/pull/5479) is optional.
With it, Muon keeps compact buffers while fallback Adam stays sharded
(`ddp.use_distributed_optimizer=True`), as in the reference logs. Without it,
configuration finalization sets `ddp.use_distributed_optimizer=False` and
LayerWise owns both in its compact layout. That costs optimizer memory per rank;
the amount has not been measured. Native indexer scoring uses BF16.

GB300 needs no special image: the regular nightly's MoK build covers SM100 and
SM103. Rubin needs a MoK build with SM107 kernels, such as the image from
pipeline 82972924 (`nvcr.io/nvidian/nemo:dgxctestingtemp-nemofw-nightly.83236778`,
MoK `31a3dd0e`). Compatible Transformer Engine, HybridEP/MoK, fused mHC, cuDNN
DSA, and emerging-optimizers implementations are required.

PyTorch 26.11 (`2.15.0a0+fd23299d96.nv26.11`, the base of that image) makes
`nccl2` the default `nccl` backend, and its snapshot has two defects these
recipes hit. Both are fixed upstream but not in that image:

- An async barrier followed by `block_current_stream()` deadlocks
  ([pytorch#202783](https://github.com/pytorch/pytorch/pull/202783)). MoK's
  workspace setup makes that call, so training stalls in the first MoK forward.
- `all_gather` fails with `shape '[N]' is invalid for input of size M` when its
  first two output tensors are empty
  ([pytorch#200585](https://github.com/pytorch/pytorch/pull/200585)). LayerWise
  parameter gathering passes one slot per rank, and most ranks own nothing in a
  given bucket.

The Pro benchmark recipes therefore export `TORCH_DIST_USE_NCCL2=0`, which
selects the legacy NCCL backend. A value set by the launcher takes precedence.
With that setting and `NCCL_NVLS_ENABLE=0`, the VRNVL72 eager proxy gets past
the parameter gather. Separately,
[MoK #10](https://github.com/QiZhangNV/mixture-of-kittens/pull/10) replaces
MoK's async barrier with a blocking one, pinned by
[Bridge #6687](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6687);
whether the unpatched MoK works on the legacy backend is untested. Remove the
setting once the image carries both upstream fixes.

The adapter also retains the development API used with MCore
`264817a74075a0c4b003723054c3ae71020058ad`. It selects declared dataclass fields
and resolves main's static hybrid spec or development's spec factory.
Native main uses **BF16 indexer scoring**; development retains its compact
MXFP8 indexer. Both runtimes retain model MXFP8 and FP8 parameter gathering,
`optimizer.overlap_param_gather=True` and `ddp.overlap_param_gather=True`.
Muon remains layer-wise (`optimizer.use_distributed_optimizer=False`,
`use_layer_wise_distributed_optimizer=True`) with compact parameter layout.
Where MCore has #5479, the shared DDP flag retains sharding for fallback Adam;
it does not change Muon to the ordinary distributed Adam optimizer. Optimizer-step gather overlap stays
disabled because it is incompatible with this LayerWise/MXFP8-buffer-reuse path.

The main plus #7337 runtime still has these differences:

- Native CSA indexer scoring uses BF16 and has no indexer precision selector;
  model training remains MXFP8. The development MXFP8 indexer port is not required.
- #7337 enables attention graphs with mHC recompute group two, but retains a
  copied attention aggregate in the static graph buffer. Its memory/performance
  behavior differs from development's aggregate recomputation implementation.
- Main infers the fixed microbatch graph count and uses
  `cuda_graph_thd_max_packed_sequences=8`; the development-only dynamic graph
  flag is unnecessary for these fixed 64-microbatch optimizer steps.
- Main uses native pipeline shape exchange. The development fixed-P2P-shape
  optimization is absent; this matters for full PP4/VPP4, not PP1 proxy traffic.
- Grouped output projection uses main's existing implementation. The separate
  BatchedLinear optimization is not part of this runtime.
- Without #5479, fallback Adam state is not sharded across data-parallel ranks.

Configuration tests do not exercise GPU kernels. For a functional run, verify
finite completed LM/MTP/auxiliary losses and gradient norms, no skipped/NaN
iterations, actual Muon execution, graph capture and post-capture steps, and a
clean training exit. Do not infer attention graph execution from config alone.

The earlier public-development target was the immutable merge revision
`159e66cd2491b1bf4fdb0fa0895dd58f4dd9e4ed` of MCore
[PR #5722](https://github.com/NVIDIA/Megatron-LM/pull/5722), with
`nvcr.io/nvidian/nemo:nightly-2026-10-16`. Historical timing and memory results
are not guarantees for the refreshed source and kernel combination.

The recipes set both native CP layouts to `contiguous` and retain fixed THD
boundaries. Development-only route finalization is used only when available.
Bridge's shared DSA initialization and logging own the metric lifecycle;
training-attempt resets clear metric storage in place so captured graph
addresses remain valid. The recipes explicitly set `is_hybrid_model=True`,
matching native MCore argument processing for output-weight initialization and
layer accounting. Earlier Bridge runs inherited the non-hybrid default, so
their initial weights are not reproduced by the updated recipes.

| Recipe | GPUs | Main logical blocks | TP / CP / EP | PP / VPP | GBS / MBS |
|--------|------|---------------------|--------------|----------|-----------|
| `deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config` | 256 | 61 | 1 / 16 / 64 | 4 / 4 | 256 / 1 |
| `deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config` | 64 | 16 | 1 / 16 / 64 | 1 / none | 256 / 1 |
| `deepseek_v4_pro_pretrain_64gpu_vrnvl72_fp8mx_proxy_config` | 64 | 16 | 1 / 16 / 64 | 1 / none | 256 / 1 |

All three retain the Pro widths, expert count, one MTP module, 65,536-token sequence
length, MXFP8 parameter gathering, and Muon with the `deepseekv4` coefficients
and ten Newton–Schulz iterations. Each main logical block contains one attention
layer and one MoE layer, so MCore reports 122 and 32 main physical layers,
respectively. Both configurations have dense DP4, expert DP1, and 64
microbatches per optimizer step. The full model splits its logical blocks across
the sixteen VPP chunks as `[3] + [4] * 14 + [2]`.

Both use selective `mla_up_proj` and mHC recompute with mHC group size two,
Transformer Engine attention CUDA graphs with one warmup step and support for
the fixed microbatch count, and no activation offload or paged stash. The proxy reduces the
allocation by four while retaining the long-context attention, MoE, mHC, MTP,
and optimizer paths. Its PP1 schedule does not exercise the full recipe's
pipeline communication or interleaved scheduling, and the smaller allocation
does not guarantee a fourfold reduction in step time.

Start with the proxy, then run the full recipe using the same runtime. With
four GPUs per node, the canonical launches are:

```bash
./scripts/training/train.sh --nodes 16 --gpus-per-node 4 \
  --recipe deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config \
  --mode pretrain

./scripts/training/train.sh --nodes 64 --gpus-per-node 4 \
  --recipe deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config \
  --mode pretrain
```

Use the account, partition, and container variables described above, and adjust
the node count if the cluster exposes a different number of GPUs per node.
These recipes own their mock dataset and 50-step benchmark schedule. Forced
load balancing and disabled benchmark correctness checks mean a completed run
is only an execution smoke test, not a convergence or numerical-parity test.
For a stricter diagnostic run, enable `ddp.check_for_nan_in_grad=true` and
`rerun_state_machine.check_for_nan_in_loss=true`; record those overrides when
comparing performance.

For the Rubin proxy, select
`deepseek_v4_pro_pretrain_64gpu_vrnvl72_fp8mx_proxy_config` instead. It uses the
same model and training contract, with MoK communication SMs set to 40/40 from
the Rubin reference, compared with GB300's 28/28.

The launchers select `dsv4_step` automatically for these recipes. The
synthetic dataset emits fixed THD packed metadata, including boundaries on
intermediate pipeline stages, with one full-length sequence per microbatch.
It reproduces the fixed pack shape used in the logs, rather than their random
token stream or the general variable-length `dp_balanced` data scheduler.

### Eager Pro functional tests

Use the same main plus #7337 runtime as the graph benchmarks for these
explicit variants. Eager execution also requires #7337's mHC projection
readiness check so direct weight reads wait for overlapped parameter gathering.

| Recipe | Performance selector variant |
|--------|------------------------------|
| `deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_eager_config` | `eager` |
| `deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_eager_config` | `proxy_eager` |
| `deepseek_v4_pro_pretrain_64gpu_vrnvl72_fp8mx_proxy_eager_config` | `proxy_eager` |

For the performance launcher's Rubin proxy, retain your allocation, image and
source-mount arguments and use this model selection:

```bash
-m deepseek -mr deepseek_v4_pro -g vrnvl72 -c fp8_mx -ng 64 \
  --task pretrain --config_variant proxy_eager
```

Keep the flat performance recipe path (`--use_recipes` omitted). Remove explicit
CUDA-graph or parameter-gather-overlap overrides from an earlier launch so they
do not override the recipe's restored policy. For the training launcher, select the
full factory name from the table with `--recipe`.

These variants retain all model layers for their respective full/proxy sizes,
one MTP, sequence 65,536, CP16/EP64, GBS256/MBS1, Muon, model MXFP8 and parameter
gather buffer reuse, and selective `mla_up_proj`/mHC recompute with
`mhc_recompute_layer_num=2`. Eager mode retains native BF16 indexer scoring
(or MXFP8 with the development runtime).
Fixed THD metadata capacity remains eight in eager mode.

The deliberate execution changes are:

- `cuda_graph_impl="none"`, empty graph scopes/modules and zero graph warmup.
- Both model and RNG TE-tracker flags are false.
- Gather alignment is disabled. Parameter-gather overlap remains enabled in
  both DDP and the optimizer; optimizer-step gather overlap remains disabled.
  Gradient-reduction overlap and the full model's PP overlap remain configured
  by the corresponding benchmark.

The helper requires the declared mHC readiness/graph-split capability in both
modes and sets the split false for eager execution. The benchmark no longer
supports bare main by disabling parameter-gather overlap. Select an explicit
eager variant before construction rather than overriding a graph recipe later.

The value two groups physical HybridStack layers for activation recomputation;
in this attention/MoE pattern that is one logical block. It does not reduce model
depth. Eager runs omit graph replay, so they are functional-test variants and
their throughput is not directly comparable to the
graph-enabled reference. GPU execution must still be validated on the target
cluster.

### MCore graph-helper import compatibility

[MCore #8924](https://github.com/NVIDIA/Megatron-LM/pull/8924) renamed
`get_vision_cuda_graph_seq_length` to `get_vision_te_cuda_graph_seq_length` on
main, preserving its signature and behavior. This is unrelated to mHC #7337 or
Rubin kernels. The shared training module imports vision support even for a
text-only DSv4 run; Bridge resolves either helper name through its compatibility
utility. Validation includes importing `megatron.bridge.training.pretrain` in
addition to constructing recipe configs.

The Slurm performance launcher bind-mounts its Bridge source, but its Git packager
excludes submodules. Bridge imported from a shared checkout and MCore imported from
`/opt` can therefore be intentional. Check the actual imported files and their Git
revisions in the training container; a path difference alone does not establish a
revision mismatch.

For an explicit MCore checkout, select it in the launcher's Python environment and mount
the same source for workers. Set `BRIDGE_SOURCE` and `MCORE_SOURCE` to your existing
checkouts, and verify `git -C "$MCORE_SOURCE" rev-parse HEAD` against the desired pinned
revision. Before invoking `scripts/performance/setup_experiment.py`:

```bash
export PYTHONPATH="${BRIDGE_SOURCE}/src:${MCORE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
```

Add these arguments to that invocation, combining the mount with any existing mounts:

```bash
--custom_mounts "${MCORE_SOURCE}:/opt/dsv4-mcore" \
--custom_bash_cmds 'export PYTHONPATH=/opt/dsv4-mcore:$PYTHONPATH'
```

The pre-command runs inside the worker container before Python. Use this explicit
pre-command rather than relying solely on `--custom_env_vars PYTHONPATH=...`: NeMo-Run
merges the task environment after executor variables, and this launcher supplies its own
task `PYTHONPATH` prefix.

Inside the same container and source environment as the training process, this read-only
check verifies imports and identifies their source revisions:

```python
import importlib
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)

for name in ("megatron.bridge", "megatron.core", "megatron.bridge.training.pretrain"):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    result = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    revision = result.stdout.strip() if result.returncode == 0 else "Git revision unavailable"
    logging.info("%s: %s (%s)", name, path, revision)
```

For an unpacked snapshot without `.git`, the path and installed package version cannot
establish the source commit: retain the launch's source manifest and pin the checkout
before packaging. A successful import check does not validate GPU kernels or training.

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
kernel; Hopper uses the unfused path, which matches DeepSeek-V4's mHC
normalization (see Known Limitations). The GB200 packed recipe additionally
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
- [`inference.sh`](inference.sh) needs `PP * EP` GPUs (32 for the Pro
  defaults `PP=4`, `EP=8`) and, like the other examples, runs from the
  repository root. On 4-GPU nodes such as GB200 NVL72 spread the ranks over
  several nodes with one task per node:
  `srun --ntasks-per-node=1 bash -c 'NNODES=$SLURM_JOB_NUM_NODES NODE_RANK=$SLURM_NODEID MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) examples/models/deepseek_v4/inference.sh'`.
  With the variables unset the script launches all ranks on the current node.
- Fused mHC requires Blackwell (`sm_100`). Hopper uses the unfused path.
- DeepSeek-V4 normalizes the mHC mapping input as
  `x * rsqrt(mean(x^2) + rms_norm_eps)`. Only the unfused mHC path computes
  this (`mhc_norm_eps_inside_sqrt=True`); the fused kernels add epsilon outside
  the square root. The bridge's model config therefore uses the unfused path
  on every GPU. Recipes enable fused mHC on Blackwell for speed; for exact
  reference numerics there, override
  `model.use_fused_mhc=False model.mhc_norm_eps_inside_sqrt=True`.
- CPU import and export require enough RAM for the full BF16 model plus
  conversion workspace and remain unverified in the current card.
