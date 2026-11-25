# Reproduce ERNIE 5.5 Flash 256K BF16 training on 64 GB300s

This experiment trains the public ERNIE-5.5-Flash-Base checkpoint with both its main LM head and MTP head. It compares natural MoK routing with the same routing under MoonEP expert replication. Use one 64-GPU NVLink domain (16 nodes × 4 GB300s). TP1, PP1 and CP16 give four dense data replicas. EP64 spans all ranks. Each replica sees one packed 262,144-position row per microbatch and accumulates eight microbatches. An optimizer update therefore contains 32 rows. The recipe runs eagerly with full uniform recomputation, two hybrid symbols per group, BF16 weights and expert math, FP32 router scores, and a CUDA token-chunked head (4,096 rows) on both outputs.

## 1. Check out the source and prepare the training environment

Use a Linux GPU container supporting SM103/SM103a on **every** node. The measured base was NGC PyTorch 26.04 (Python 3.12, CUDA 13.2, PyTorch 2.12.0a0, Triton 3.6). The training stack used Transformer Engine 2.20.2, FlashMLA with sparse sink attention (commit `b7643bd`), DeepEP hybrid-ep (`10d4dd7`), cuDNN frontend 1.29.1 with CUTLASS DSL 4.8.1, fast-hadamard-transform 1.1.0, Emerging Optimizers 0.3.0 and Transformers 5.20.2. The base NGC image alone needs these additions. Install the dependencies in the same Python environment used by torchrun. [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) · [FlashMLA source and build instructions](https://github.com/deepseek-ai/FlashMLA/tree/b7643bd54521f563b839b98289b5cd048c062ba2) · [DeepEP HybridEP source and build instructions](https://github.com/deepseek-ai/DeepEP/tree/10d4dd7377d5bce900fbb4b80cce863764892b95) · [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

Run the following from a shared workspace visible on all nodes. `BRIDGE` is the checkout of [Bridge #7000](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7000) containing this README and its helpers. The performance runs and their helpers in steps 1–6b use `aeec72b844cbb8b951300b738cea6ab4fddb59c5` (training implementation introduced in `43acf2daf25ab96870fb530dad546f9dbb201ad4`). The companion [MCore #9506](https://github.com/NVIDIA/Megatron-LM/pull/9506) supplies the MoonEP backend, packed H support and stable D-indexer selection.

```bash
export WORK=$PWD
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
git -C Megatron-Bridge fetch origin refs/pull/7000/head
git -C Megatron-Bridge checkout --detach aeec72b844cbb8b951300b738cea6ab4fddb59c5
git clone https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM fetch origin refs/pull/9506/head
git -C Megatron-LM checkout --detach 32efd940666ed3efd8fbbaeacc2a75d9d7caa3c9
export BRIDGE=$WORK/Megatron-Bridge MCORE=$WORK/Megatron-LM
export EX=$BRIDGE/examples/models/ernie55
export PYTHONPATH=$MCORE:$BRIDGE/src:${PYTHONPATH:-}
git -C "$BRIDGE" rev-parse HEAD
git -C "$MCORE" rev-parse HEAD

# Set this to the environment in which the training dependencies are installed.
export UV_PROJECT_ENVIRONMENT=$(python -c 'import sys; print(sys.prefix)')
export MOK_ARCH=SM103
mkdir -p "$WORK/mok-wheels"
bash "$MCORE/docker/common/install_mok.sh" ee7bdaba515fd3d557105d71809ec74cd523d5a1 "$WORK/mok-wheels"
export ERNIE55_CUDA_HEAD_LIBRARY=$WORK/libernie55_multimax_sm103.so
python "$BRIDGE/src/megatron/bridge/models/ernie/modeling_ernie55/cuda_head.py" \
  --arch sm_103 --output "$ERNIE55_CUDA_HEAD_LIBRARY"
python -c 'import torch, transformer_engine, flash_mla, deep_ep, mok; print(torch.__version__, torch.version.cuda)'
```

The installer builds and installs the native extension from the [public MoK commit](https://github.com/QiZhangNV/mixture-of-kittens/commit/ee7bdaba515fd3d557105d71809ec74cd523d5a1) and checks its SM103a cubin. Keep the checkout and its compiled head available on every worker. Installing a wheel on one node only suffices when the Python environment is shared.

### Optional public container build

If a compatible SM103 training environment is already installed, continue to step 2. For a source-built environment, the companion MCore checkout provides its [CI development Dockerfile](https://github.com/NVIDIA/Megatron-LM/blob/32efd940666ed3efd8fbbaeacc2a75d9d7caa3c9/docker/Dockerfile.ci.dev). The following uses the public NGC **26.09** base and the Dockerfile's locked GPU libraries, including cuDNN frontend 1.32.0. This is a source-reviewed build route. The historical throughput table used the **26.04** stack above. A complete build and 64-GPU training from the 26.09 image have not been validated for this benchmark.

```bash
# Build on a Linux arm64 host with a CUDA toolchain and Docker Buildx.
cd "$MCORE"
docker buildx build --platform linux/arm64 --target main --load -t ernie55-mcore:sm103 \
  -f docker/Dockerfile.ci.dev --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3 \
  --build-arg IMAGE_TYPE=dev --build-arg 'NVTE_CUDA_ARCHS=100a;103a' \
  --build-arg MOK_ARCH=SM103 --build-arg MOK_COMMIT=ee7bdaba515fd3d557105d71809ec74cd523d5a1 \
  --build-arg DEEPEP_COMMIT=10d4dd7377d5bce900fbb4b80cce863764892b95 .
```

The Dockerfile builds MoK from source; skip the separate `install_mok.sh` in this route. Deploy the image using your site's multi-node container launcher and mount both checkouts and the shared workspace on all workers. Inside the container, install Bridge's dependencies and both editable checkouts in the active environment (the public Bridge checkout declares its own Transformers range), set the same environment paths, and compile the head there before launching.

```bash
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-build-isolation -e "$MCORE" -e "$BRIDGE"
export ERNIE55_CUDA_HEAD_LIBRARY=$WORK/libernie55_multimax_sm103.so
python "$BRIDGE/src/megatron/bridge/models/ernie/modeling_ernie55/cuda_head.py" \
  --arch sm_103 --output "$ERNIE55_CUDA_HEAD_LIBRARY"
# Inside an allocated GB300 container.
python - <<'PYGPU'
import inspect
import torch
import cudnn, mok, transformer_engine.pytorch
import megatron.bridge, megatron.core
from deep_ep import HybridEPBuffer
from fast_hadamard_transform import hadamard_transform
from flash_mla import flash_mla_sparse_fwd
assert torch.cuda.get_device_capability() == (10, 3)
assert {"attn_sink", "topk_length", "indexer_topk"} <= set(inspect.signature(flash_mla_sparse_fwd).parameters)
print(torch.__version__, torch.version.cuda)
PYGPU
```

## 2. Download and convert the weights

Use the pinned original HF directory as the weight source. Use the generated architecture directory as the trainer's model configuration. This keeps source FP8 scales available to the importer. Runtime and routed experts use BF16. A distributed checkpoint from the same weights can be reused. The source is approximately 288 GB and the distributed checkpoint approximately 561 GB, in decimal units.

```bash
export DATA=$WORK/ernie256k-inputs
export HF_MODEL=$WORK/ERNIE-5.5-Flash-Base
export ERNIE55_HF_PATH=$WORK/ernie256k-training-config
export CKPT=$WORK/ernie256k-mcore-step0
mkdir -p "$DATA"
hf download baidu/ERNIE-5.5-Flash-Base --revision 028544b2ffceea189b2d8768ed2b9d09b06f2343 --local-dir "$HF_MODEL"
python "$EX/benchmark_256k.py" architecture "$HF_MODEL" "$ERNIE55_HF_PATH"
printf '%s  %s\n' 3d1790a86b6debaa6cb0201aa3397541e17743b030a08bb3b5024a04f2011e62 "$ERNIE55_HF_PATH/config.json" | sha256sum -c -
# Once on one four-GB300 node. CKPT must be a new path.
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$EX/benchmark_256k.py" convert "$HF_MODEL" "$CKPT"
```

The architecture helper keeps the official model dimensions and MTP topology, selects the 262,144-position benchmark limit, makes the release's YaRN factor (16) and original context (65,536) explicit, selects adjacent-pair D RoPE, uses the registered ERNIE alias and removes the HF quantization metadata from the training config. The pretraining recipe sets the D indexer loss coefficient to 0.01. The official checkpoint is read without modification. A four-GB300 EP4 preflight with the public runtime checked all 134,231 indexed keys across the 128 official shards. The three complete sampled FP8 projection matrices dequantized bitwise identically to an independent BF16 calculation. Earlier conversions wrote and verified 561.3 GB distributed checkpoints at two sites. The public training validation reused a converted checkpoint. [Model release](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/tree/028544b2ffceea189b2d8768ed2b9d09b06f2343)

## 3. Prepare one fixed packed row

For the exact public-input control, rebuild the first 12 source rows from the pinned public corpora and the official tokenizer. The preparation helper needs `pyarrow`, `huggingface_hub` and `tokenizers`. Then cut the documents for the three layouts. The benchmark source rows and the three fixture files have the SHA-256 digests below. Each private or synthetic replacement can change routing, load balance, memory, timing and loss.

```bash
python "$EX/prepare_256k_source_rows.py" --tokenizer "$HF_MODEL/tokenizer.json" --output "$DATA/source-12.jsonl"
for layout in single uniform skew; do
  python "$EX/benchmark_256k.py" prepare "$layout" "$DATA/$layout.jsonl" --source "$DATA/source-12.jsonl"
done
sha256sum "$DATA"/{source-12,single,uniform,skew}.jsonl
```

| File | Unpadded document lengths | SHA-256 |
| --- | --- | --- |
| source-12 | 12 source rows of 262,144 tokens | `5fb767243f8369b0a0c9150103848bf7be22846325b78dfe272ef9a43373d513` |
| single | 262,144 | `1f8b16058ae3cfb8ca4509ff87f22364b21623cae3f805e8a2173f0fabe0fe93` |
| uniform | 8 × 32,768 | `b83d07c8faad3b27ce5571bbc375cf029a271aed0aedc2cfda39f0a8d9f5b854` |
| skew | 131200, 65408, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 117 | `52b018ed97fd503d6fbc0ed5c96e708b50e7f4d9615fc6a8c8e43e156339e984` |

Each JSONL line is a single document with `input_ids` and unshifted `labels`. The fixed fixtures mask the first eight labels in each document. The ERNIE loader shifts the targets within each document and pads to 32 positions at CP16. The skewed row has 262,133 real positions and 11 padding positions. The loader repeats this same packed row across replicas and updates. To test only the shapes, omit `--source` to generate seeded synthetic tokens. To run a private workload, supply tokenized document lines with the desired labels directly as `dataset.path` and set the document cap for its rows. For a cache-local timing comparison, copy the same fixture to the same node-local path on all nodes before both arms and verify its digest.

## 4. Launch two matched 64-GPU runs

Set the following environment on all 16 nodes. Assign `NODE_RANK` from 0 to 15 and set `MASTER_ADDR` to the address of rank zero. Each node starts exactly one torchrun with four workers. The example uses a twelve-document row. For `single` and `uniform`, set `DOCS` to 1 and 8 respectively. Disable Nsight, record fresh logs per arm, and wait for each run to finish on all nodes.

```bash
export LAYOUT=skew DOCS=12
export MCORE_DSV4_GROUP_PROJ_BMM=1 NCCL_MAX_P2P_NCHANNELS=8 NCCL_GRAPH_REGISTER=0
export MCORE_DSA_STABLE_INDEXER_TOPK=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export INPUT=$DATA/$LAYOUT.jsonl
export LOGS=$WORK/ernie256k-logs
mkdir -p "$LOGS"
set -o pipefail
run_arm() {
  local arm=$1 moon=$2
  shift 2
  python -m torch.distributed.run --nnodes=16 --nproc-per-node=4 \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29500}" \
    "$EX/benchmark_256k.py" train "dataset.path=$INPUT" "checkpoint.pretrained_checkpoint=$CKPT" \
    "model.thd_max_packed_sequences=$DOCS" "model.moe_virtual_expert_load_balance=$moon" \
    "logger.tensorboard_dir=$LOGS/$arm-tb" "$@" 2>&1 | tee "$LOGS/$arm-node$NODE_RANK.log"
}
run_arm mok false
run_arm moon true
```

The `false` and `true` arms use identical MoK macrobatches (16,384), expert routes, shapes, full recomputation and optimizer settings. The second arm enables expert replication. MTP is enabled by the model configuration and has loss scale 0.1. Confirm both `ERNIE55_CUDA_HEAD dispatch=main` and `dispatch=mtp` appear. Optional W&B logging uses `logger.wandb_project` and `logger.wandb_exp_name` overrides passed to `run_arm`, with local credentials. Give each arm a distinct name. Disable experiment logging hooks that add profiling to timing runs.

### 4b. Reproduce the two MoonEP-enabled arms in the throughput and memory table

Prepare a separate checkout **once on the shared workspace** for the retain-source baseline. Both checkouts begin at the same verified MCore commit. Reverse just the release commit in the baseline checkout. It removes one runtime assignment and the corresponding unit-test assertion. All packed attention, MoK, MoonEP, CUDA-head and other source changes are shared between the arms. The normal checkout used in step 4 stays intact.

```bash
export MCORE_RETAIN=$WORK/Megatron-LM-retain-256k
release=d4c8c19d4b10f9ea8bc6c33ef1ebb979effca024
git -C "$MCORE" worktree add --detach "$MCORE_RETAIN" 32efd940666ed3efd8fbbaeacc2a75d9d7caa3c9
git -C "$MCORE" diff "$release^" "$release" > "$WORK/ernie256k-release.patch"
git -C "$MCORE_RETAIN" apply --reverse --check "$WORK/ernie256k-release.patch"
git -C "$MCORE_RETAIN" apply --reverse "$WORK/ernie256k-release.patch"
git -C "$MCORE_RETAIN" diff --stat
```

Run the following on each node of the same allocation, using the `run_arm` function from step 4. Both arms start from the same model and real expert routing. Each subshell selects the intended MCore path. For each input the script writes to its own directory. Add W&B project and unique run names to the two `run_arm` calls if desired.

```bash
(
  unset ERNIE55_COMPARE_HEADS
  export MCORE_DSA_STABLE_INDEXER_TOPK=0
  for LAYOUT in single uniform skew; do
    case "$LAYOUT" in single) DOCS=1 ;; uniform) DOCS=8 ;; skew) DOCS=12 ;; esac
    export LAYOUT DOCS
    export INPUT="$DATA/$LAYOUT.jsonl" LOGS="$WORK/ernie256k-memory-$LAYOUT"
    mkdir -p "$LOGS"
    ( export PYTHONPATH="$MCORE_RETAIN:$BRIDGE/src:${PYTHONPATH:-}"
      run_arm moon-retain true model.recompute_num_layers=1 )
    ( export PYTHONPATH="$MCORE:$BRIDGE/src:${PYTHONPATH:-}"
      run_arm moon-release true model.recompute_num_layers=2 )
  done
)
# After all sixteen nodes finish, summarize from one node.
for layout in single uniform skew; do
  for arm in moon-retain moon-release; do
    python "$EX/benchmark_256k.py" summarize "$WORK/ernie256k-memory-$layout"/"$arm"-node*.log
  done
done
```

## 5. Compare time and memory

```bash
python "$EX/benchmark_256k.py" summarize "$LOGS"/mok-node*.log
python "$EX/benchmark_256k.py" summarize "$LOGS"/moon-node*.log
```

Use all updates 4–12, including long steps. Each update processes 8,388,608 physical token positions, or 131,072 per GPU. Divide the summed token count by the summed wall time. The helper checks that all nine updates and all 64 rank reports are present, and reports full-run maxima for allocated and reserved GPU memory separately, in GiB. CUDA allocator watermarks and device-wide NVML usage measure different quantities.

The following integrated measurements compare two **MoonEP-enabled** arms. The baseline used one symbol per checkpoint region and retained source activations. The optimized arm groups two symbols and releases source activations after their last use. Each pair used a single rack, the native D selector, 12 updates, eager BF16 and real weights. The table reports the integrated implementation. The public PR heads can be measured with the preceding commands.

| Input | Baseline → optimized tokens/s/GPU | Baseline → optimized max allocated GiB |
| --- | ---: | ---: |
| Single | 3,744.8 → 3,751.1 | 236.377 → 182.627 |
| Eight equal | 4,440.8 → 4,548.1 | 227.582 → 173.082 |
| Twelve skewed | 4,233.8 → 4,303.4 | 228.662 → 174.162 |

### Public checkout validation

These runs loaded the same released weights and twelve-document fixture on 64 GB300s. Both used public MCore `32efd9406`, natural MoK + MoonEP, full two-symbol recomputation, both heads and the 26.04 environment from step 1. All twelve optimizer updates completed. The 4–12 window includes every step.

| Public Bridge revision | D-indexer tie selection | Average seconds/update | Tokens/s/GPU | Max allocated / reserved GiB |
| --- | --- | ---: | ---: | ---: |
| `a170a3321` | Stable (precision) | 48.104 | 2,724.74 | Unavailable |
| `b4994e8e5` | Native (performance) | 30.372 | 4,315.58 | Unavailable |
| **`aeec72b84`** | **Native (performance)** | **30.274** | **4,329.53** | **174.161674 / 176.146484** |

The final run used the step 4 launcher and public `summarize` command, with all 64 rank watermarks present. The normalized token count was 1,179,648 per GPU across updates 4–12, and the fastest and slowest updates in that window took 30.262 and 30.288 seconds. Both heads were enabled, with zero skipped or NaN updates. The older launchers did not retain their watermarks; the pinned launcher preserves its measurements across environment re-execution and writes them outside the training logger’s warning filter.

The public release/group-2 versus retain/group-1 precision pair completed one full GA8 update per arm on the skew fixture. With stable D ties and production MoK, all 13 captured fields at the first microbatch were byte-identical on each of the 64 ranks. Step 6b gives the tensor coverage.

## 6. Compare head values before the first optimizer update

Run this separately from timing. Enable deterministic D-indexer tie selection, keep the input/checkpoint/seed fixed and allocate a **new directory** for each capture. The probe reads real CUDA-head logits and records all per-token losses, sampled raw logits and head-input gradients for the **first microbatch**. It maps CP16 zigzag rows back to global padded positions, verifies document coordinates and requires the complete set of 64 rank records. Run on every node with the same function and environment from step 4.

```bash
export MCORE_DSA_STABLE_INDEXER_TOPK=1
export ERNIE55_COMPARE_HEADS=$LOGS/mok-heads
run_arm mok-check false train.train_iters=1
export ERNIE55_COMPARE_HEADS=$LOGS/moon-heads
run_arm moon-check true train.train_iters=1
unset ERNIE55_COMPARE_HEADS
python "$EX/head_parity_256k.py" "$LOGS/mok-heads" "$LOGS/moon-heads"
```

The comparator checks token IDs, document boundaries, target masks and physical positions in every dense replica before comparing LM and MTP. It reports mean valid-token CE, sampled raw-logit L2, sampled head-input gradient L2 and peak-relative errors. The diagnostic gates are 0.001% relative mean CE, 0.5% relative L2 and 1% maximum absolute difference divided by reference peak. All outputs of the helper are unitless fractions. Training loss averaged over a full update, including W&B loss, cannot identify the per-token or per-head differences measured here.

A separate 32-GPU test of the public packed H implementation used two independent CP16 groups and a per-document CP1 oracle. Both the window-only and ratio-128 configurations passed on all ranks. The worst relative L2 across contiguous and zigzag layouts was 0.21575% for output, 0.54794% for input gradients and 0.66828% for any parameter gradient; cross-document leakage was zero. This H-layer test used its unit-test geometry.

With stable index selection, earlier integrated runs of MoK versus MoK + MoonEP matched all captured CE, sampled raw logits and sampled head-input gradients on all three layouts before the first update. The native cuDNN selector requests the top 2,048 keys; equal FP32 scores at the cutoff can admit different eligible keys. The stable diagnostic sorts each query’s entire candidate axis with descending `torch.argsort(stable=True)` before slicing the first 2,048 entries, so lower key IDs win ties. This additional work affects throughput; the independent Swift gate below measures the resulting model values. On the uniform fixture a full GA8 gradient check of the optimized versus baseline integrated MoK + MoonEP configuration gave 0.2982% relative L2 and 0.2428% peak-relative over 279.6B gradient elements. Some individual shards exceeded those aggregate thresholds.

The earlier pre-FC2 and key-ID tie controls still showed differences against independent Swift. Section 6c gives the subsequent reference-arithmetic recipe and its checks for both heads.

For an independent reference that uses contiguous CP, export the same NPZ schema as the probe. A file contains its dense replica ID (`dp`), global padded document boundaries (`bounds`), local rows' physical global coordinates (`pos`), `tokens`, the actual `main_labels` and rolled `mtp_labels` with invalid targets set to −100, and both `main_ce` and `mtp_ce`. Store the global positions in `sample` and the corresponding `main_logits`, `mtp_logits`, `main_dh` and `mtp_dh` in that order. Raw logits take vocabulary IDs 0, 1237, 2474, …; the hidden-gradient samples contain every channel. The comparator reconstructs each full replica independently of CP sharding and joins sampled values on global position. For each implementation write `rank00.npz` through `rank63.npz`, covering every global position in each of the four dense data replicas. Use the same vocabulary IDs and loss scaling in both trainers. For matching samples, select physical positions where `pos % 2048` is 15 or 1023, plus each padded document end minus two. The comparator uses the intersection of sampled coordinates. Check coverage before interpreting a small difference: for the fixed twelve-document skew fixture, the rule gives 268 sampled physical coordinates per dense replica, including 266 valid main targets and 255 valid MTP targets. After recording the original Swift run and the candidate run into separate directories, use Swift as the reference denominator.

```bash
python "$EX/head_parity_256k.py" "$SWIFT_CAPTURE" "$CANDIDATE_CAPTURE"
```

### 6b. Check precision of the activation-memory optimization

Use the two MCore checkouts from step 4b. Keep MoonEP enabled in **both** arms. Enable stable D-indexer selection and compare their first microbatch before any weights are updated. These short instrumented runs are separate from the throughput table. Start from the same checkpoint, reuse the same JSONL within each pair, and use a fresh `PARITY_DIR` shared by all nodes. The recipe fixes the RNG seed at 42.

```bash
# Run on each of the 16 nodes with the step 4 function.
(
  export PARITY_DIR=$WORK/ernie256k-memory-precision
  export MCORE_DSA_STABLE_INDEXER_TOPK=1
  for LAYOUT in single uniform skew; do
    case "$LAYOUT" in single) DOCS=1 ;; uniform) DOCS=8 ;; skew) DOCS=12 ;; esac
    export LAYOUT DOCS
    export INPUT="$DATA/$LAYOUT.jsonl" LOGS="$PARITY_DIR/$LAYOUT"
    mkdir -p "$LOGS"
    ( export PYTHONPATH="$MCORE_RETAIN:$BRIDGE/src:${PYTHONPATH:-}"
      export ERNIE55_COMPARE_HEADS="$LOGS/retain-capture"
      run_arm retain-check true model.recompute_num_layers=1 train.train_iters=1 )
    ( export PYTHONPATH="$MCORE:$BRIDGE/src:${PYTHONPATH:-}"
      export ERNIE55_COMPARE_HEADS="$LOGS/release-capture"
      run_arm release-check true model.recompute_num_layers=2 train.train_iters=1 )
  done
)
# After all workers exit, run on one node.
for layout in single uniform skew; do
  d="$WORK/ernie256k-memory-precision/$layout"
  python "$EX/head_parity_256k.py" "$d/retain-capture" "$d/release-capture"
done
```

On the twelve-document fixture, the public Bridge/MCore pair used production MoK expert arithmetic with stable D-indexer ties and produced 64 rank files per arm. All 13 recorded fields on every rank were byte-identical between retain/group 1 and release/group 2. The comparison covered 1,048,576 padded positions across four replicas and 1,048,148 valid CE entries for each head. At the sampled valid positions, it compared 173,432 main and 166,260 MTP logit elements, and 4,358,144 main and 4,177,920 MTP head-input gradient elements. All ten error metrics were zero. These are observations of the first pre-update microbatch. The independent Swift comparison and the earlier eight-document full GA8 parameter-gradient measurement check different quantities.

Each of these instrumented arms completed one full GA8 optimizer update. Across all 64 ranks, allocated/reserved peaks were 222.659/224.350 GiB with retain/group 1 and 157.527/165.072 GiB with release/group 2. These one-update peaks include precision instrumentation and a stable D indexer. Use the clean twelve-update native-selector run above for a production memory number.

### 6c. Compare the 256K training forward and head gradients with Swift

The accuracy run uses the original Base weights, real router choices, one MTP layer, MoK with MoonEP and full recomputation. Use the three fixed input files and the converted checkpoint from steps 2–3. Keep the throughput runs in their original environment. The following commands set up separate checkouts for the accuracy measurement.

**1. Fetch both source revisions.** Run this once in the shared workspace. The Bridge revision includes the native source patch, both head recorders, the comparator and the reference-arithmetic switches.

```bash
git -C "$BRIDGE" fetch origin refs/pull/7000/head
git -C "$BRIDGE" worktree add --detach "$WORK/Bridge-accuracy" 88efc31c89d7e0ccf279ecaa9d4fd22dbe2ec5fd
git -C "$MCORE" fetch origin refs/pull/9506/head
git -C "$MCORE" worktree add --detach "$WORK/MCore-accuracy" 6ce5dfe9c8c9988972f4a5123e1e6a6c51231636
export BRIDGE=$WORK/Bridge-accuracy MCORE=$WORK/MCore-accuracy
export EX=$BRIDGE/examples/models/ernie55
```

**2. Build the BF16 MoK variant on a GB300.** Use the SM103 development environment from step 1. All workers must be able to read the resulting checkout and extension. The build generates the SiTU lookup table from BF16 inputs and FP32 elementary functions on the GPU.

```bash
export MOK=$WORK/MoK-accuracy
git clone https://github.com/QiZhangNV/mixture-of-kittens.git "$MOK"
git -C "$MOK" checkout --detach ee7bdaba515fd3d557105d71809ec74cd523d5a1
git -C "$MOK" submodule update --init --recursive
git -C "$MOK/third_party/ThunderKittens" apply "$MOK/patches/thunderkittens-mxfp8-mn-major.patch"
git -C "$MOK" apply --check "$EX/mok-sm103-bf16-reference.patch"
git -C "$MOK" apply "$EX/mok-sm103-bf16-reference.patch"
make -C "$MOK" ARCH=SM103 PYTHON="$(command -v python)"
export PYTHONPATH=$MOK:$MCORE:$BRIDGE/src:${PYTHONPATH:-}
python -c 'import pathlib,os,mok._C as c; assert c.strict_bf16_reference_situ and pathlib.Path(c.__file__).resolve().is_relative_to(pathlib.Path(os.environ["MOK"]).resolve())'
```

**3. Run one update for each packing.** In the 16-node allocation, run the step-4 `run_arm` function on each node with its own `NODE_RANK`. Export `BRIDGE`, `MCORE`, `EX`, `MOK` and `PYTHONPATH` above on every node. Set `WANDB_PROJECT` to a project writable by the logging rank. Use separate capture directories. The model uses per-document zigzag at CP16, EP64, GBS32, MBS1 and GA8.

```bash
export MCORE_DSA_STABLE_INDEXER_TOPK=1 MCORE_DSA_PRESERVE_SELECTED_SLOT_ORDER=1
export MCORE_DSV4_GROUP_PROJ_BMM=1 NCCL_MAX_P2P_NCHANNELS=8 NCCL_GRAPH_REGISTER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=${WANDB_PROJECT:-ernie55-256k-accuracy}
for LAYOUT in single uniform skew; do
  case "$LAYOUT" in single) DOCS=1 ;; uniform) DOCS=8 ;; skew) DOCS=12 ;; esac
  INPUT=$DATA/$LAYOUT.jsonl
  export ERNIE55_COMPARE_HEADS="$LOGS/accuracy-$LAYOUT-heads"
  run_arm "accuracy-$LAYOUT" true train.train_iters=1 \
    optimizer.lr=1e-5 optimizer.min_lr=1e-5 scheduler.lr_decay_style=constant \
    scheduler.lr_warmup_iters=5 scheduler.lr_decay_iters=6 \
    model.recompute_num_layers=1 model.ernie_reference_rmsnorm=true \
    model.ernie_fused_mhc=false model.ernie_vha_batched_mix=false \
    model.ernie_dsa_preserve_cp_key_slots=true \
    '+model.moe_megakernel_backend_config.reference_pre_fc2_gate=true' \
    "logger.wandb_project=$WANDB_PROJECT" "logger.wandb_exp_name=ernie55-accuracy-$LAYOUT"
done
```

**4. Capture the original Swift trainer and compare.** In the reference environment, prepare the same three JSONL rows and launch the original trainer with its head recorder. The companion guide gives the setup, complete multi-node commands and the expected 64 files for each packing. Use a shared `WORK` and a fresh capture directory for each arm. [Swift head capture and input preparation](SWIFT-HEAD-PARITY.md)

After recording both trainers, run this on one node with NumPy. The comparator reconstructs each dense replica separately, checks the actual targets, masks and global coordinates, and returns the two head scores.

```bash
for LAYOUT in single uniform skew; do
  python "$EX/head_parity_256k.py" "$WORK/swift-heads/$LAYOUT" "$LOGS/accuracy-$LAYOUT-heads" --check
done
```

The preregistered thresholds for each head are 0.001% for mean valid-token CE, 0.5% for relative L2 and 1% for maximum difference divided by the reference peak. The latter two checks cover valid per-token CE, sampled pre-multimax logits and sampled head-input gradients. The CLI prints relative errors as fractions and exits nonzero on a failed gate.

#### Independent reference measurements

The commands above were exercised on the same 64 GB300s for both implementations. Each fixture repeats one distinct 256K row across four dense replicas. The table pools the first pre-update microbatch across those replicas. The raw-logit comparison uses 163 vocabulary columns at each shared valid position. The head-input gradient comparison uses all 4,096 channels at those positions. The mean CE covers every valid target in every replica.

| Documents per row | Valid targets per head per replica | Sampled positions per replica, main / MTP | Unequal sampled raw logits, main / MTP | Mean CE relative error, main / MTP | Head-input gradient relative L2, main / MTP |
|---|---:|---:|---:|---:|---:|
| 1 | 262,136 | 257 / 256 | 0 / 0 | 0.00000136% / 0.00000072% | 0.00321848% / 0.00003684% |
| 8 equal | 262,080 | 264 / 256 | 0 / 0 | 0.00000145% / 0.00000085% | 0.00629173% / 0.00000736% |
| 12 unequal | 262,037 | 266 / 255 | 0 / 0 | 0.00000141% / 0.00000075% | 0.00128327% / 0.00066723% |

Both heads passed every stated threshold on all three fixtures. A separate single-document test saved the main and MTP head inputs produced by original Swift and by true routed MoK + MoonEP. Projecting those inputs through the released head weights gave exactly equal BF16 values for all 52,745,957,376 valid-position logits per head on one dense replica. Projected Swift logits also matched the raw columns sampled directly by its trainer. The largest absolute differences from the native Swift per-token CE were 1.91e-6 for the main head and 2.38e-6 for MTP.

The complete single-document decoder backward still exceeds the fixed thresholds. At the inputs to blocks 1, 22 and 43, the full four-stream gradient relative L2 differences against Swift were 2.26123%, 0.98234%, 0.36896%, respectively. Two fresh runs of original Swift on the same 64 physical GPUs and the same rank assignments differed by 2.30726%, 1.16300%, 0.05733% at the same points. Each comparison covers all 262,144 tokens on one replica. The unreduced CE gradient coefficients agreed at every token in both heads. These measurements cover the first microbatch. Parameter-gradient and multi-update optimizer agreement with Swift remain open.
