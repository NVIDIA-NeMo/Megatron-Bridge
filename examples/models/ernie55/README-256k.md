# Reproduce ERNIE 5.5 Flash 256K BF16 training on 64 GB300s

This experiment trains the public ERNIE-5.5-Flash-Base checkpoint with both its main LM head and MTP head. It compares natural MoK routing with the same routing under MoonEP expert replication. Use one 64-GPU NVLink domain (16 nodes × 4 GB300s). TP1, PP1 and CP16 give four dense data replicas. EP64 spans all ranks. Each replica sees one packed 262,144-position row per microbatch and accumulates eight microbatches. An optimizer update therefore contains 32 rows. The recipe runs eagerly with full uniform recomputation, two hybrid symbols per group, BF16 weights and expert math, FP32 router scores, and a CUDA token-chunked head (4,096 rows) on both outputs.

**Accuracy status.** The native throughput recipe exceeds the original Swift head error limits on eight equal documents. Reference arithmetic passes both heads on the first microbatch for all three packings. On eight equal documents, using the original BF16 weight-gradient accumulation also passes the sampled shared LM/MTP weight update. When each trainer performs its own updates, the embedding and last shared-expert FC2 updates and the subsequent predictions still exceed the fixed thresholds. Independent Swift restarts also exceed the post-update head thresholds. [Throughput by packing](#packing-zigzag-and-moonep) · [Optimizer and post-update checks](OPTIMIZER-PARITY.md#capture-both-heads-after-the-first-nonzero-weight-update)

On the first microbatch after that nonzero update, five original Swift runs had a median MTP raw-logit distance of 3.061%. Four independent runs of the same Bridge recipe had a median of 7.165%. Across the twenty Swift–Bridge pairs the median was 6.191%. The main-head medians were 1.570%, 1.640% and 1.634%, respectively. These comparisons use symmetric relative L2 and count each run once per pair. [All 36 comparisons, per-token loss and reproduction](benchmarks/post-update-repeats.md)

At identical updated weights, original Swift and strict MoK + MoonEP matched every sampled raw logit for the main and MTP heads on eight equal documents. The full valid-token CE relative L2 was 0.00000774% on main and 0.00001579% on MTP. The original Swift process also reproduced its own captured heads bitwise after HF export and reload. [Common-state reproduction and three-way results](COMMON-STATE-256K.md)

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

#### Variable packed lengths with a fixed MoK workspace

For training with variable packed lengths, install the MoK wheel from the MCore workspace fix, then set `max_local_tokens` to the largest **physical token count per EP rank in one microbatch**. Keep the historical environment from step 1 if you also need the original timing baselines. In the environment for your variable-length run, build the wheel from the companion installer:

```bash
export MCORE_MOK_CAPACITY=$WORK/Megatron-LM-mok-capacity
git -C "$MCORE" fetch origin refs/pull/9506/head
git -C "$MCORE" worktree add --detach "$MCORE_MOK_CAPACITY" 1b3027de9ca4f42d67d26d7c8d5e2dba7993b393
export UV_PROJECT_ENVIRONMENT=$(python -c 'import sys; print(sys.prefix)')
export MOK_ARCH=SM103
bash "$MCORE_MOK_CAPACITY/docker/common/install_mok.sh" \
  ee7bdaba515fd3d557105d71809ec74cd523d5a1 "$WORK/mok-capacity-wheels"
python -c 'from mok.functional import MoKConfig; assert MoKConfig(max_local_tokens=16384).max_local_tokens == 16384'
```

Deploy the wheel in the torchrun environment on every node. For this TP1, CP16 benchmark, one microbatch contains one row of up to 262,144 positions including padding. Each rank therefore needs capacity for at most 16,384 tokens. Set `INPUT` to your tokenized JSONL and `DOCS` to the maximum document count per row, and add the override to either arm:

```bash
run_arm mok_varlen false '++model.moe_megakernel_backend_config.max_local_tokens=16384'
run_arm moon_varlen true '++model.moe_megakernel_backend_config.max_local_tokens=16384'
```

MoK reserves the communication buffers on first use and processes each microbatch at its actual token count. All ranks in the EP group use the same capacity and the same local token count on each invocation. Both values must be at least 512 and divisible by 256. For a different TP1 recipe, divide the maximum combined padded length across the rows in one microbatch by CP and round the capacity up to a supported size. Gradient accumulation does not enter this bound. The workspace capacity can be set independently of `macrobatch_size`, which controls kernel scheduling.

**Reproduce the variable-length throughput check.** From the single-document fixture in step 3, write one JSONL line per sample. Each group of four consecutive lines has the same length so the four DP replicas agree on the MoK input shape. The eight groups form one optimizer update (GBS32, microbatch 1, GA8):

```bash
python - "$DATA/single.jsonl" "$DATA/variable.jsonl" <<'PY'
import json, sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text())
assert len(source["input_ids"]) == len(source["labels"]) == 262144
with Path(sys.argv[2]).open("x") as stream:
    for length in (131072, 196608, 262144, 131072, 196608, 262144, 131072, 262144):
        row = {key: source[key][:length] for key in ("input_ids", "labels")}
        stream.write((json.dumps(row, separators=(",", ":")) + "\n") * 4)
PY
export DOCS=1 INPUT=$DATA/variable.jsonl
variable_args=(dataset.pack=false dataset.dataloader_type=single train.train_iters=9)
# Use the original wheel from step 1 for the first run and the workspace-fix wheel for the second.
run_arm variable_original true "${variable_args[@]}"
run_arm variable_capacity true "${variable_args[@]}" '++model.moe_megakernel_backend_config.max_local_tokens=16384'
```

The single-document loader preserves file order and repeats the 32 rows each update. After CP16 zigzag splitting, each GPU sees 8,192, 12,288, 16,384, 8,192, 12,288, 16,384, 8,192 and 16,384 positions. For physical tokens/s/GPU, divide **98,304** by the mean measured step time in seconds. The fixed 256K control uses **131,072** positions/GPU/update. The `summarize` command in step 5 assumes the fixed 256K numerator.

Each pair ran once on the same allocation of 64 GB300 GPUs with BF16 MoK + MoonEP and full recomputation. Throughput includes every update in the listed window.

| Input row lengths | Updates | Original workspace (tokens/s/GPU) | Fixed capacity of 16,384 (tokens/s/GPU) | Change |
|---|---:|---:|---:|---:|
| 256K | 4–12 | 3,755.42 | 3,741.90 | −0.36% |
| 128K, 192K, 256K | 4–9 | 3,659.41 | 3,727.37 | +1.86% |

Across the six measured variable-length updates, step times ranged from 25.70 to 28.68 seconds with the original workspace and from 25.86 to 27.04 seconds with fixed capacity.

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

### Packing, zigzag and MoonEP

Each row below gives physical tokens/s/GPU for 12-update BF16 jobs using the source versions in step 1 and two-symbol full recomputation. The three stages for each packing ran on the same 64 GPUs. The three packings used different racks. Each stage starts from the original checkpoint. The measurement includes every update from 4 through 12 and uses the same 131,072 physical token positions/GPU/update for every stage.

| Documents per 256K row | MoK, contiguous | MoK, all-zigzag | MoK + MoonEP, all-zigzag | Gain from zigzag | Further gain from MoonEP | Final seconds/update | Final peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One × 256K | 2,849.4 | 3,099.0 | 3,760.0 | +8.76% | +21.33% | 34.860 | 182.63 |
| Eight × 32K | 3,861.3 | 3,919.1 | 4,556.0 | +1.50% | +16.25% | 28.769 | 173.08 |
| Twelve unequal | 3,446.8 | 3,668.3 | 4,330.1 | +6.43% | +18.04% | 30.270 | 174.16 |

The contiguous baseline converts D and MTP-D into their internal zigzag layout and converts back. The all-zigzag stack gives H a per-document zigzag layout and feeds D directly. In the isolated CP16 H layer, forward plus backward with cached metadata took 55.88 and 47.09 ms with the two layouts on one document, and 39.25 and 39.10 ms on eight equal documents. MoonEP saved a further 4.675 seconds per eight-document optimizer update. [Performance and precision figures (PDF)](benchmarks/ernie55-256k-bf16-ablation.pdf) · [Per-step timings, allocator peaks and nine W&B runs (CSV)](benchmarks/ernie55-256k-bf16-ablation.csv)

Prepare the three fixture files from step 3. In the same environment on each node, define `run_arm` from step 4 and run the block below. Set `WANDB_PROJECT` to a project you can write to. Use a fresh log directory shared by all nodes for each benchmark invocation. For node-local fixture staging, copy each JSONL to the same path on all nodes and set `INPUT` to that path.

```bash
(
  set -eo pipefail
  : "${WANDB_PROJECT:?Set a W&B project for these runs}"
  unset ERNIE55_COMPARE_HEADS
  export MCORE_DSA_STABLE_INDEXER_TOPK=0 NCCL_NVLS_ENABLE=0 CUDA_DEVICE_MAX_CONNECTIONS=1
  export WANDB_MODE=online
  common=(train.train_iters=12 model.recompute_num_layers=2 profiling.nvtx_ranges=false
          logger.log_interval=1 "logger.wandb_project=$WANDB_PROJECT")
  for LAYOUT in single uniform skew; do
    case "$LAYOUT" in single) DOCS=1 ;; uniform) DOCS=8 ;; skew) DOCS=12 ;; esac
    export LAYOUT DOCS INPUT="$DATA/$LAYOUT.jsonl" LOGS="$WORK/ernie256k-ablation-$LAYOUT"
    mkdir -p "$LOGS"
    for arm in mok_contiguous mok_zigzag moon_zigzag; do
      case "$arm" in
        mok_contiguous) layout=contiguous; zigzag=false; moon=false ;;
        mok_zigzag)     layout=zigzag;     zigzag=true;  moon=false ;;
        moon_zigzag)    layout=zigzag;     zigzag=true;  moon=true  ;;
      esac
      run_arm "$arm" "$moon" "${common[@]}" "model.cp_partition_mode=$layout" \
        "model.dsv4_cp_zigzag=$zigzag" "logger.wandb_exp_name=ernie256k-$LAYOUT-$arm"
    done
  done
)
# On one node, after every node has finished.
for layout in single uniform skew; do
  for arm in mok_contiguous mok_zigzag moon_zigzag; do
    python "$EX/benchmark_256k.py" summarize "$WORK/ernie256k-ablation-$layout"/"$arm"-node*.log
  done
done
```

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

For an independent reference that uses contiguous CP, export the same NPZ schema as the probe. A file contains its dense replica ID (`dp`), global padded document boundaries (`bounds`), local rows' physical global coordinates (`pos`), `tokens`, the actual `main_labels` and rolled `mtp_labels` with invalid targets set to −100, and both `main_ce` and `mtp_ce`. Store the sampled head-input positions in `sample` and use this order for `main_dh` and `mtp_dh`. When the raw logits have different physical owners, store their positions in `logit_sample` and use that order for `main_logits` and `mtp_logits`. For files without `logit_sample`, all four tensors use `sample`. The combined logits and input gradients must cover the same sampled positions in each dense replica. Raw logits take vocabulary IDs 0, 1237, 2474, …; the hidden-gradient samples contain every channel. The comparator reconstructs each full replica independently of CP sharding and joins sampled values on global position. For each implementation write `rank00.npz` through `rank63.npz`, covering every global position in each of the four dense data replicas. Use the same vocabulary IDs and loss scaling in both trainers. For matching samples, select physical positions where `pos % 2048` is 15 or 1023, plus each padded document end minus two. The comparator uses the intersection of sampled coordinates. Check coverage before interpreting a small difference: for the fixed twelve-document skew fixture, the rule gives 268 sampled physical coordinates per dense replica, including 266 valid main targets and 255 valid MTP targets. After recording the original Swift run and the candidate run into separate directories, use Swift as the reference denominator.

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
git -C "$BRIDGE" worktree add --detach "$WORK/Bridge-accuracy" b336a6e6b48e15b5669b070fea1f57082b71fd68
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

The complete single-document decoder backward still exceeds the fixed thresholds. At the inputs to blocks 1, 22 and 43, the full four-stream gradient relative L2 differences against Swift were 2.26123%, 0.98234%, 0.36896%, respectively. Two fresh runs of original Swift on the same 64 physical GPUs and the same rank assignments differed by 2.30726%, 1.16300%, 0.05733% at the same points. Each comparison covers all 262,144 tokens on one replica. The unreduced CE gradient coefficients agreed at every token in both heads. These measurements cover the first microbatch. The eight-document optimizer comparison below covers three parameter matrices. Other parameter gradients and the longer training trajectory require separate checks.

#### Eight-document throughput and precision

Both unprofiled arms used the same 64 GB300s, released weights, eight 32,768-token documents, true MoK + MoonEP, full zigzag, the CUDA head and eight microbatches per update. Each arm completed ten optimizer updates. The table uses every update from 4 through 10. Peak allocated memory is the largest PyTorch watermark across all 64 GPUs.

| Configuration | Mean update, s | Tokens/s/GPU | Peak allocated, GiB | W&B |
|---|---:|---:|---:|---|
| Native throughput arithmetic, two-symbol recompute | 30.100 | 4,354.5 | 173.08 | [run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/td9wjdgh) |
| Reference arithmetic, one-symbol recompute | 56.264 | 2,329.6 | 216.96 | [run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/5dgrccsw) |

The reference-arithmetic arm includes score-ordered stable D top-k and preserved physical key slots, reference routed BF16 arithmetic and RMSNorm, eager mHC and VHA, and one-symbol recompute. The measured difference includes all these choices.

The native throughput arm exceeded at least one of the fixed Swift head gates when tested separately with the public four-replica capture. Its sampled raw logit relative L2 was 1.605553% for main and 8.014075% for MTP. A third accuracy arm retained reference arithmetic and changed only recompute grouping to two symbols. It passed the same gates.

A separate unprofiled pair isolates recompute grouping within reference arithmetic. Both arms started from the same checkpoint on the same 64 GPUs. G2 ran first. Each arm completed seven updates. The step window is updates 4–7, with 131,072 physical positions per GPU per update. The memory columns use the maximum whole-run PyTorch allocator watermark across all 64 GPUs.

| Reference arithmetic | Mean update, s | Tokens/s/GPU | Peak allocated, GiB | Peak reserved, GiB | W&B |
|---|---:|---:|---:|---:|---|
| One symbol per checkpoint (G1) | 56.382 | 2,324.72 | 216.96 | 218.74 | [G1](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/tb2o5p0o) |
| Two symbols per checkpoint (G2) | 56.286 | 2,328.67 | 175.57 | 176.43 | [G2](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/117318ku) |

G2 saved 41.39 GiB allocated (19.08%) and 42.31 GiB reserved. The 0.17% difference in throughput comes from four warm updates per arm. In the separate four-replica head test, every one of the seven comparison metrics on each head was identical for G1 and G2, including zero sampled raw-logit error.

To repeat the timing pair, complete section 6c steps 1–3 and run this on all 16 nodes. Leave the section 6c environment and the step 4 `run_arm` function in place. Each invocation starts from the released checkpoint and writes a fresh set of per-node logs.

```bash
unset ERNIE55_COMPARE_HEADS
DOCS=8
INPUT=$DATA/uniform.jsonl
strict_args=(
  optimizer.lr=1e-5 optimizer.min_lr=1e-5 scheduler.lr_decay_style=constant
  scheduler.lr_warmup_iters=5 model.ernie_reference_rmsnorm=true
  model.ernie_fused_mhc=false model.ernie_vha_batched_mix=false
  model.ernie_dsa_preserve_cp_key_slots=true
  '+model.moe_megakernel_backend_config.reference_pre_fc2_gate=true'
  "logger.wandb_project=$WANDB_PROJECT"
)
for group in 2 1; do
  run_arm "strict-g$group" true "${strict_args[@]}" "model.recompute_num_layers=$group" \
    train.train_iters=7 scheduler.lr_decay_iters=12 "logger.wandb_exp_name=ernie55-uniform-strict-g$group"
done
# Once both runs have completed, execute on one node.
for group in 2 1; do
  python "$EX/benchmark_256k.py" summarize --first-step 4 --last-step 7 "$LOGS"/strict-g"$group"-node*.log
done
```

#### Opt-in MTP head backward for gradient comparisons

Add `model.ernie_mtp_strict_head_backward=true` to a strict run to reproduce the BF16 MTP head gradients. The standard recipe uses `false`. Both settings use the same chunked forward and ordinary main-head backward. The opt-in MTP backward computes the corrected-logit gradient in FP32 with the reference normalization order, rounds it to BF16 before multimax backward, and contracts up to 1,024 rows of raw-logit gradients at a time. A 512-row softmax tile and a 1,024-row projection tile need a 393 MiB BF16 raw-gradient buffer at vocabulary size 201,216. The implementation uses PyTorch and the Bridge multimax operator included in this checkout.

For this check, use Bridge commit `fe1504750075c9907eaa8071f147d5f3bb164672` in the worktree command in section 6c step 1. To repeat the single-document GA8 run, retain that section’s strict environment and use the `run_arm` function and `strict_args` from above on every node:

```bash
(
  export DOCS=1 INPUT="$DATA/single.jsonl" NCCL_ALGO=Ring
  run_arm strict-mtp-head-single true "${strict_args[@]}" model.recompute_num_layers=1 \
    model.ernie_dsa_contiguous_backward=true model.ernie_mtp_strict_head_backward=true \
    train.train_iters=1 "logger.wandb_exp_name=ernie55-single-strict-mtp-head"
)
```

One 64-GB300 update processed all eight microbatches. On the first microbatch, all 52,747,567,104 MTP raw-logit gradients and all 1,073,741,824 head-input gradients on one reconstructed dense replica matched independent Swift training bitwise. The LM/MTP losses were 1.285155/1.493400 with zero skipped or NaN updates. In a separate single-document whole-branch comparison, MTP-to-trunk gradient relative L2 and peak error changed from 0.3793%/1.3575% to 0.3772%/0.9050%. The independent thresholds were 0.5% relative L2 and 1% of the reference peak.

A separate complete GA8 update on the 12 unequal documents passed both independent Swift head gates. All sampled MTP head-input gradients and raw logits matched Swift exactly. Each of the 64 ranks ran eight standard main-head backwards and eight strict MTP backwards. Four GPU workers also passed the saved-projection, recomputed-projection and head-dispatch tests.

For the isolated MTP head forward and backward on four real 16,384-token shards, the median changed from 58.35 ms to 92.98 ms over twelve warmed observations per mode. Peak allocated above resident inputs and weights changed from 7.83 GiB to 7.61 GiB. Whole-training step times depend on the rest of the model.

#### Match the original shared output-weight gradient

The main and MTP heads share one output weight. With `model.ernie_head_reference_wgrad=true`, the BF16 projection runs on contiguous physical token owners. A zigzag stack sends hidden states and labels to these owners, then receives the per-token loss, optional predictions and head-input gradient in stack order. One BF16 GEMM per head spans every local token. Autograd adds the two head gradients in BF16 before the ordinary FP32 accumulation over microbatches. The default `false` keeps the fused FP32 accumulation in 4,096-row tiles and the stack's token layout.

On a 256K row at CP16, the opt-in weight-gradient GEMM retains a temporary 6.14 GiB BF16 raw-logit gradient for the current head. Inter-rank exchanges carry the 4,096-wide hidden state and its gradient, labels and token-level results. Both heads can use the option alongside the strict MTP backward described above. The global-position probe records the logit and head-input sampling coordinates independently. In a complete two-update comparison on eight equal documents, the true LM weight update had 0.03947% relative L2 and 0.37453% peak error against Swift. All 807,229 sampled LM coordinates met the peak gate. The embedding and final shared-expert update still exceed the preset limits. [Two-update reproduction and results](OPTIMIZER-PARITY.md#shared-output-weight-rounding)

#### Opt-in main head backward for gradient comparisons

Check out Bridge commit `87c53ed871c86390e646511d40f693237bb4c9bf` or a descendant. Add `model.ernie_main_strict_head_backward=true` to the same TP1 chunked training recipe. This applies the reference CE and multimax rounding to the main head with its own per-token loss gradient. The MTP flag and the shared output-weight gradient flag are independent. All three switches default to `false`.

For one complete 256K row of eight equal documents, we replayed Swift's actual main-head hidden state, output weight, labels and incoming loss gradient. The regular backward differed from Swift at 31,952,252 of 1,073,741,824 head-input gradient values (0.002268% relative L2). With the main strict option, all values agreed exactly. All 1,024 raw-logit gradient slabs also matched Swift bitwise, covering 52,747,567,104 BF16 values. [Original main-head capture](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/k78wca3i)

In a separate two-update G2 comparison, both candidates enabled the reference shared output-weight gradient. They also used the same diagnostic BF16-return implementation on original contiguous token owners for both uses of the final shared-expert FC2. Each trainer used its own router updates and optimizer state. Enabling the strict main backward reduced the sampled LM preclip gradient error from 0.00169726% to 0.00000258% relative L2 and the actual LM update error from 0.113838% to 0.052419% against the same Swift run. The LM update peak error remained 0.374532%. The full final shared-expert FC2 update error remained above the 0.5% L2 gate, changing from 2.08444% to 2.07966%. [Swift](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/cy21lsw1) · [Regular main backward](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/5zppnu2a) · [Strict main backward](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/czbkyt3t)

#### First GA8 parameter gradients on eight equal documents

The two trainers accumulated the same eight microbatches on each of four dense replicas, checked the input tokens and main labels at every physical position, and ran their native distributed optimizer reduction. All 1,656,750,080 initial FP32 master-weight values in the three matrices below matched bitwise. The LM and embedding weights are independent. The last shared-expert FC2 is shared with MTP. The comparison reconstructs weights by logical coordinates across the two optimizer shard layouts, and reads the FP32 gradient before clipping.

| Native reduced gradient | Elements | Relative L2 difference | Max error / Swift peak | Gate |
|---|---:|---:|---:|---|
| LM output weight | 824,180,736 | 0.177565% | 0.155163% | Pass |
| Last shared-expert FC2 | 8,388,608 | 0.087456% | 0.102957% | Pass |
| Token embedding, all rows | 824,180,736 | 0.516128% | 0.305729% | Fail |
| Embedding, BOS row (token 1) | 4,096 | 0.516128% | 0.305729% | Fail |
| Embedding, remaining rows | 824,176,640 | 0.463432% | 0.682193% | Pass |

The native whole-model gradient norms were 110.937294 for Swift and 110.546158 for Bridge. Both loss scales were 1. The measured warmup step used a zero learning rate for these parameter groups. Its gradient comparison is valid, and the unchanged weights provide no test of a nonzero optimizer update. Swift needed a one-time allocator cache release before its native distributed gradient synchronization to complete on this allocation. Live allocated bytes on all ranks were identical immediately before and after that cache release.

#### First nonzero parameter update on eight equal documents

Swift and Bridge each ran two consecutive GA8 updates in one process group. Both consumed the same tokens and shifted main labels for all 32 microbatch/replica pairs in each update. Both started update 2 with bitwise-identical sampled FP32 master weights at loss scale 1. The native optimizer retained its Adam moments and Muon momentum from the LR-0 update. At the first nonzero LR (2e-6), all three sampled update vectors exceeded the 0.5% relative L2 and 1% peak-relative limits. The native whole-model gradient norms were 80.538742 for Swift and 105.850609 for Bridge. The runs used separate 64-GPU racks. [Original Swift run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/62nxkb13) · [Bridge run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/rkt1q2vg)

| Parameter | Sampled coordinates | Preclip gradient relative L2 | Master update relative L2 | Max update error / Swift update peak |
|---|---:|---:|---:|---:|
| LM output weight | 807,229 | 4.10352% | 6.27027% | 69.5693% |
| Token embedding, including BOS | 811,321 | 51.1103% | 38.8568% | 170.901% |
| Last shared-expert FC2 | 8,217 | 1.52254% | 6.48272% | 27.3452% |

Embedding updates also differed away from BOS. The relative L2 was 36.1015% for the complete BOS row and 39.6441% for the sampled non-BOS rows. Across two fresh runs per trainer, the four Swift-to-Bridge **first-update** BOS gradient comparisons ranged from 0.30639% to 0.54442% relative L2 (two passed). The within-Swift and within-Bridge pairs differed by 0.35908% and 0.16287%, respectively.

To run the two Bridge updates, use the section 6c environment and `strict_args` from the timing pair above. In the original Swift environment, use its native training entry with the eight-document cache, deterministic settings and optimizer arguments from the Swift guide. Set `--train_iters 2 --lr_decay_iters 6`, retain the five-update warmup, and let the same optimizer run both updates. The Swift head recorder exits after one microbatch and is only suitable for the head test. [Swift setup and training arguments](SWIFT-HEAD-PARITY.md)

```bash
unset ERNIE55_COMPARE_HEADS
DOCS=8
INPUT=$DATA/uniform.jsonl
run_arm strict-two-updates true "${strict_args[@]}" model.recompute_num_layers=1 \
  train.train_iters=2 scheduler.lr_decay_iters=6 logger.wandb_exp_name=ernie55-uniform-strict-two-updates
```

To reproduce the parameter measurement in either trainer, record the FP32 master values immediately before and after each native optimizer step, and the reduced gradients passed to the gradient norm routine before clipping. Map each local shard to the global row-major index for its parameter. Select indices divisible by 1,021 and additionally keep all 4,096 entries of embedding row 1 (BOS). Deduplicate replicated coordinates, verify input IDs, labels, pre-step weights and optimizer-state continuity, then compute `delta = post.double() - pre.double()` on each selected coordinate. Divide the L2 norm of the update difference by the Swift update L2 norm, and its largest absolute component by the largest absolute Swift update. Ordinary training logs give the native gradient norm. The public recorder and NumPy comparator reproduce these measurements in both trainers. [Two-update capture, scoring and expected files](OPTIMIZER-PARITY.md)

#### Match the original BF16 router-bias update under zigzag CP

When training with count-based expert correction, the original uses BF16 counters for physical contiguous CP stripes. Rounding each rank's accumulated load before the global sum makes CP ownership observable. A zero optimizer learning rate still permits the correction bias to change. For the reference-arithmetic training command above, use a Megatron-Core checkout containing [the BF16 router-count option](https://github.com/jingqiny-99/Megatron-LM/commit/3ba272c0e128d6e0b5147142363710fcbcaa2671), set `NCCL_ALGO=Ring` as on the deterministic reference, and append both overrides.

```bash
model.moe_router_bias_bf16_counts=true \
model.ernie_router_bias_contiguous_cp_counts=true
```

The Bridge option collects selected expert counts by physical token position within each packed training row. Before the optimizer update, one integer reduce-scatter on the CP group delivers each contiguous owner's counts for every local router and microbatch. Each router accumulates its own counts in BF16 in microbatch order. Megatron-Core then performs its ordinary count all-reduce and correction-bias update. The router uses its actual padding mask. The data, attention and expert computation keep their configured partition and logical routes. The option supports packed THD, static CP, TP=1 and eager router execution. Both overrides default to false.

On 64 GB300s with CP16, EP64 and eight 32,768-token documents per row, the first GA8 update had 43 routed modules including MTP. Their 22,016 FP32 corrections matched independent Swift before and after the first update on every rank. Under the default integer/FP32 correction, 1,827 of the 22,016 entries on each rank differed after the update, even though the exact full-step global integer counts were equal. The first router then selected different experts for 172 of 262,144 tokens in one DP replica.

The source-enabled two-update comparison used the reference-arithmetic recipe, one-symbol full recompute and the default MTP-head backward. Against original Swift, update 2 had native whole-model gradient norms of 80.2315 and 80.5387 respectively. FP32 update comparisons still exceed the 0.5% relative-L2 or 1% relative-peak gate.

| Parameter | Step-2 preclip gradient relative L2 | Step-2 FP32 update relative L2 | Step-2 FP32 update relative peak |
|---|---:|---:|---:|
| LM head | 0.18961% | 0.20479% | 7.25655% |
| Token embedding | 0.58638% | 0.60379% | 15.63181% |
| Final shared expert FC2 | 0.09009% | 2.89354% | 16.48452% |

The normal 64-GPU training entry also ran one G2 full-recompute GA8 update on a single 256K document and on twelve unequal documents. For each pack, an independent observer summed the actual boolean route maps over contiguous token intervals and checked all 180,355,072 local integer histogram entries and all 1,409,024 owner-local BF16 counters across 64 ranks. Every entry matched, including the MTP router. This checks count ownership across packing shapes. The two-update comparison against independent Swift above uses the eight-document case.

A clean same-rack pair measured both settings on the eight-document reference-arithmetic recipe with G2 full recompute. The jobs used identical 64 GB300s, NCCL Ring and default MTP-head backward. Each ran 12 optimizer updates with graphs and profiling disabled. Throughput uses the summed time for every update from 4 through 12. Memory is the largest PyTorch watermark over all 64 ranks.

| G2 router correction | Mean update, s | Physical tokens/s/GPU | Peak allocated, GiB | Peak reserved, GiB |
|---|---:|---:|---:|---:|
| Default integer counts | 56.1764 | 2,333.22 | 175.5705 | 176.4316 |
| Contiguous owners and BF16 counts | 56.1671 | 2,333.61 | 175.5796 | 176.4414 |

The measured difference was +0.0167% throughput and +0.0091 GiB allocated. This single pair does not resolve a systematic step-time cost.

#### Optional contiguous CP owners for D attention backward

The opt-in setting `model.ernie_dsa_contiguous_backward=true` keeps the forward zigzag layout. In backward it moves query-side tensors to their contiguous CP owners, runs sparse attention backward there, reduce-scatters the BF16 key gradient to the contiguous owners, and returns the local gradients to zigzag. It adds CP communication to every D backward including MTP. The benchmark table above uses the default backward.

After completing steps 1–4 of section 6c, run the following on all 16 nodes for a new twelve-document capture. It uses the step-4 `run_arm` function and the strict arithmetic switches from the previous example. Set the same `NCCL_ALGO=Ring` in both frameworks when comparing reduction order.

```bash
export NCCL_ALGO=Ring
DOCS=12
INPUT=$DATA/skew.jsonl
export ERNIE55_COMPARE_HEADS="$LOGS/accuracy-skew-d-backward-heads"
run_arm accuracy-skew-d-backward true train.train_iters=1 \
  optimizer.lr=1e-5 optimizer.min_lr=1e-5 scheduler.lr_decay_style=constant \
  scheduler.lr_warmup_iters=5 scheduler.lr_decay_iters=6 \
  model.recompute_num_layers=1 model.ernie_reference_rmsnorm=true \
  model.ernie_fused_mhc=false model.ernie_vha_batched_mix=false \
  model.ernie_dsa_preserve_cp_key_slots=true model.ernie_dsa_contiguous_backward=true \
  '+model.moe_megakernel_backend_config.reference_pre_fc2_gate=true' \
  "logger.wandb_project=$WANDB_PROJECT" logger.wandb_exp_name=ernie55-accuracy-skew-d-backward
```

On one node compare the 64 saved shards with the original Swift twelve-document capture.

```bash
python "$EX/head_parity_256k.py" "$WORK/swift-heads/skew" "$LOGS/accuracy-skew-d-backward-heads" --check
```

For a single 256K document, a controlled replay of the final backbone D layer held the original input and upstream gradient fixed. With contiguous query owners, switching the BF16 key gradient reduction from zigzag to contiguous owners reduced the full input-gradient relative L2 difference from 0.315478% to 0.006776%. Repeating original Swift with the same input and upstream gradient gave 0.007088%. This local test covers all 1,073,741,824 input coordinates of that layer.

Two complete backward runs on one rack with NCCL Ring used the same contiguous query owners. The additional contiguous **key** reduction changed the full block-input relative L2 differences against original Swift as follows. Each value covers all 262,144 physical positions and four residual streams on one dense replica.

| D backward configuration | Block 1 | Block 22 | Block 43 |
|---|---:|---:|---:|
| Contiguous query owners | 2.78964% | 1.07683% | 0.34410% |
| Contiguous query and key owners | 1.67824% | 1.10718% | 0.31989% |

The two arms used independent native starts and the original main and MTP losses. Early and middle block gradients continue to cross the fixed 0.5% threshold. [Query-owner run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/f10s69mw) · [Query-and-key-owner run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/gyyha0mx)

The published option was also exercised in the normal trainer on 64 GB300s with twelve unequal documents. All ranks completed one GA8 update. Each rank executed all seven D backward modules eight times. Both heads passed the four-replica Swift check on the first microbatch. All 173,432 main and 166,260 MTP raw logits sampled at valid positions agreed exactly. The head-input gradient relative L2 was 0.0012833% for main and 0.00066723% for MTP.
