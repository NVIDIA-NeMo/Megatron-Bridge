# Reproduce ERNIE 5.5 Flash 256K BF16 training on 64 GB300s

This experiment trains the public ERNIE-5.5-Flash-Base checkpoint with both its main LM head and MTP head. It compares natural MoK routing with the same routing under MoonEP expert replication. Use one 64-GPU NVLink domain (16 nodes × 4 GB300s). TP1, PP1 and CP16 give four dense data replicas. EP64 spans all ranks. Each replica sees one packed 262,144-position row per microbatch and accumulates eight microbatches. An optimizer update therefore contains 32 rows. The recipe runs eagerly with full uniform recomputation, two hybrid symbols per group, BF16 weights and expert math, FP32 router scores, and a CUDA token-chunked head (4,096 rows) on both outputs.

## 1. Check out the source and prepare the training environment

Use a Linux GPU container supporting SM103/SM103a on **every** node. The measured base was NGC PyTorch 26.04 (Python 3.12, CUDA 13.2, PyTorch 2.12.0a0, Triton 3.6). The training stack used Transformer Engine 2.20.2, FlashMLA with sparse sink attention (commit `b7643bd`), DeepEP hybrid-ep (`10d4dd7`), cuDNN frontend 1.29.1 with CUTLASS DSL 4.8.1, fast-hadamard-transform 1.1.0, Emerging Optimizers 0.3.0 and Transformers 5.20.2. The base NGC image alone needs these additions. Install the dependencies in the same Python environment used by torchrun. [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) · [FlashMLA](https://github.com/deepseek-ai/FlashMLA) · [DeepEP](https://github.com/deepseek-ai/DeepEP) · [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

Run the following from a shared workspace visible on all nodes. `BRIDGE` is the checkout of [Bridge #7000](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/7000) containing this README and its helpers. The training implementation was introduced in `43acf2daf25ab96870fb530dad546f9dbb201ad4`. The companion [MCore #9506](https://github.com/NVIDIA/Megatron-LM/pull/9506) supplies the MoonEP backend, packed H support and stable D-indexer selection.

```bash
export WORK=$PWD
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
git -C Megatron-Bridge fetch origin refs/pull/7000/head
git -C Megatron-Bridge checkout --detach FETCH_HEAD
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
# Once on one four-GB300 node. CKPT must be a new path.
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$EX/benchmark_256k.py" convert "$HF_MODEL" "$CKPT"
```

The architecture helper keeps the official model dimensions and MTP topology, selects the 262,144-position benchmark limit, makes the release's YaRN factor (16) and original context (65,536) explicit, selects adjacent-pair D RoPE, uses the registered ERNIE alias and removes the HF quantization metadata from the training config. The pretraining recipe sets the D indexer loss coefficient to 0.01. The official checkpoint is read without modification. [Model release](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/tree/028544b2ffceea189b2d8768ed2b9d09b06f2343)

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

With stable index selection, earlier integrated runs of MoK versus MoK + MoonEP matched all captured CE, sampled raw logits and sampled head-input gradients on all three layouts before the first update. The native selector permits different choices at tied score cutoffs. On the uniform fixture a full GA8 gradient check of the optimized versus baseline integrated MoK + MoonEP configuration gave 0.2982% relative L2 and 0.2428% peak-relative over 279.6B gradient elements. Some individual shards exceeded those aggregate thresholds.

**Whole-model parity with the independent Baidu Swift implementation remains open on all three layouts.** In a deterministic-reference comparison using native pre-FC2 weighting and stable key-ID ties, the main/MTP raw-logit L2 values were 2.306%/14.398% (single), 2.138%/3.265% (uniform) and 1.915%/2.541% (skew). A separate same-input first-routed-FFN diagnostic attained 0.05113% output L2 against deterministic Swift using a modified pre-FC2 weighting kernel. Those independent-reference diagnostics use different expert arithmetic from the default public MoK production command above. A backend-to-backend pass is compatible with a failure against Swift. To test Swift, load the same released weights independently, fix the identical packed token/target rows and record raw LM/MTP values before updating; compare them at the same physical positions and valid targets.


For an independent reference that uses contiguous CP, export the same NPZ schema as the probe. A file contains its dense replica ID (`dp`), global padded document boundaries (`bounds`), local rows' physical global coordinates (`pos`), `tokens`, the actual `main_labels` and rolled `mtp_labels` with invalid targets set to −100, and both `main_ce` and `mtp_ce`. Store the global positions in `sample` and the corresponding `main_logits`, `mtp_logits`, `main_dh` and `mtp_dh` in that order. Raw logits take vocabulary IDs 0, 1237, 2474, …; the hidden-gradient samples contain every channel. Use identical loss scaling. The comparator reconstructs each full replica independently of CP sharding and joins sampled values on global position.
