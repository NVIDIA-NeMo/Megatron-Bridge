# Compare native optimizer updates on eight equal 256K documents

The two trainers process the same eight 32,768-position documents, with four dense replicas × eight microbatches per optimizer step. Both reload the Base checkpoint, run the zero-LR warmup step, retain their own Adam/Muon state, and take a second step at LR 2e-6. The recorder saves the actual native reduced gradient before clipping and the FP32 master values before and after the optimizer. These jobs are instrumented and are separate from throughput measurements.

### Prepare the inputs and public scripts

Follow steps 1–3 and section 6c steps 1–2 of the [256K benchmark](README-256k.md) to build the environment and reference-arithmetic MoK, obtain the weights and prepare `uniform.jsonl`. Use a fresh output directory per run. The source revision below includes the recorder and NumPy comparator. Fetch the PR before switching the shared accuracy worktree.

```bash
git -C "$BRIDGE" fetch origin refs/pull/7000/head
git -C "$BRIDGE" checkout --detach 54a9641a46af01040bd9a80c797b24d53f88b555
export EX=$BRIDGE/examples/models/ernie55
export OPT=$WORK/ernie55-optimizer-check
mkdir -p "$OPT"
```

### Run Bridge

Use a 16-node × four-GB300 allocation. On each node set `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` for torchrun and export the accuracy environment from section 6c. Section 6c supplies the compiled CUDA head via `ERNIE55_CUDA_HEAD_LIBRARY`, the architecture directory via `ERNIE55_HF_PATH` and the strict MoK source and extension. The following reproduces the comparison's public strict G1 recipe, including the default integer router-bias counters. For W&B, set `WANDB_PROJECT` to a project you can write to.

```bash
export PYTHONPATH="$MOK:$MCORE:$BRIDGE/src${PYTHONPATH:+:$PYTHONPATH}"
export MCORE_DSA_STABLE_INDEXER_TOPK=1 MCORE_DSA_PRESERVE_SELECTED_SLOT_ORDER=1
export MCORE_DSV4_GROUP_PROJ_BMM=1 NCCL_MAX_P2P_NCHANNELS=8 NCCL_GRAPH_REGISTER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
capture_optimizer_bridge() {
  local capture=$1 layers=$2 experiment=$3
  shift 3
  torchrun --nnodes=16 --nproc-per-node=4 --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "$EX/optimizer_parity_256k.py" bridge "$capture" -- \
    "dataset.path=$DATA/uniform.jsonl" "checkpoint.pretrained_checkpoint=$CKPT" \
    train.train_iters=2 train.micro_batch_size=1 train.global_batch_size=32 \
    optimizer.lr=1e-5 optimizer.min_lr=1e-5 scheduler.lr_warmup_iters=5 \
    scheduler.lr_decay_style=constant scheduler.lr_decay_iters=6 \
    model.thd_max_packed_sequences=8 model.moe_virtual_expert_load_balance=true \
    "model.recompute_num_layers=$layers" model.ernie_reference_rmsnorm=true \
    model.ernie_fused_mhc=false model.ernie_vha_batched_mix=false \
    model.ernie_dsa_preserve_cp_key_slots=true \
    '+model.moe_megakernel_backend_config.reference_pre_fc2_gate=true' \
    "logger.tensorboard_dir=$OPT/bridge-tb" "logger.wandb_project=$WANDB_PROJECT" \
    "logger.wandb_exp_name=$experiment" "$@"
}
capture_optimizer_bridge "$OPT/bridge-capture" 1 ernie55-uniform-public-optimizer
```

### Run original Swift

Use the original `ms-swift-int`, `mcore-bridge-int` and Megatron Core environment from the [Swift setup guide](SWIFT-HEAD-PARITY.md). Use the same pinned inputs and the Swift-loadable equivalent of the Base checkpoint. The preparation helper writes 32 identical packed rows. Two consecutive GBS32 updates need 64 rows. Run this preparation once with the `datasets` package.

```bash
python "$EX/swift_head_parity_256k.py" prepare "$DATA/uniform.jsonl" "$OPT/swift-cache-32"
python - "$OPT/swift-cache-32" "$OPT/swift-cache-64" <<'PYDATA'
import sys
from pathlib import Path
from datasets import concatenate_datasets, load_from_disk
rows = load_from_disk(sys.argv[1])
assert len(rows) == 32 and not Path(sys.argv[2]).exists()
concatenate_datasets([rows, rows]).save_to_disk(sys.argv[2])
PYDATA
```

In the original environment, execute the following on each of the 16 nodes. Keep the original Megatron Core on the import path. The recorder calls the installed Swift trainer and stops after its second native optimizer/scheduler call. It releases unused CUDA allocator cache once, before the first distributed gradient synchronization.

```bash
export PYTHONPATH="$MCORE_BRIDGE_INT/src:$MS_SWIFT_INT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NCCL_ALGO=Ring
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nnodes=16 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$EX/optimizer_parity_256k.py" swift "$OPT/swift-capture" -- \
  --model "$SWIFT_MODEL" --model_type ernie5_5 --template ernie5_5 \
  --cached_dataset "$OPT/swift-cache-64" --max_length 262144 --truncation_strategy split \
  --packing false --padding_free true --split_dataset_ratio 0 \
  --dataset_shuffle false --train_dataloader_shuffle false --tuner_type full \
  --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1 --context_parallel_size 16 \
  --cp_partition_mode contiguous --sequence_packing_scheduler dp_balanced \
  --expert_model_parallel_size 64 --expert_tensor_parallel_size 1 --sequence_parallel false \
  --micro_batch_size 1 --global_batch_size 32 --mtp_num_layers 1 --bf16 true \
  --attention_backend local --apply_rope_fusion false --masked_softmax_fusion false \
  --bias_activation_fusion false --gradient_accumulation_fusion false \
  --moe_grouped_gemm true --csa_dense_mode true --moe_permute_fusion false \
  --moe_token_dispatcher_type alltoall --moe_aux_loss_coeff 0 --moe_z_loss_coeff 0 \
  --megatron_extra_kwargs '{"ernie_router_bias_update_rate":0.0001,"deterministic_mode":true}' \
  --calculate_per_row_loss true --dsa_indexer_loss_coeff 0.01 --dsa_indexer_use_sparse_loss true \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --optimizer dist_muon --muon_momentum 0.95 --muon_use_nesterov true --muon_scale_mode spectral \
  --muon_extra_scale_factor 0.2 --muon_num_ns_steps 5 --muon_coefficient_type quintic \
  --muon_split_qkv false --muon_fp32_matmul_prec medium \
  --train_iters 2 --lr_decay_iters 6 --lr 1e-5 --min_lr 1e-5 --lr_warmup_iters 5 \
  --lr_decay_style constant --weight_decay 0.1 --clip_grad 1.0 --seed 42 \
  --logging_steps 1 --save_steps 1000000 --no_save_optim true --no_save_rng true \
  --output_dir "$OPT/swift-log" --check_model false
```

### Compare after both trainers finish

Run the CPU command once with NumPy. Each capture should contain 64 rank completion records and 64 local batch files for each step. The comparator verifies all physical row positions and the input/target digest for all 32 GA/DP slots of both steps. It also verifies initial FP32 master values, nonzero step-2 updates and native optimizer-state continuity across the warmup. It accepts separate runs from the same trainer to measure within-framework variation.

```bash
python "$EX/compare_optimizer_256k.py" "$OPT/swift-capture" "$OPT/bridge-capture" \
  --json "$OPT/optimizer-result.json" --check
```

The output reports fractions. Multiply by 100 for the percentages in the benchmark README. For each coordinate, the optimizer delta is the difference between the actual FP32 post- and pre-step master values, subtracted in FP64. Relative L2 divides the difference norm by the reference vector norm. Peak-relative error divides the largest absolute component difference by the largest absolute reference component. The command writes the numerical report before returning exit code 1 when a preclip-gradient or optimizer-update metric exceeds 0.5% L2 or 1% peak-relative.

| Parameter | Sampled coordinates |
| --- | ---: |
| LM output weight | 807,229 |
| Token embedding, including the complete 4,096-channel BOS row | 811,321 |
| Last shared-expert FC2 | 8,217 |

The sample is every 1,021st row-major coordinate of each parameter, plus all 4,096 features of embedding row 1. The comparator also reports the BOS and non-BOS embedding subsets. Each rank records only the optimizer shards it owns. Replicated ranges are represented once per tensor when scoring.

### Measured two-update reproduction

The original Swift trainer and the public strict G1 Bridge recipe each finished both GA8 optimizer steps on 64 GB300s. One fixed row of eight documents was repeated across the four dense replicas and eight microbatches on each step. The comparator verified all 32 GA/DP inputs on each step and identical starting samples for all three FP32 master weights. Both sides retained their own optimizer state through the zero-LR warmup. This comparison uses the default integer accounting for Bridge expert-bias updates. The run links identify this pair. Swift's W&B entry records completion. [Swift run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/5txaup4e) · [Bridge run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/riwhky4w)

| Step-2 parameter | Preclip gradient relative L2 | Preclip gradient peak-relative | True master-update relative L2 | True master-update peak-relative |
| --- | ---: | ---: | ---: | ---: |
| LM output weight | 4.1035% | 2.5220% | 6.2543% | 69.5693% |
| Token embedding | 50.9742% | 43.0957% | 38.8382% | 170.9043% |
| Embedding BOS row | 50.9742% | 43.0957% | 35.9677% | 170.9043% |
| Sampled non-BOS embedding | 27.2516% | 23.3006% | 39.6564% | 165.0557% |
| Last shared-expert FC2 | 1.5226% | 1.4401% | 6.4816% | 27.2769% |

All three parameter updates exceeded at least one fixed gate. Comparing the Bridge capture with itself returned zero for every gradient and update error. A separate CPU reduction over the two native traces reproduced every number in the table. The comparison covers these sampled coordinates, this repeated packing and this first nonzero update.

### Combined G2 optimizer comparison

This second Bridge run groups two recomputed symbols and enables the original contiguous CP owners for BF16 router-bias accounting, contiguous query/key owners for D attention backward, and the strict MTP head backward. It retains the same eight-document row, weights, optimizer and schedule. Reuse the Swift capture from above. On each Bridge node, keep the training environment and shell function from **Run Bridge**, then select the following source revisions and output path.

```bash
git -C "$BRIDGE" fetch origin refs/pull/7000/head
git -C "$BRIDGE" checkout --detach c5ac3c22b772db7c433fe42614d60295b0a811b8
git -C "$MCORE" fetch origin refs/pull/9506/head
git -C "$MCORE" checkout --detach 3ba272c0e128d6e0b5147142363710fcbcaa2671
export NCCL_ALGO=Ring
capture_optimizer_bridge "$OPT/bridge-all-strict-g2" 2 ernie55-uniform-all-strict-g2 \
  model.moe_router_bias_bf16_counts=true model.ernie_router_bias_contiguous_cp_counts=true \
  model.ernie_dsa_contiguous_backward=true model.ernie_mtp_strict_head_backward=true
```

After all 64 ranks finish, score the capture on a CPU host.

```bash
python "$EX/compare_optimizer_256k.py" "$OPT/swift-capture" "$OPT/bridge-all-strict-g2" \
  --json "$OPT/optimizer-all-strict-g2.json" --check
```

| Step-2 parameter | Gradient relative L2 | Gradient peak-relative | True update relative L2 | True update peak-relative |
| --- | ---: | ---: | ---: | ---: |
| LM output weight | 0.1896% | 0.0807% | 0.1745% | 7.2566% |
| Token embedding | 0.4061% | 0.1702% | 0.5155% | 26.7598% |
| Embedding BOS row | 0.4061% | 0.1702% | 0.7890% | 26.7598% |
| Sampled non-BOS embedding | 0.4118% | 0.4320% | 0.3987% | 6.3753% |
| Last shared-expert FC2 | 0.0904% | 0.0950% | 2.9537% | 17.0310% |

The comparator confirmed identical actual input IDs and main labels across both updates and identical sampled starting masters. All three sampled parameter gradients met both gates. Each of the three master-weight updates exceeded at least one gate. The table measures the combined configuration in one instrumented run. Measuring steady-state throughput or the effect of an individual switch requires a separate control. [Swift reference](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/5txaup4e) · [Combined G2 Bridge run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/cbw736ek)
