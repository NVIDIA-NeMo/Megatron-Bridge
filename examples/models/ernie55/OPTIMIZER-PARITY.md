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
  local iters=2
  local -a recorder=("$EX/optimizer_parity_256k.py")
  if [[ ${ERNIE_CAPTURE_NEXT_HEADS:-0} == 1 ]]; then
    recorder+=(--next-heads)
    iters=3
  fi
  torchrun --nnodes=16 --nproc-per-node=4 --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "${recorder[@]}" bridge "$capture" -- \
    "dataset.path=$DATA/uniform.jsonl" "checkpoint.pretrained_checkpoint=$CKPT" \
    "train.train_iters=$iters" train.micro_batch_size=1 train.global_batch_size=32 \
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
capture_optimizer_swift() {
  local capture=$1 cache=$2 log=$3 iters=2
  local -a recorder=("$EX/optimizer_parity_256k.py")
  if [[ ${ERNIE_CAPTURE_NEXT_HEADS:-0} == 1 ]]; then
    recorder+=(--next-heads)
    iters=3
  fi
  torchrun --nnodes=16 --nproc-per-node=4 --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    "${recorder[@]}" swift "$capture" -- \
    --model "$SWIFT_MODEL" --model_type ernie5_5 --template ernie5_5 \
    --cached_dataset "$cache" --max_length 262144 --truncation_strategy split \
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
    --train_iters "$iters" --lr_decay_iters 6 --lr 1e-5 --min_lr 1e-5 --lr_warmup_iters 5 \
    --lr_decay_style constant --weight_decay 0.1 --clip_grad 1.0 --seed 42 \
    --logging_steps 1 --save_steps 1000000 --no_save_optim true --no_save_rng true \
    --output_dir "$log" --check_model false
}
capture_optimizer_swift "$OPT/swift-capture" "$OPT/swift-cache-64" "$OPT/swift-log"
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

### Full final shared-expert FC2 on the eight-document update

A separate experiment captured all 8,388,608 FP32 coordinates of the shared FC2 matrix used by the last trunk block and MTP, before and after the native second GA8 update at LR 2e-6. The trainers consumed the original inputs. All three Bridge arms applied the same measured original-Swift router bias at the first update boundary to isolate the FC2 comparison.

| Last shared FC2 backward | Reduced preclip gradient relative L2 | Actual FP32 update relative L2 | Actual update relative peak |
|---|---:|---:|---:|
| Fused, zigzag owners | 0.08750% | 3.39679% | 38.1548% |
| BF16 return, zigzag owners | 0.11878% | 3.96637% | 39.5100% |
| BF16 return, original contiguous CP owners | 0.04199% | 2.10345% | 37.6205% |

All three full-matrix updates exceed both preset tolerances against independent Swift. With identical full gradients, optimizer inputs and clipping norms, the two container implementations of Muon produced elementwise-identical FP32 updates. Changing ownership reduced the observed FC2 gradient discrepancy under BF16 returns. Substantial update differences remain.

An independent repeat of original Swift on the same inputs found 0.00088% relative L2 in the full step-2 gradient and 0.42574% relative L2 in the actual FP32 update. The relative update peak was 18.4519%, with 678 of 8,388,608 coordinates exceeding 1% of the original update peak. Read the peak metric alongside relative L2 and the number of affected weights. Both preset thresholds still apply.

A separate diagnostic tested the natural G2 configuration with the reference head weight gradient described below. Bridge computed its own router bias. Across 64 ranks, every bias bit matched original Swift before step 1, before step 2, and at the final microbatch. Only the final shared-expert FC2 in the last trunk block and MTP used a diagnostic BF16-return backward with the original contiguous CP owners.

| Step 2 versus the same original Swift update | FC2 coordinates | Reduced gradient relative L2 | Actual FP32 update relative L2 | Actual update relative peak |
|---|---:|---:|---:|---:|
| Public G2 with reference head weight gradient, default FC2 | 8,217 sampled | 0.09039% | 2.93746% | 17.0765% |
| Same sampled coordinates, diagnostic FC2 ownership and BF16 return | 8,217 sampled | 0.04208% | 1.62520% | 14.7541% |
| Diagnostic FC2 ownership and BF16 return, complete matrix | 8,388,608 | 0.04130% | 2.08444% | 37.8551% |

On the full matrix, 30,147 update coordinates exceeded 1% of the original peak. The FC2 ownership change above is experimental. The commands below exercise the public head weight-gradient option with the default FC2.

To locate the gradient difference, the diagnostic captured real operands for the first microbatch on one data replica, aligning all 262,144 physical tokens across CP. For both optimizer steps the FC2 input activations matched bitwise across 131 sampled channels at both call sites. The incoming output gradients at MTP also matched across all 133 sampled channels. At the last trunk call on step 2, they differed at 4,170,710 of 34,865,152 values, with 0.02357% relative L2. Substituting only that incoming gradient reproduced all differences in the sampled FP64 weight-gradient contraction, totaling 0.00841% relative L2 after adding the two call sites. The full native optimizer comparison above includes all microbatches, data replicas and the trainer’s reductions.

A subsequent paired run also used reference CE and projection-input gradient rounding for the **main** head. On fixed original inputs, every main raw-logit gradient block and all 1,073,741,824 head-input gradients matched bitwise. In two native updates, sampled LM-weight gradient relative L2 fell from 0.00169726% to 0.00000258%. The full final FC2 update error remained 2.07966% relative L2 and 37.5814% relative peak, compared with 2.08444% and 37.8551% before the main-head change. The incoming trunk FC2 gradient remained 0.02342% relative L2 on the sampled channels.

The packed loss masks isolate a small subset of this gradient by branch. At the first seven rows in each of the eight documents, the main loss supplies zero gradient. All 56 rows had an FC2 gradient discrepancy from the MTP-return path on the measured channels. Together they accounted for 0.000472% of its total squared error. At each document's penultimate row, only the main loss can contribute and all sampled FC2 gradient values matched. Most error occurred on rows where both branches can contribute. Separate branch gradients were unavailable for those rows.

### Shared output-weight rounding

Run the same natural eight-document G2 optimizer comparison with one additional option. Keep the reference capture, weights and training environment used above. On each Bridge node, run:

```bash
git -C "$BRIDGE" fetch origin refs/pull/7000/head
git -C "$BRIDGE" checkout --detach 25be9c87f028af2cef2e2a430aba43cc5401635a
git -C "$MCORE" checkout --detach 3ba272c0e128d6e0b5147142363710fcbcaa2671
export NCCL_ALGO=Ring
capture_optimizer_bridge "$OPT/bridge-head-wgrad-g2" 2 ernie55-uniform-shared-head-wgrad \
  model.moe_router_bias_bf16_counts=true model.ernie_router_bias_contiguous_cp_counts=true \
  model.ernie_dsa_contiguous_backward=true model.ernie_mtp_strict_head_backward=true \
  model.ernie_head_reference_wgrad=true
```

After all ranks have exited, score the two captures on one CPU host:

```bash
python "$EX/compare_optimizer_256k.py" "$OPT/swift-capture" "$OPT/bridge-head-wgrad-g2" \
  --json "$OPT/optimizer-head-wgrad-g2.json"
```

A controlled two-update experiment first supplied the same observed original router bias to both trainers after the zero-LR update. At 807,229 output-weight coordinates, the second pre-clipping gradient error was 0.18961% relative L2 with the fused FP32 accumulation, 0.25945% with BF16 weight GEMMs on zigzag token owners, and 0.00439% with BF16 weight GEMMs on the original contiguous token owners. The largest error in the actual master-weight update fell from 7.26% of the reference peak to 0.375%. No sampled coordinate exceeded the 1% update peak gate with contiguous owners.

On identical original-training hidden states and raw-logit gradients, replaying fused FP32 accumulation reproduced 0.18965% gradient error. The original BF16 per-head GEMMs followed by BF16 addition reproduced the gradient within 0.0000032% after global reduction. This accounts for the two distinct roundings at the shared weight.

The ordinary public training recipe then ran two complete optimizer updates with all G2 strict options and the new head option. All 64 ranks used their own router counts and model gradients. The recorded inputs and pre-update weights matched the independent Swift run. The second optimizer learning rate was 2e−6.

| Step-2 parameter | Preclip gradient L2 | Gradient peak | Actual update L2 | Actual update peak |
|---|---:|---:|---:|---:|
| LM output weight | 0.00170% | 0.00276% | 0.03947% | 0.37453% |
| Token embedding | 0.58350% | 0.28696% | 0.61977% | 35.24053% |
| BOS embedding row | 0.58350% | 0.28696% | 1.05725% | 35.24053% |
| Non-BOS embedding | 0.41202% | 0.41182% | 0.40583% | 10.31305% |
| Final shared-expert FC2 | 0.09039% | 0.09486% | 2.94596% | 16.80328% |

All 807,229 sampled LM updates met the 1%-of-reference-peak limit. The LM update also met the 0.5% aggregate L2 limit. The token embedding and final shared-expert weight updates still exceed at least one limit. The independent full-model gradient norms were 80.470665 (Swift) and 80.551552 (Bridge). [Original Swift training](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/5txaup4e) · [Bridge with reference head weight gradients](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/lm8zjwea)

Two clean training processes ran successively on the same 64 GB300 GPUs using the eight-document fixture and the G2 settings above. Each ran eight GA8 optimizer updates. Throughput divides the physical tokens in updates 4–8 by their total elapsed time and the number of GPUs. Each memory column reports the maximum PyTorch high-water mark across 64 ranks over the entire process.

| Shared output-weight gradient | Seconds/update | Tokens/s/GPU | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| [Fused FP32 accumulation](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/4zq294lg) | 58.303 | 2,248.13 | 176.874 | 181.266 |
| [Contiguous-owner BF16](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/67lslrwq) | 58.444 | 2,242.68 | 178.197 | 180.506 |

Enabling the option changed throughput by -0.24% and the full-model allocated peak by +1.322 GiB on this pair.

### Input embedding and the BOS row

This comparison isolates the input embedding lookup on the combined G2 recipe above. In the eight-document fixture, the original Swift trainer places one of the eight BOS tokens on each even-numbered CP rank. Per-document zigzag places all eight on CP rank zero. Both use BF16 embedding weights and an FP32 main gradient. The production Bridge lookup uses `F.embedding`. Its MTP-shared input weight has `zero_out_wgrad=True` and bypasses fused embedding-gradient accumulation. Original Swift indexes the weight by token ID with PyTorch deterministic algorithms enabled. Neither trainer's shifted MTP embedding input contains a BOS token on this fixture.

**Native training measurements.** Each row compares the first nonzero AdamW update after a zero-LR warmup. Both optimizers retain their own state. All 32 GA/DP input and target vectors on both steps and all sampled starting master values matched. For the three diagnostic Bridge runs, the actual input IDs to both main and MTP lookups also matched the original at every physical position. There are 811,321 sampled embedding coordinates, including every channel of the 4,096-element BOS row. Each cell gives relative L2 / peak-relative error.

| Compared pair | Preclip embedding gradient | True embedding update | True BOS update |
| --- | ---: | ---: | ---: |
| Two original Swift starts | 0.3361% / 0.1923% | 0.4502% / 32.4442% | 0.8028% / 32.4442% |
| Production combined G2 vs fresh Swift | 0.3096% / 0.2059% | 0.5009% / 25.0264% | 0.7476% / 25.0264% |
| Advanced indexing on zigzag owners vs fresh Swift | 0.3083% / 0.1156% | 0.5163% / 22.6154% | 0.7671% / 22.6154% |
| Advanced indexing on contiguous owners vs fresh Swift | 0.2756% / 0.1571% | 0.5421% / 25.6948% | 0.8859% / 25.6948% |
| Deterministic advanced indexing on contiguous owners vs fresh Swift | 0.3272% / 0.1460% | 0.5088% / 20.3386% | 0.7723% / 20.3386% |

The fixed gates are 0.5% relative L2 and 1% peak-relative. Every Bridge embedding update exceeded the peak limit. Across the three interventions, the incoming BOS gradients on step 2 already differed by 2.06%–2.21% relative L2 before the lookup backward combined them by token ID. This covers 1,048,576 BF16 components across all eight microbatches and four dense replicas. The independent original-Swift repeat also exceeded the peak gate. [Previous Swift run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/5txaup4e) · [Fresh Swift run](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/ii7w48np) · [Production G2](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/cbw736ek) · [Zigzag index](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/dembu758) · [Contiguous index](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/iuqjfqav) · [Contiguous deterministic index](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/n4fprtlo)

**Fixed-input lookup replay.** To isolate the local backward, record the actual input IDs and the BF16 incoming gradients at every BOS occurrence on every rank of both Swift updates. Keep the full original token-ID vectors and original hidden dimension when replaying the CUDA lookup backward. Other vocabulary rows have independent weight gradients, so their incoming gradients can be zeroed for this BOS-only replay. Reconstruct each layout in physical row order, accumulate eight native microbatch gradients, average over the 64-rank dense DP-with-CP group, and use the original step's clip factor and AdamW state progression. The reconstructed original step-2 BOS gradient matched the measured preclip gradient to 0.0000013% L2 and 0.0000037% peak-relative.

On these identical original operands, the production zigzag `F.embedding` gives **0.02906% L2 / 0.02529% peak** on the step-2 BOS gradient and **0.00610% / 0.08113%** on the resulting two-step AdamW update. Using advanced indexing on zigzag owners raises the replayed update peak to 1.2680%. On the original contiguous owners, both advanced-index algorithms reproduce the same BOS gradients and replayed update. The production lookup therefore passes the fixed-input gates on this BOS row. The independent training measurements above continue to include the variation already present in the incoming gradients.

## Capture both heads after the first nonzero weight update

The optional `--next-heads` flag records the two optimizer updates above, then captures the first training microbatch of iteration 3. With this warmup, update 1 has learning rate zero and update 2 uses 2e-6. Each trainer uses its own weights, routing bias and optimizer state throughout. Use the same eight-document input on both sides.

1. Use the software and checkpoint from **Prepare the inputs and public scripts**. Fetch the recorder and matching MCore from the two PRs. Perform this checkout once in your shared worktrees, before launching either job. Keep the original Swift MCore on the Python path of the Swift job.

   ```bash
   git -C "$BRIDGE" fetch origin refs/pull/7000/head
   git -C "$BRIDGE" checkout --detach 2d88bee4a96060994fb4aaaf0d43857d1de8635a
   git -C "$MCORE" fetch origin refs/pull/9506/head
   git -C "$MCORE" checkout --detach 3ba272c0e128d6e0b5147142363710fcbcaa2671
   ```

2. Prepare 96 rows in the original Swift environment. Use the existing 32-row cache from **Run original Swift** and a fresh destination.

   ```bash
   python - "$OPT/swift-cache-32" "$OPT/swift-cache-96" <<'PYDATA'
   import sys
   from pathlib import Path
   from datasets import concatenate_datasets, load_from_disk

   rows = load_from_disk(sys.argv[1])
   assert len(rows) == 32 and not Path(sys.argv[2]).exists()
   concatenate_datasets([rows, rows, rows]).save_to_disk(sys.argv[2])
   PYDATA
   ```

3. In the 16-node Bridge allocation, use its environment and `capture_optimizer_bridge` definition from **Run Bridge**. On each node, run

   ```bash
   export NCCL_ALGO=Ring
   ERNIE_CAPTURE_NEXT_HEADS=1 capture_optimizer_bridge "$OPT/bridge-post" 2 ernie55-post-update-heads \
     model.moe_router_bias_bf16_counts=true model.ernie_router_bias_contiguous_cp_counts=true \
     model.ernie_dsa_contiguous_backward=true model.ernie_mtp_strict_head_backward=true \
     model.ernie_head_reference_wgrad=true
   ```

4. In the original Swift allocation, use its environment and `capture_optimizer_swift` definition from **Run original Swift**. On each node, run

   ```bash
   ERNIE_CAPTURE_NEXT_HEADS=1 capture_optimizer_swift "$OPT/swift-post" "$OPT/swift-cache-96" "$OPT/swift-post-log"
   ```

   Both commands enable `--next-heads` and set three training iterations. Swift exits after the forward and backward of microbatch 17. Bridge finishes its third iteration and writes the measurements from microbatch 17. The optimizer recorder covers the first two updates on each side.

5. On one CPU host, run both checks independently. Each check returns zero on a pass and writes its report before returning one on a failed numerical gate. Treat a missing or incomplete report as a capture or scoring error.

   ```bash
   optimizer_status=0
   python "$EX/compare_optimizer_256k.py" "$OPT/swift-post" "$OPT/bridge-post" \
       --json "$OPT/post-optimizer.json" --check || optimizer_status=$?
   heads_status=0
   python "$EX/head_parity_256k.py" "$OPT/swift-post/heads" "$OPT/bridge-post/heads" \
       --check > "$OPT/post-heads.json" || heads_status=$?
   printf 'Optimizer exit %s, heads exit %s\n' "$optimizer_status" "$heads_status"
   ```

Raw logits and head-input gradients are joined by their actual global token positions, including when the output projection temporarily changes CP owners. The comparison includes all valid cross-entropies on four dense replicas, sampled raw logits and head-input gradients. A head prediction and an applied parameter update have separate error measures.

### Verified public capture on eight equal documents

Both jobs used 64 GB300 GPUs, CP16, EP64, GBS32 and GA8. Original Swift loaded the recorder at `875ed8df3815bbf004238eec36910f2c8bf6414e`. Bridge used `2d88bee4a96060994fb4aaaf0d43857d1de8635a`, which attaches the delayed probe to the already constructed Bridge output layers. The Swift recorder and training path are identical in those two revisions. The comparator verified identical token IDs and main targets for each of the 32 rows on both optimizer steps, matching sampled pre-step master values and identical tokens, main targets and MTP targets on the following microbatch. Both trainers retained their own optimizer state, router biases and weights.

| Head on microbatch 17 | Mean CE, Swift / Bridge | Relative mean CE difference | Valid CE L2 / peak | Raw logits L2 / peak | Head-input gradient L2 / peak | Head gate |
|---|---:|---:|---:|---:|---:|---|
| Main | 1.28177782 / 1.28177921 | 0.00011% | 1.8506% / 13.3437% | 2.0472% / 3.7500% | 3.0471% / 7.3052% | Fail |
| MTP | 1.45026067 / 1.45031427 | 0.00370% | 2.3908% / 15.7615% | 6.6506% / 26.2868% | 4.8855% / 10.5159% | Fail |

The CE comparison covers 1,048,320 valid main targets and 1,048,320 valid MTP targets over four dense replicas. The sampled tensors cover 1,056 valid main positions and 1,024 valid MTP positions, with 163 raw-logit columns and 4,096 incoming-gradient channels at each position. The head limits are 0.001% relative mean CE, 0.5% relative L2 and 1% relative peak.

| Step-2 parameter | Sampled FP32 coordinates | Preclip gradient L2 / peak | True FP32 update L2 / peak | Update gate |
|---|---:|---:|---:|---|
| LM output weight | 807,229 | 0.0017% / 0.0028% | 0.0672% / 0.3745% | Pass |
| Token embedding | 811,321 | 0.2952% / 0.1049% | 0.4568% / 14.7111% | Fail |
| BOS embedding row | 4,096 | 0.2952% / 0.1049% | 0.5901% / 14.7111% | Fail |
| Non-BOS embedding | 807,225 | 0.4294% / 0.4760% | 0.4086% / 14.6258% | Fail |
| Final shared-expert FC2 | 8,217 | 0.0904% / 0.0947% | 2.9238% / 17.0765% | Fail |

The native step-2 gradient norms were 80.515190 (Swift) and 80.408943 (Bridge). The optimizer threshold is 0.5% relative L2 and 1% relative peak. These results cover one repeated eight-document row and the first nonzero update. [Original Swift](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/mfoe2aft) · [Bridge](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/83k9gzii)

### Repeated training starts and the largest sampled MTP difference

This table uses one original Swift run (A) as the common denominator. Each run consumed identical physical input and target rows on the two complete GA8 updates and the following microbatch. The two G2 runs with the reference output-weight gradient have the same model, optimizer, topology and precision settings.

| Compared with original Swift A | Main raw logits L2 | MTP raw logits L2 | MTP raw logits L2 at position 15,359 | True MTP target probability there |
|---|---:|---:|---:|---:|
| Second original Swift start | 1.776% | 2.190% | 11.94% | 99.894% |
| Fourth original Swift start | 1.565% | 6.125% | 17.04% | 99.884% |
| Public G2 | 1.563% | 3.416% | 10.18% | 99.904% |
| G2 with reference output-weight gradient, first start | 1.953% | 6.257% | 129.51% | 99.759% |
| G2 with reference output-weight gradient, second start | 1.963% | 2.386% | 20.44% | 99.869% |

The first G2 start with the reference weight gradient produced 6.257%–6.781% MTP raw-logit L2 against the first three original starts. The second G2 start produced 2.289%–2.386% against those same references. A fourth original Swift start gave 6.101%–6.146% against the first three originals. The two identically configured G2 starts differ by 7.151% with the first G2 start as denominator. The observed ranges support substantial run-to-run variation in both trainers.

At global position 15,359 in the first document, the original Swift A assigned the true MTP target probability 99.926%. This one position accounted for 87.73% of the first candidate’s sampled raw-logit squared error against A. Its 163 measured vocabulary columns shifted together, from a mean of 3.794 to −2.457. The common shift accounted for 96.71% of the local squared error. A constant cancels from softmax when applied to every vocabulary entry. The table uses the actual full-vocabulary cross-entropy to calculate the target probability. Raw-logit gates use uncentered values.

On the fourth original start, all 64 ranks completed and recorded both native optimizer updates, all GA8 input shards, and the next forward and backward for both heads. The scheduler then reached its 15-minute wall limit during shutdown. Its recorded allocation status is TIMEOUT. The CPU comparator verified all completion records, input IDs and labels before scoring.

[Original A](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/tktbc9jm) · [Original B](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/mmm0sqjj) · [Original C](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/mfoe2aft) · [Original D](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/4latrcz4) · [G2](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/e6vx6x3l) · [Reference weight gradient, first start](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/83k9gzii) · [Reference weight gradient, second start](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/q6nsinez)

### Replaying the observed output projection on the updated weights

A separate matched capture used the fourth original Swift run and an additional G2 run with the reference output-weight gradient. On microbatch 17, both recorded the actual outer head inputs at each sampled global position in all four replicas and the BF16 projection weights for vocabulary IDs `0, 1237, …`. All 163 vocabulary rows × 4,096 hidden channels (667,648 weights) matched **bitwise** across the two trainers and both heads. Each rank's weight hash matched the copy used for replay.

For each head, join the BF16 hidden rows by the outer layer’s physical token position. Reorder the recorded output logit rows by their own physical positions after the G2 projection changes CP owners. Convert the two operands to FP64, compute `hidden @ weight.T` on a CPU and round to BF16. The last two columns give the replay’s relative L2 error against the native GPU logits in the respective trainer.

| Head | Hidden input difference L2 | Native raw-logit difference L2 | Weight contribution | Swift replay error | G2 replay error |
|---|---:|---:|---:|---:|---:|
| Main | 2.641% | 1.417% | 0% | 0.0081% | 0.0081% |
| MTP | 7.329% | 6.260% | 0% | 0.0131% | 0.0102% |

The largest sampled MTP error in this pair moved to position **84,991**, at offset 19,455 in the third document. It accounts for **90.95%** of the sampled squared raw-logit error. At that position, the input to the MTP projection differed by 273.51% relative L2. Multiplying each trainer's hidden row by the identical sampled weights and rounding to BF16 reproduced all 163 native logits exactly on both sides. The mean over those logits moved from −3.4304 to 2.5365. The actual full-vocabulary target probabilities were 99.8553% and 99.3517%, respectively.

At position 15,359, the two trainers instead differed by 3.56% raw-logit L2 in this pair, accounting for 0.08% of the total sampled squared error. They selected the same ten MTP experts there. The MTP incoming token embedding was identical. The input to the hidden branch norm already differed by 2.27%, and the input and output at the final MTP norm differed by 12.26% and 4.55%, respectively. These observations locate the measured logit discrepancy in the hidden activations arriving at the shared output projection. Intermediate norm and router data in this capture cover position 15,359 and three preselected positions in the sixth document. [Original Swift with operand capture](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/4latrcz4) · [G2 with operand capture](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/unz0sfjv)

### Repeatability of the original D sparse attention backward

On four GB300 GPUs with cuDNN frontend 1.29.1, we replayed original block 43 with the same 291 captured query rows, key indices, natural incoming gradient, and full 262,144-row KV table. The original sparse backward omitted the deterministic keyword. Even with PyTorch deterministic algorithms enabled, its BF16 dKV differed in all 16 same-device repeats, at 155–211 of 150,994,944 coordinates per repeat. Passing `deterministic=True` and the scratch buffer sized by `DSA.SparseAttentionBackward.scratch_workspace_bytes()` made all 16 dKV repeats bitwise identical, with the same hash across devices. Both modes matched the captured training dQ at all 10,727,424 coordinates. The contribution of this source of variation to the next-update logits has not been isolated. [Original Swift block capture](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/4hzmo0ea)

| Shape at CP16, 256K physical tokens | Query rows | KV rows × width | Physical key slots | Deterministic workspace (GiB/GPU) |
|---|---:|---:|---:|---:|
| D backward, captured selection | 291 | 262,144 × 576 | 2,176 | 72.00014 |
| D backward, full rank | 16,384 | 262,144 × 576 | 2,176 | 72.00781 |
| H backward, original contiguous layout, eight equal documents | 16,384 | 18,576 × 512 | 384 | 4.54297 |

The replay allocated the first workspace. For the full-rank rows we passed real device tensor descriptors to the installed cuDNN plan and queried its supported workspace size without allocating that scratch. All queries use 64 query heads and a 512-wide output. Original H calls the same sparse backward with per-query key lengths. Its table geometry comes from the original contiguous CP implementation. Enabling deterministic D throughout training requires checking available memory at each D backward.
