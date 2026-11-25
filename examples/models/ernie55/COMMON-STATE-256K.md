# Compare original Swift and Bridge at one updated model state

This control uses one original Swift training run to produce the weights for three measurements. The live original processes eight 32K documents per row, with two native GA8 updates. Its first nonzero update uses a learning rate of 2e-6. A fresh Swift process and a fresh Bridge process load the resulting model-only HF directory and each record one forward and loss backward before any further optimizer update. Each run uses 64 GB300 GPUs, TP1, PP1, CP16, EP64 and GBS32.

## 1. Prepare the common input and software

Follow the [256K setup and strict MoK build](README-256k.md#6c-compare-the-256k-training-forward-and-head-gradients-with-swift) and [Swift optimizer setup](OPTIMIZER-PARITY.md#run-original-swift). Prepare the 96-row eight-document Swift cache from the [post-update capture guide](OPTIMIZER-PARITY.md#capture-both-heads-after-the-first-nonzero-weight-update) and the corresponding `$DATA/uniform.jsonl` for Bridge. Each row has the same tokens, labels and document boundaries. Set `OPT` to a shared directory with enough space for one model-only export (approximately 0.57 TB) and a fresh subdirectory for each capture. Set `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` for each 16-node allocation.

The public Bridge recorder used in job 430960 is available at `9b58a4c68c87b480c404ffa1028608bb2d3ce52b`. The companion MCore revision is `3ba272c0e128d6e0b5147142363710fcbcaa2671`. Fetch the PR heads in the shared source checkouts and select these commits before running the jobs. [Bridge entry](https://github.com/jingqiny-99/Megatron-Bridge/blob/9b58a4c68c87b480c404ffa1028608bb2d3ce52b/examples/models/ernie55/shared_state_256k.py) · [MCore](https://github.com/jingqiny-99/Megatron-LM/commit/3ba272c0e128d6e0b5147142363710fcbcaa2671)

## 2. Train, save and capture the live original

Define `capture_optimizer_swift` from the [original Swift command](OPTIMIZER-PARITY.md#run-original-swift). Use the original Swift software, original Base weights and original deterministic launch environment on all nodes. In that environment run the following once per node.

```bash
export CASE=$OPT/eight-doc-common-state
mkdir -p "$CASE"
capture_optimizer_swift "$CASE/live-swift" "$OPT/swift-cache-96" "$CASE/export" export-step2
```

The export mode saves HF weights at update 2 with `--save_safetensors true --save_missing_weights false --no_save_optim true --no_save_rng true`. The trainer continues to the first microbatch of update 3 and records both heads in `$CASE/live-swift/heads`. After all nodes finish, resolve the emitted directory from the index. Use the same `CASE` on all nodes in the two fresh runs.

```bash
export SHARED_HF="$(python - "$CASE/export" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
indexes = list(root.rglob("checkpoint-2/model.safetensors.index.json"))
if len(indexes) != 1:
    raise SystemExit(f"Expected one completed step-2 HF index under {root}, found {len(indexes)}")
print(indexes[0].parent)
PY
)"
test -s "$SHARED_HF/model.safetensors.index.json" || exit 1
```

In the measured export, the 113 HF shards totaled 561,093,673,792 bytes. The 66,119 keys shared with the Base HF model kept their shapes. Of these, 65,258 stayed BF16, 595 stayed FP32 and 266 were saved as FP32 after loading from BF16 at initialization. Those 266 include the two expert router matrices, latent projections, VHA mixing matrices and some indexer normalization parameters. Preserve the dtypes in the native export when testing a post-update state. The Base files also contained 1,573 duplicate terminal-layer tensors for MTP that matched their canonical values byte for byte. The native updated export used only the canonical entries.

All 861 live Swift FP32 tensors were preserved in the export. In the 64-GPU Bridge job, the check immediately before the first microbatch compared all 917,314,140 distinct coordinates with the FP32 values in the safetensors files on each rank. Every bit matched, including the 878,838,759 coordinates in the 266 promoted tensors whose values could no longer be represented exactly in BF16. All 43 router-bias vectors were checked separately, including MTP's own bias. The MTP body shared its 53 distinct parameters with the last decoder block.

## 3. Reload the original from the exported weights

Use the original Swift allocation and `capture_optimizer_swift` definition again. This mode uses the public single-microbatch head recorder and preserves the GA8 objective scaling.

```bash
SWIFT_MODEL="$SHARED_HF" capture_optimizer_swift "$CASE/reload-swift/heads" \
  "$OPT/swift-cache-96" "$CASE/reload-swift-log" reload-heads
```

## 4. Load the same weights into Bridge

Use the Bridge strict MoK environment and the `strict_args` array from the [eight-document precision recipe](README-256k.md#eight-document-throughput-and-precision). Keep its stable D selector and physical key slots. On every Bridge node run the public entry below. It reads the HF weights directly and writes the head capture under `$CASE/reload-bridge/heads`.

```bash
export NCCL_ALGO=Ring
torchrun --nnodes=16 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$EX/shared_state_256k.py" --hf "$SHARED_HF" --output "$CASE/reload-bridge" -- \
  "${strict_args[@]}" "dataset.path=$DATA/uniform.jsonl" \
  train.train_iters=1 scheduler.lr_decay_iters=6 train.micro_batch_size=1 train.global_batch_size=32 \
  model.thd_max_packed_sequences=8 model.recompute_num_layers=2 model.moe_virtual_expert_load_balance=true \
  model.cp_partition_mode=zigzag model.dsv4_cp_zigzag=true \
  model.moe_router_bias_bf16_counts=true model.ernie_router_bias_contiguous_cp_counts=true \
  model.ernie_dsa_contiguous_backward=true model.ernie_mtp_strict_head_backward=true \
  model.ernie_head_reference_wgrad=true checkpoint.pretrained_checkpoint=null checkpoint.load=null checkpoint.save=null \
  "logger.tensorboard_dir=$CASE/bridge-tb" logger.wandb_exp_name=ernie55-eight-doc-common-state
```

## 5. Compare the three measurements

Run the public comparator once per pairing. It joins tokens, target labels, raw-logit samples and head-input gradient samples by physical positions across CP layouts. The fixed thresholds are relative mean CE at most 0.001%, relative tensor L2 at most 0.5% and maximum absolute error divided by the reference peak at most 1%. The full per-token CE uses every valid target. Raw logits and head-input gradients use the sampled positions.

```bash
for pair in live-swift,reload-swift reload-swift,reload-bridge live-swift,reload-bridge; do
  left=${pair%,*}
  right=${pair#*,}
  status=0
  python "$EX/head_parity_256k.py" "$CASE/$left/heads" "$CASE/$right/heads" --check \
    > "$CASE/$left-vs-$right.json" || status=$?
  printf '%s vs %s  exit %s\n' "$left" "$right" "$status"
done
```

The live-to-reloaded Swift comparison checks how faithfully the HF export and reload reproduce the original process. The reloaded Swift-to-Bridge comparison evaluates the two implementations from that common serialized state. The live Swift-to-Bridge comparison includes both effects. Separate experiments measure how model states diverge when each trainer performs its own updates. [Nine independent training starts](benchmarks/post-update-repeats.md)

## Measured common-state results

The updated HF checkpoint index SHA-256 was `d074f56a56b7b5016eb7bc004b477192ddd9ca5e52ad826bc4daac818631c23a`. The model and all three observations used the same 256K packed fixture. The original live measurement was job 430903, the Swift reload was 431710 and the Bridge reload was 430960. [Live Swift](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/24ce40ov) · [Reloaded Swift](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/g248m0m4) · [Reloaded Bridge](https://wandb.ai/megatron-core-moe-dev/ernie55-perf/runs/ul4ztlbu)

| Comparison | Head | Mean CE error | All valid CE L2 / peak | Sampled raw logits L2 / peak | Head-input gradient L2 / peak | Fixed gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Original Swift live → original Swift reload | Main | 0.000000% | 0.000000% / 0.000000% | 0.000000% / 0.000000% | 0.000000% / 0.000000% | Pass |
| Original Swift live → original Swift reload | MTP | 0.000000% | 0.000000% / 0.000000% | 0.000000% / 0.000000% | 0.000000% / 0.000000% | Pass |
| Original Swift reload → Bridge reload | Main | 0.000001% | 0.000008% / 0.000011% | 0.000000% / 0.000000% | 0.000037% / 0.000159% | Pass |
| Original Swift reload → Bridge reload | MTP | 0.000001% | 0.000016% / 0.000007% | 0.000000% / 0.000000% | 0.000000% / 0.000000% | Pass |
| Original Swift live → Bridge reload | Main | 0.000001% | 0.000008% / 0.000011% | 0.000000% / 0.000000% | 0.000037% / 0.000159% | Pass |
| Original Swift live → Bridge reload | MTP | 0.000001% | 0.000016% / 0.000007% | 0.000000% / 0.000000% | 0.000000% / 0.000000% | Pass |

The target counts below include all four replicas. Each replica repeated the same packed row.

- Main used 1,048,320 valid CE targets, 172,128 sampled raw logits and 4,325,376 sampled head-input gradient elements.
- MTP used 1,048,320 valid CE targets, 166,912 sampled raw logits and 4,194,304 sampled head-input gradient elements.

In both reloaded processes, every valid main target received a CE gradient of 7.631258085893933e-6 and every valid MTP target received 7.631257972207095e-7. The remaining 256 positions per head had zero CE gradient. Both reloads performed the native first microbatch of a GA8 update and stopped before changing any weights.

[Three paired measurements and per-replica coverage](benchmarks/common-state-heads.json)

In the same-state diagnostic, all 264 MTP sample positions on each replica had identical top-10 logical expert sets, FP32 routing scores, attention outputs, FFN outputs and head inputs in Swift and Bridge. Swift supplied an FP32 residual at the MTP hidden-norm input and Bridge supplied BF16. Rounding the Swift residual to BF16 reproduced all 17,301,504 Bridge input coordinates bitwise. The hidden-norm outputs also matched bitwise.
