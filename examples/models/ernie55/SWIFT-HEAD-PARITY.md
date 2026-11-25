# Export the original Swift heads for the 256K comparison

Use the ERNIE-enabled `ms-swift-int` and `mcore-bridge-int` supplied with your reference trainer. The exporter calls `swift.megatron.megatron_sft_main` and requires `mcore_bridge.model.gpts.ernie5_5.Ernie5_5GPTModel`. The validation uses `ms-swift-int` revision `8ecce78` and `mcore-bridge-int` revision `f4e9ab9`. Install those packages in their corresponding Megatron-Core environment. Set `SWIFT_MODEL` to the Swift-loadable BF16 HF Base checkpoint with the same weights, YaRN configuration and tokenizer as the candidate. `DATA` contains the `single.jsonl`, `uniform.jsonl` and `skew.jsonl` prepared in the benchmark README. `EX` points to this example directory. Use shared, fresh output directories under `WORK`.

Run once on a host with `datasets` and `numpy`. Preparation writes 32 copies of one already-packed row to a Swift cached dataset, preserves document order, restarts position IDs at each document and adds masked padding to a multiple of 32 tokens per document. Both trainers apply their own training-time target shifts.

```bash
mkdir -p "$WORK/swift-cache" "$WORK/swift-heads" "$WORK/swift-logs"
for layout in single uniform skew; do
  python "$EX/swift_head_parity_256k.py" prepare "$DATA/$layout.jsonl" "$WORK/swift-cache/$layout"
done
```

In the reference environment, run the following on each of 16 nodes with four GPUs per node. Set `NODE_RANK` to the node’s zero-based index, `MASTER_ADDR` to node zero’s reachable hostname, and `MASTER_PORT` to the same available port on all nodes. Use the ordinary Swift model-loading environment and the same three layouts on every node. The reference uses BF16, CP16 contiguous, EP64, MBS1, GBS32, GA8 and MTP1. The wrapper stops before the second microbatch, after capturing the original first microbatch's backward. This retains the trainer's real loss scaling and writes one file per global rank. It changes no model parameters.

```bash
export PYTHONPATH="$MCORE_BRIDGE_INT/src:$MS_SWIFT_INT${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NCCL_ALGO=Ring
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
for layout in single uniform skew; do
  torchrun --nnodes 16 --nproc-per-node 4 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    "$EX/swift_head_parity_256k.py" record "$WORK/swift-heads/$layout" --deterministic -- \
    --model "$SWIFT_MODEL" --model_type ernie5_5 --template ernie5_5 \
    --cached_dataset "$WORK/swift-cache/$layout" --max_length 262144 \
    --truncation_strategy split --packing false --padding_free true --split_dataset_ratio 0 \
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
    --train_iters 1 --lr_decay_iters 6 --lr 1e-5 --min_lr 1e-5 --lr_warmup_iters 5 \
    --lr_decay_style constant --weight_decay 0.1 --clip_grad 1.0 --seed 42 \
    --logging_steps 1 --save_steps 1000000 --no_save_optim true --no_save_rng true \
    --output_dir "$WORK/swift-logs/$layout" --check_model false
done
```

Set `LOGS` to the benchmark output root used for the candidate in section 6c. The check reconstructs all physical token positions of every dense replica and compares the valid main and shifted MTP labels, token CE, raw pre-multimax logit columns and head-input gradients. Sampling uses the same absolute positions and vocabulary columns for both implementations, independently of their CP layouts. The fixture repeats one logical packed sequence across the four replicas. Report both the number of dense replicas and the number of distinct packed inputs when presenting the results.

```bash
for layout in single uniform skew; do
  python "$EX/head_parity_256k.py" "$WORK/swift-heads/$layout" "$LOGS/accuracy-$layout-heads" --check
done
```
