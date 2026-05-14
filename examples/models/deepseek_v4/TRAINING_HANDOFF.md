# DSv4 Training Benchmark Handoff

Branch: `chcui/dsv4-train-pr3562-pr4518` in `NVIDIA-NeMo/Megatron-Bridge`.

This branch is based on Megatron Bridge PR 3562 plus Megatron-LM dev/PR4518 support. On HSG, use the synced worktree:
`/lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-Bridge-dsv4-train`

Important scripts:

- `toy_training.py`: shared DSv4 training smoke harness. Uses mock GPT data, random init, and HF config translation.
- `toy_training_hsg.sbatch`: one-node toy matrix for `adam`, `muon`, `adam_mxfp8`, `muon_mxfp8`.
- `full_training_hsg.sbatch`: full DSv4-Flash Adam smoke, currently PP4/EP8 across 8 nodes.
- `hsg_runtime.sh`: installs/loads side-site dependencies for Transformers 5.8.1 and Muon's emerging-optimizers package.
- `roundtrip_hsg.sbatch`: conversion roundtrip helper.

Known HSG results:

- Full Adam full-model smoke passed: job `2691854`, PP4/EP8, 289.81B params, mock GPT data, no NaNs.
- Toy plain Muon passed with `CSA_BACKEND=unfused`: job `2691750`.
- Toy Adam and toy Adam MXFP8 passed earlier.
- Toy Muon MXFP8 fails in the optimizer path: job `2691857`, `Float16OptimizerWithFloat16Params` is missing `_copy_main_params_to_param_buffer`.
- Plain Muon with `CSA_BACKEND=cudnn_dsa` NaN'd around iteration 2; use `CSA_BACKEND=unfused` as the current working baseline.

Useful commands:

```bash
cd /lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-Bridge-dsv4-train
CASES="muon" CSA_BACKEND=unfused sbatch examples/models/deepseek_v4/toy_training_hsg.sbatch
sbatch examples/models/deepseek_v4/full_training_hsg.sbatch
```

For benchmarking, start by increasing `TRAIN_ITERS`, `SEQ_LENGTH`, and batch sizes in the sbatch environment. The current runs are correctness smokes on mock data, not throughput benchmarks.

No Python packages are vendored in this branch. The HSG scripts populate shared side-site dependency directories on first use.
