# Gemma 4 E4B Support

**Gemma 4 E4B** (3.8B dense text model) integration for Megatron, including HuggingFace checkpoint conversion, numerical parity verification, and TP-distributed training.

## What's included

| File | Purpose |
|------|---------|
| `train_gemma4_e4b_pipeline.sh` | Full pipeline: convert → parity check → training |
| `train_gemma4_e4b_parity.sh` | Logit parity check: Megatron (TP=2) vs HuggingFace |
| `parity_check_e4b.py` | Distributed parity check implementation |
| `src/megatron/bridge/models/gemma/gemma4_layer_specs.py` | Layer spec, attention, MoE, and dual-RoPE implementation |
| `examples/models/gemma/gemma4/loader_gemma4_hf.py` | HF → Megatron checkpoint loader |
| `tests/unit_tests/models/gemma/test_gemma4_{provider,bridge}.py` | Provider and bridge mapping unit tests |

## Quick start

**Step 1 — Convert HuggingFace weights:**

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$PWD/src:$PWD/examples/models/gemma/gemma4:$MEGATRON_LM_ROOT/tools/checkpoint:$PYTHONPATH

python $MEGATRON_LM_ROOT/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader gemma4_hf \
    --saver core \
    --load-dir /path/to/gemma-4-E4B-it \
    --save-dir /path/to/gemma4-e4b-megatron \
    --model-size gemma4-e4b \
    --tokenizer-model /path/to/gemma-4-E4B-it \
    --bf16 \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 1 \
    --no-checking
```

**Step 2 — Verify conversion (logit parity):**

```bash
NVIDIA_VISIBLE_DEVICES=0,1 \
GEMMA4_HF_DIR=/path/to/gemma-4-E4B-it \
GEMMA4_CKPT=/path/to/gemma4-e4b-megatron \
bash examples/models/gemma/gemma4/train_gemma4_e4b_parity.sh
```

Expected result: `max |diff|: ~0.15  (atol=1.0)  --> PASSED`

**Or run all steps at once (convert → parity → training):**

```bash
NVIDIA_VISIBLE_DEVICES=0,1 \
HF_MODEL_DIR=/path/to/gemma-4-E4B-it \
MEGATRON_CKPT=/path/to/gemma4-e4b-megatron \
TRAIN_DATA_PATH=/path/to/data \
bash examples/models/gemma/gemma4/train_gemma4_e4b_pipeline.sh
```

## Running tests

Provider and bridge mapping unit tests:

```bash
python -m pytest \
    tests/unit_tests/models/gemma/test_gemma4_provider.py \
    tests/unit_tests/models/gemma/test_gemma4_bridge.py \
    -v
```

Multi-GPU tests (TP=2, requires 2 GPUs, when TP-specific tests are added):

```bash
NVIDIA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m pytest tests/unit_tests/models/gemma -v -k "Gemma4 and TensorParallel"
```

## Implemented components

- **Attention**: GQA, mixed sliding-window / full-attention, layer-dependent head dimension, attention normalization
- **RoPE**: dual RoPE (sliding=10000, full=1000000), partial-factor 0.25 for full-attention layers
- **Per-Layer Embeddings (PLE)**: `embed_tokens_per_layer` weight mapping, per-layer projection forwarding through transformer blocks
- **Shared KV layers**: `--num-kv-shared-layers 18` (last 18 layers reuse KV from earlier layers)
- **GEGLU activation**: `--geglu-tanh` flag for tanh-approximate GELU matching HF `gelu_pytorch_tanh`
- **Logit softcapping**: `final_logit_softcapping=30.0` applied in parity check
- **Checkpoint conversion**: QKV fusion/layout mapping, PLE weight mapping, GEGLU interleaved TP split (see fix note below)

## Key fix: GEGLU weight TP splitting

Megatron's GEGLU forward uses `fc1_stride=2` (interleaved gate/up per rank). The checkpoint saver in `tools/checkpoint/saver_base.py` was fixed to split `[gate, up]` weights interleaved rather than contiguously:

```
# Correct: interleaved per-rank layout
rank 0 gets: [gate_rank0, up_rank0]
rank 1 gets: [gate_rank1, up_rank1]

# Wrong (before fix): contiguous split
rank 0 gets: [gate_full]
rank 1 gets: [up_full]
```

Without this fix, TP=2 logit error exceeds 50 (vs expected ~3 for bf16 numerical noise).
