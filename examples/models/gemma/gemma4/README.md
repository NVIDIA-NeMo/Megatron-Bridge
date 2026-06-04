# Gemma 4 E4B Support

**Gemma 4 E4B** (3.8B dense text model) integration for Megatron-Bridge, including HuggingFace checkpoint conversion, numerical parity verification, and TP-distributed training.

Works with **clean Megatron-Core** — no Gemma4-specific CLI arguments or `TransformerConfig` fields are required in MCore. All Gemma4 specifics live in Bridge via `Gemma4E4BProvider`.

## What's included

| File | Purpose |
|------|---------|
| `src/megatron/bridge/models/gemma/gemma4_layer_specs.py` | Layer spec, attention, dual-RoPE, PLE, shared-KV, `Gemma4E4BProvider` |
| `examples/models/gemma/gemma4/loader_gemma4_hf.py` | HF → Megatron checkpoint loader |
| `examples/models/gemma/gemma4/parity_check_e4b.py` | Distributed parity check (uses `Gemma4E4BProvider`) |
| `train_gemma4_e4b_parity.sh` | Logit parity check launcher: Megatron (TP=2) vs HuggingFace |
| `train_gemma4_e4b_pipeline.sh` | Full pipeline: convert → parity check → training |
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

Expected results:
- fp32: `max |diff|: ~0.15  (atol=0.3)  --> PASSED`
- bf16: `max |diff|: ~2.73  (atol=3.0)  --> PASSED`

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
PYTHONPATH=$PWD/src python -m pytest \
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

- **Attention**: GQA, mixed sliding-window / full-attention, layer-dependent head dimension (`kv_channels=256` sliding, `global_kv_channels=512` global), attention normalization (q/k layernorm)
- **RoPE**: dual RoPE (sliding θ=10000 full rotation, global θ=1000000 partial-factor=0.25), handled by `Gemma4RotaryEmbedding` in Bridge
- **Per-Layer Embeddings (PLE)**: `embed_tokens_per_layer` weight mapping; per-layer projection forwarded through transformer blocks via MCore's generic `per_layer_inputs` hook in `TransformerBlock`
- **Shared KV layers**: last 18 layers reuse KV from earlier layers, wired post-construction by `wire_gemma4_kv_sharing()`
- **GEGLU activation**: tanh-approximate GELU matching HF `gelu_pytorch_tanh`, configured as provider default (no CLI flag needed)
- **Logit softcapping**: `final_logit_softcapping=30.0` applied inside `Gemma4E4BProvider`
- **Checkpoint conversion**: QKV fusion/layout mapping, PLE weight mapping, GEGLU interleaved TP split (see note below)
- **`Gemma4E4BProvider`**: all-in-one Bridge provider — builds `TransformerConfig`, injects Gemma4 attrs, replaces `rotary_pos_emb`, attaches PLE modules, patches `forward()` for PLE computation, wires shared-KV

## Key fix: GEGLU weight TP splitting

Megatron's GEGLU forward uses `fc1_stride=2` (interleaved gate/up per rank). The HF checkpoint loader (`loader_gemma4_hf.py`) signals this via `md.geglu = True`, so the checkpoint saver splits `[gate, up]` weights interleaved rather than contiguously:

```
# Correct: interleaved per-rank layout
rank 0 gets: [gate_rank0, up_rank0]
rank 1 gets: [gate_rank1, up_rank1]

# Wrong (contiguous split)
rank 0 gets: [gate_full]
rank 1 gets: [up_full]
```

Without this fix, TP=2 logit error exceeds 50 (vs expected ~3 for bf16 numerical noise).
