# Gemma 4 E4B Support

**Gemma 4 E4B** (3.8B dense text model) integration for Megatron-Bridge, including HuggingFace checkpoint conversion, numerical parity verification, and TP-distributed training.

Works with **clean Megatron-Core** — no Gemma4-specific CLI arguments or `TransformerConfig` fields are required in MCore. All Gemma4 specifics live in Bridge via `Gemma4E4BProvider` and `Gemma4VLBridge`.

## What's included

| File | Purpose |
|------|---------|
| `src/megatron/bridge/models/gemma/gemma4_layer_specs.py` | Layer spec, attention, dual-RoPE, PLE, shared-KV, `Gemma4E4BProvider` |
| `src/megatron/bridge/models/gemma/gemma4_bridge.py` | Bridge-native HF↔Megatron conversion (`Gemma4VLBridge` for E4B HF checkpoints) |
| `examples/models/gemma/gemma4/parity_check_e4b.py` | Distributed parity check (uses `Gemma4E4BProvider`) |
| `examples/models/gemma/gemma4/slurm_pretrain.sh` | Full pipeline: convert → parity check → training |
| `tests/unit_tests/models/gemma/test_gemma4_{provider,bridge}.py` | Provider and bridge mapping unit tests |

## Quick start

**Step 1 — Convert HuggingFace weights:**

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$PWD/src:$MEGATRON_LM_ROOT
export GEMMA4_CONVERSION_MODE=text

torchrun --nproc_per_node=2 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model /path/to/gemma-4-E4B-it \
    --megatron-path /path/to/gemma4-e4b-megatron \
    --tp 2 \
    --pp 1 \
    --torch-dtype bfloat16
```

**Step 2 — Verify conversion (logit parity):**

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH=$PWD/src \
torchrun --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt /path/to/gemma4-e4b-megatron \
    --tp 2 --bf16 --atol 3.0
```

Expected results:
- fp32: `max |diff|: ~0.15  (atol=0.3)  --> PASSED`
- bf16: `max |diff|: ~2.94  (atol=3.0)  --> PASSED`

**Or run all steps at once (convert → parity → training):**

```bash
NVIDIA_VISIBLE_DEVICES=0,1 \
HF_MODEL_DIR=/path/to/gemma-4-E4B-it \
MEGATRON_CKPT=/path/to/gemma4-e4b-megatron \
TRAIN_DATA_PATH=/path/to/data \
bash examples/models/gemma/gemma4/slurm_pretrain.sh
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
- **GEGLU activation**: tanh-approximate GELU matching HF `gelu_pytorch_tanh`, handled automatically by Bridge's `GatedMLPMapping` (interleaved TP split)
- **Logit softcapping**: `final_logit_softcapping=30.0` applied inside `Gemma4E4BProvider`
- **Checkpoint conversion**: Bridge-native via `Gemma4VLBridge` registered for `Gemma4ForConditionalGeneration`; QKV/GEGLU/PLE handled by `GatedMLPMapping`, `_Gemma4E4BQKVMapping`, `AutoMapping`
- **`Gemma4E4BProvider`**: all-in-one Bridge provider — builds `TransformerConfig`, injects Gemma4 attrs, replaces `rotary_pos_emb`, attaches PLE modules, patches `forward()` for PLE computation, wires shared-KV

## Bridge conversion architecture

```
AutoBridge.from_hf_pretrained("google/gemma-4-E4B-it")
  └─ Gemma4VLBridge                      # registered for Gemma4ForConditionalGeneration
       ├─ provider_bridge()               # text mode → Gemma4E4BProvider for pretraining
       │                                  # auto/vl mode → Gemma4E4BVLProvider for full VL
       ├─ _dense_e4b_mapping_registry()   # language mappings (4 norms, QKV, GEGLU, PLE, ...)
       └─ maybe_modify_loaded_hf_weight() # shared-KV: synthesize zero K/V rows
                                          # (last 18 layers have no k/v proj in HF)
```
