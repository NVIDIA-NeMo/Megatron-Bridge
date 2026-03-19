---
name: adding-model-support
description: Guide for adding support for new LLM or VLM models in Megatron-Bridge. Covers bridge, provider, recipe, tests, docs, and examples. Use when the user asks to add, support, onboard, or integrate a new model, or when creating bridges, providers, or recipes for a new model family.
---

# Adding New Model Support in Megatron-Bridge

## Phase 1: Discovery

### Step 1 — Get the HF model link

Ask the user for the HuggingFace model link (e.g. `https://huggingface.co/Qwen/Qwen3.5-VL-27B`).

If the model is **not public**, ask the user to provide the `config.json` file directly.

### Step 2 — Fetch and analyze config.json

Read the model's `config.json` from HuggingFace (or from the user-provided file). Key fields to extract:

- `model_type` — used for `@register_bridge(model_type=...)`
- `architectures` — the HF model class name (used for `source=...` in registration)
- `tie_word_embeddings` — critical for weight tying
- Architecture fields: `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `vocab_size`, `max_position_embeddings`, `rope_theta`, etc.
- MoE fields (if present): `num_local_experts`, `num_experts_per_tok`, `moe_intermediate_size`
- MLA fields (if present): `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`

If there are config fields you don't recognize from previously supported models (check `CONFIG_MAPPING` in `model_bridge.py` and existing bridges), this likely indicates a **new architectural block** (e.g., a novel attention variant, custom normalization, or a new layer type). Ask the user to provide the HuggingFace `modeling_*.py` implementation of that block so you can understand the computation and create the correct Megatron-side mapping or custom module.

### Step 3 — Determine VLM vs LLM

**VLM** (Vision-Language Model) if config.json contains:
- `text_config` AND `vision_config` sub-configs
- Note: VLMs may or may not have "VL" in the name

**LLM** (Text-only) if:
- No `text_config` / `vision_config`
- Single flat config for the language model

This distinction affects:
- Which files to create (VLMs need a model.py combining vision + language)
- Where to read config fields from (`text_config` vs top-level for VLMs)
- Test patterns (VLMs need vision inputs in functional tests)

## Phase 2: Bridge Support

### File structure

**LLM** — Reference: GPT-OSS (`src/megatron/bridge/models/gpt_oss/`)

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py      # Config + weight mappings
└── <model>_provider.py    # Megatron config + model construction
```

**VLM** — Reference: Qwen3.5-VL (`src/megatron/bridge/models/qwen_vl/`)

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py         # Config + weight mappings
├── <model>_provider.py       # Megatron config + model construction
└── modelling_<model>/        # If using Megatron vision encoder
    ├── __init__.py
    └── model.py              # Combines vision + language
```

OR with HF vision encoder (Reference: Gemma3-VL):

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py
├── <model>_provider.py
└── modeling_<model>.py       # HF vision + Megatron language wrapper
```

### Implementation order

1. **Provider** — Map HF config to Megatron-Core transformer config
2. **Bridge** — Register bridge, implement `provider_bridge()` and `mapping_registry()`
3. **Model class** (VLM only) — Combine vision encoder + language decoder

For detailed patterns, see:
- VLM: [vlm-patterns.md](vlm-patterns.md)
- LLM: [llm-patterns.md](llm-patterns.md)

### Critical: `tie_word_embeddings` for VLMs

For VLMs, `tie_word_embeddings` lives on the **top-level** HF config, NOT on `text_config`. Always read from the parent config:

```python
provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
```

### Critical: Config field location for VLMs

When reading HF config for VLMs, check whether each field is in:
- `hf_config` (top-level) — e.g. `tie_word_embeddings`, `image_token_id`, `video_token_id`
- `hf_config.text_config` — e.g. `num_hidden_layers`, `hidden_size`, etc.
- `hf_config.vision_config` — e.g. vision encoder dimensions

## Phase 3: Recipe Support

Recipes provide pre-configured training settings for each model size.

**LLM recipes:** `src/megatron/bridge/recipes/<family>/<model>.py`
**VLM recipes:** `src/megatron/bridge/recipes/<family>/<model>.py`

Each recipe file defines functions for each model size + training mode:
- `<model>_<size>_sft_config()` — Full supervised fine-tuning
- `<model>_<size>_peft_config()` — LoRA/DoRA parameter-efficient fine-tuning
- `<model>_<size>_pretrain_config()` — Pretraining (LLM only, usually)

For detailed recipe patterns, see [recipe-patterns.md](recipe-patterns.md).

### Export checklist

1. Family `__init__.py` — import and add to `__all__`
2. Top-level `src/megatron/bridge/recipes/__init__.py` — wildcard import
3. `train_any_basic.py` — add to `config_map`, docstring, and `--model` choices

## Phase 4: Tests

### Unit tests (no GPU)

```
tests/unit_tests/models/<model>/
├── __init__.py
├── test_<model>_bridge.py    # Mock HF config → verify provider mapping
└── test_<model>_provider.py  # Direct provider instantiation → verify defaults
```

### Functional tests (GPU)

```
tests/functional_tests/models/<model>/
├── __init__.py
├── test_<model>_conversion.py  # Toy model HF↔Megatron roundtrip
└── test_<model>_provider.py    # compare_provider_configs (optional)
```

For detailed test patterns, see [tests-and-examples.md](tests-and-examples.md).

## Phase 5: Docs and Examples

### Examples

```
examples/models/<type>/<model>/
├── README.md
├── conversion.sh        # HF↔Megatron conversion commands
├── inference.sh         # Generation commands
├── slurm_sft.sh         # SFT training on SLURM
└── slurm_peft.sh        # PEFT training on SLURM
```

- LLM type folder: `examples/models/<model>/`
- VLM type folder: `examples/models/vlm/<model>/`

### Documentation

Add a model page at `docs/models/<type>/<model>.md` covering:
- Supported variants and sizes
- Conversion commands
- Training examples (SFT, PEFT)
- Known limitations

## Verification Workflow

After implementing bridge support, prompt the user to run these commands on the cluster:

### 1. Smoke test (single GPU)

```bash
uv run python -c "
from megatron.bridge import AutoBridge
bridge = AutoBridge.from_hf_pretrained('<org>/<model>')
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.load_hf_weights(model)
for i, (name, tensor) in enumerate(bridge.export_hf_weights(model, cpu=True)):
    print(name, tuple(tensor.shape))
    if i > 10: break
"
```

### 2. Conversion roundtrip (multi-GPU)

```bash
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model <org>/<model> \
    --megatron-path /workspace/<model> \
    --torch-dtype bfloat16

uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model <org>/<model> \
    --megatron-path /workspace/<model>/iter_0000000 \
    --hf-path /workspace/<model>-hf-export
```

### 3. Generation test

For LLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path <org>/<model> --prompt "Hello"
```

For VLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path <org>/<model> \
    --image_path "https://example.com/image.jpeg" \
    --prompt "Describe this image."
```

### 4. Run tests

```bash
uv run pytest tests/unit_tests/models/<model>/ -v
uv run pytest tests/functional_tests/models/<model>/ -v --run-gpu
```

## Quick Decision Tree

```
User wants to add a model
│
├─ Has HF link? ─── No ──→ Ask for link (or config.json if private)
│
├─ Has text_config + vision_config? ─── Yes ──→ VLM path
│   ├─ Has Megatron vision encoder? ──→ Megatron encoder (Qwen3.5 pattern)
│   └─ No Megatron encoder ──→ HF encoder (Gemma3 pattern)
│
└─ No vision config ──→ LLM path (GPT-OSS pattern)
    ├─ Standard GPT-style? ──→ Minimal provider + bridge
    └─ Custom components? ──→ Add modeling module
```
