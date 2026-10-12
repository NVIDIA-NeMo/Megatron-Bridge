---
name: adding-model-support
description: Guide for adding support for new LLM or VLM models in Megatron-Bridge. Covers bridge, provider, recipe, tests, docs, and examples.
metadata:
  when_to_use: User asks to add, onboard, or integrate a new model family; 'add Qwen4 support', 'onboard Llama 5', 'create a bridge for X', 'write a recipe for Y'.
---

# Adding New Model Support in Megatron-Bridge

## Implementation boundaries

- Keep model-support changes in Megatron-Bridge. Do not modify Megatron-Core
  or its submodule to implement a model. Use existing module specs, builders,
  conversion hooks, and narrow Bridge subclasses or compositions.
- Before adding a block, inspect both MCore and existing Bridge implementations.
  Reuse an existing block when it represents the same computation. Differences
  in accumulation precision, rounding, fused kernels, or mathematically
  equivalent LayerNorm/RMSNorm operation ordering alone do not justify a
  replacement implementation.
- Add model-specific code under the family's `modeling_<model>/` directory only
  for genuinely missing computation or composition. For example, normalizing
  each expert before weighted summation differs mathematically from normalizing
  the combined result. Compose existing normalization and linear operations
  wherever possible instead of copying their kernels.
- Invoke Hugging Face vision/video and audio encoders directly; do not port
  their internals into Bridge. Bridge owns their construction, checkpoint
  mapping, modality insertion, placement, freezing, and gradient synchronization.
  Existing native encoder ports are references for previously supported models,
  not a requirement for new support. Follow an explicit request for a different
  encoder implementation when the user provides one.
- Verify support through the requested model API, not only a shared config.
  For HybridModel, trace the actual layer specs and expand per-layer schedules
  to Hybrid positions: `*E` repeated N times has 2N positions. Configure RoPE
  explicitly, since HybridModel defaults to no positional embedding.
- Separate weight-independent checks from checkpoint validation. A checkpoint
  download need not block implementation or small config/block tests; record
  full conversion, logits, and training validation as pending until they run.

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

For unfamiliar config fields, inspect the supplied Hugging Face `modeling_*.py`
implementation and check existing MCore and Bridge components before concluding
that a new block is needed. Ask for missing source only when it is not already
available through the user's supplied artifacts or authorized locations.

### Step 3 — Identify supported modalities

Inspect the config, HF model, and processor for vision/video and audio inputs.
Perception configs commonly appear as `vision_config` or `audio_config` alongside
`text_config`, but names and nesting vary. Route models with either type of
perception encoder through the multimodal wrapper path, including audio-only
language models. A text-only model has no perception integration; its language
config may be flat or nested.

This distinction determines where to read language config fields, whether a
wrapper must combine perception with the language model, and which modality
inputs the functional checks must exercise.

### Step 4 — Check for quantized weights (FP8 / FP4)

Inspect the HF checkpoint's `model.safetensors` (or `model.safetensors.index.json`) for quantized
weight dtypes such as `float8_e4m3fn` (FP8) or `uint8`/`uint4` with accompanying `*_scale_inv` or
`*_scale` tensors. Common signs:

- `config.json` mentions `quantization_config` or dtype fields like `"torch_dtype": "float8_e4m3fn"`
- Safetensors contain `weight_scale_inv` keys alongside the main weight keys
- The model card mentions FP8/FP4/INT4 weights

**Why this matters:** The bridge's `import_ckpt` path does **not** automatically dequantize — it
loads raw quantized values as-is. This produces a silently broken model (random-level loss, huge
grad norms) instead of raising an error.

**Fix:** Dequantize before or during conversion. The current in-repo pattern is to
use a bridge hook plus the shared helpers in
`src/megatron/bridge/models/conversion/quantization_utils.py`. Existing examples
include `src/megatron/bridge/models/ministral3/ministral3_bridge.py`,
`src/megatron/bridge/models/deepseek/deepseek_v3_bridge.py`, and
`src/megatron/bridge/models/minimax_m2/minimax_m2_bridge.py`.

Override `maybe_modify_loaded_hf_weight()` in the bridge class to dequantize on
the fly during import:

   ```python
   def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
       weight = hf_state_dict[hf_param]
       scale_key = hf_param + "_scale_inv"
       if weight.dtype == torch.float8_e4m3fn and scale_key in hf_state_dict:
           return weight.to(torch.bfloat16) * hf_state_dict[scale_key].to(torch.bfloat16)
       return weight
   ```

Always add a sanity check in the verification workflow (e.g., print `std` of a weight tensor —
quantized models typically have `std ≈ 13` before dequantization vs `std ≈ 0.006` after).
Also add or update focused tests when touching export/import quantization paths; see
`tests/unit_tests/models/test_fp8_param_export.py` for current FP8 export coverage.

## Phase 2: Bridge Support

### File structure

**LLM** — Reference: Qwen2 (`src/megatron/bridge/models/qwen/qwen2_bridge.py`)

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py      # Config + weight mappings (no provider file needed)
└── modeling_<model>/      # (optional) Custom nn.Module implementations if needed
    └── ...
```

**VLM / audio-language model** — Use HF perception encoders and a Bridge wrapper.
Reference: Gemma3-VL (`src/megatron/bridge/models/gemma_vl/`).

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py         # Config + weight mappings
├── <model>_provider.py       # Only for VLMs that need custom provide()
└── modeling_<model>/         # Missing language blocks and modality integration
    ├── __init__.py
    └── model.py              # Combines vision + language
```

For a small wrapper, an existing single-file layout can also be followed:

```
src/megatron/bridge/models/<model>/
├── __init__.py
├── <model>_bridge.py
├── <model>_provider.py       # Only for VLMs that need custom provide()
└── modeling_<model>.py       # HF vision + Megatron language wrapper
```

**Model-specific modeling code:** If the model requires custom `nn.Module` implementations
(e.g. a custom RoPE variant, non-standard transformer config, custom thinker/talker
architecture), place them in a `modeling_<model>/` directory or a single `modeling_<model>.py`
file inside the model family folder. Use a directory when there are multiple files (model,
transformer config, custom ops); use a single file when one module suffices. Never put
model-specific modeling code in shared directories or as loose files in the bridge family
directory — keep them namespaced under the `modeling_<model>` prefix.

### Implementation order

**LLM:**
1. **Bridge only** — Register bridge, implement `provider_bridge()` and `mapping_registry()`.
   The bridge calls `super().provider_bridge()` to get a `GPTModelProvider` from `CONFIG_MAPPING`,
   then sets model-specific attributes on it. **Do not create a provider file** — the stock
   provider returned by `super().provider_bridge()` is usually sufficient for LLMs
   (e.g., `GPTModelProvider`, or another base provider selected via `PROVIDER_CLASS`).
   **Do not add size-specific provider classes** whose names combine
   `ModelProvider` with a model-size suffix. Examples of forbidden suffixes
   include `7B`, `200M`, and `A3B`. Model size and architecture fields should
   come from the Hugging Face config through `AutoBridge` /
   `MegatronModelBridge` config mapping. If a recipe needs a fixed
   architecture, configure the base provider inside the recipe function instead
   of exporting a provider subclass.

**VLM:**
1. **Bridge** — Register bridge, implement config and weight mappings.
2. **Provider** (when needed) — Only VLMs that require a custom `provide()` to instantiate a
   combined vision+language model need a provider subclass. The bridge manually calls
   `hf_config_to_provider_kwargs(text_config)` and instantiates the custom provider.
3. **Model class** — Combine vision encoder + language decoder.

For detailed patterns, see:
- VLM: @skills/adding-model-support/vlm-patterns.md
- LLM: @skills/adding-model-support/llm-patterns.md

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

### Encapsulating model-specific layers

When a new model introduces custom or non-standard layers (novel attention variants, custom
normalization, fused expert layouts, MTP heads, etc.), **keep all model-specific logic inside
the model family directory**. Do not modify shared files in `src/megatron/bridge/models/conversion/`
(e.g. `param_mapping.py`, `model_bridge.py`, `quant_mapping.py`) unless the change is genuinely
reusable across multiple model families.

**Principle:** The bridge and provider files for a model family are your primary extension surface.
Shared conversion infrastructure provides hooks and base classes — subclass them locally rather
than adding conditionals to shared code.

#### Prefer Transformer Engine normalization

Before implementing a model-specific normalization with raw PyTorch operations, inspect the exact
Transformer Engine revision in `pyproject.toml`/`uv.lock`, its upstream module API, and Megatron-Core's
`TENorm` adapter. Production Bridge model code should assume Transformer Engine is installed. Use
`TENorm` or the corresponding TE module directly for LayerNorm and RMSNorm; do not add model-local
norm wrappers, copy an existing norm implementation, gate the norm on `HAVE_TE`, or provide a
`torch.nn` fallback. Preserve `eps`, parameter dtype/device initialization, sequence-parallel
metadata, and `zero_centered_gamma`; standard-gamma and zero-centered-gamma norms are not
interchangeable. The attention/linear backend and normalization choice are independent, so a local
attention spec should still use TE normalization.

Treat a custom norm implementation as an architectural exception, not a compatibility pattern. It
is allowed only when the pinned TE API cannot represent the model's exact mathematical and parameter
contract, such as a genuinely parameter-free RMSNorm while TE requires an affine weight. Prefer a
framework-native module such as `torch.nn.RMSNorm(elementwise_affine=False)` or a direct functional
call over a model-local wrapper or copied formula. Document the missing TE capability and test that
the exception introduces no checkpoint parameter. Do not create a wrapper or fallback for ordinary
affine normalization.

Do not introduce a dummy affine parameter merely to reach a fused kernel. For example, current TE
RMSNorm always owns a learnable `weight`, so a genuinely weightless/scaleless RMSNorm must retain a
parameter-free implementation until TE exposes matching semantics. Adding an all-ones frozen weight
would change state-dict keys, optimizer state, distributed-checkpoint schema, and conversion coverage.

TE and reference framework kernels can differ numerically even when their architecture and weights
match. Require exact HF↔Megatron weight round-trip for unchanged parameter representations and pure
layout transformations. Investigate arithmetic parameter transformations and classify discrepancies
using the controlled module-swap procedure below. Assess forward behavior with the correlation
criteria in the parity-testing skill; do not replace reusable norms solely to reproduce one
framework's operation ordering.

#### Strategy 1: Create a local mapping subclass

If the model has a layer whose weight layout doesn't match any existing mapping class, create a
private mapping class in the bridge file or a `<model>_mappings.py` file in the family directory.

Example — GLM's fused expert down-projection disables grouped-export transpose:

```python
# src/megatron/bridge/models/glm/glm_moe_mappings.py
class GLMExpertDownProjMapping(FusedExpertMapping):
    def __init__(self, megatron_param, hf_param, permute_dims=None):
        super().__init__(megatron_param, hf_param, permute_dims, transpose_on_export=False)
```

Example — Nemotron-H's MTP layers flatten indices during resolve:

```python
# Inside nemotron_h_bridge.py (private to the module)
class _MTPFlatteningMapping(MegatronParamMapping):
    def resolve(self, captures):
        return AutoMapping(self._flatten(captures), ...)
```

Example — MiniMax-M2's non-standard QK norm layout:

```python
# Inside minimax_m2_bridge.py (private to the module)
class _FullDimQKNormMapping(MegatronParamMapping):
    def hf_to_megatron(self, hf_weights):
        # Custom scatter logic for full-dim QK norm
        ...
    def megatron_to_hf(self, megatron_weights):
        # Custom gather logic
        ...
```

#### Strategy 2: Override bridge hooks

`MegatronModelBridge` provides several override hooks — use them instead of modifying the base class:

| Hook | When to use |
|------|-------------|
| `mapping_registry()` | Define all weight name mappings (abstract, always overridden) |
| `provider_bridge()` | Configure the provider with model-specific flags (call `super()` then setattr) |
| `maybe_modify_loaded_hf_weight()` | Dequantize, rename, or reshape HF weights before conversion |
| `maybe_modify_converted_hf_weight()` | Synthesize extra HF keys on export (e.g. `inv_freq`) |
| `megatron_to_hf_config()` | Build HF `config.json` for export |
| `hf_config_to_provider_kwargs()` | Override CONFIG_MAPPING behavior for specific fields |

**Accessing HF config in `mapping_registry()`:** The bridge instance has `self.hf_config`
available during conversion — it is set automatically by the dispatch system before
`mapping_registry()` is called. Use it when your mapping registry needs config-dependent
logic (e.g. dynamic MTP layer count, number of experts):

```python
def mapping_registry(self) -> MegatronMappingRegistry:
    hf_config = getattr(self, "hf_config", None)
    num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0) if hf_config else 0
    ...
```

Do **not** override `build_conversion_tasks()` to stash `self._hf_config` — that pattern is
deprecated.

#### Strategy 3: Custom provider subclass (VLMs only)

Most models do **not** need a provider file — the stock provider (e.g., `GPTModelProvider`, or
another base selected via `PROVIDER_CLASS`) is usually sufficient for LLMs. Only create a provider subclass when a VLM needs custom `provide()` logic to instantiate
a combined vision+language model:

```python
# src/megatron/bridge/models/<model>/<model>_provider.py
class MyVLModelProvider(GPTModelProvider):
    image_token_id: int = 0

    def provide(self, ...):
        # Custom model construction combining vision encoder + language decoder
        ...
```

The bridge then references it via `PROVIDER_CLASS = MyVLModelProvider` or instantiates it directly
in `provider_bridge()`.

#### When shared file changes ARE justified

Modify `param_mapping.py` or `model_bridge.py` only when the pattern is **reusable by 2+ model
families**. Examples of justified shared changes:

- `FusedExpertMapping` / `FusedGatedExpertMapping` — used by GLM, DeepSeek, OLMoE, etc.
- `RMSNorm2ZeroCenteredRMSNormMapping` — used by Gemma, Nemotron, etc.
- New `CONFIG_MAPPING` entries — when a standard HF config key maps to a standard provider attribute

If you're tempted to add a model-specific `if model_type == "..."` branch in shared code, or
pattern-matching on specific weight names in shared conversion logic, that's a signal to use a
local subclass or hook override instead.

### Update FLOPs calculator for new architectural blocks

If the model introduces a new computational block that differs from standard attention or MLP
(e.g., Gated DeltaNet / GDN linear attention, Multi-Token Prediction / MTP heads, Mamba SSM layers),
update the FLOPs calculator in `src/megatron/bridge/training/utils/flop_utils.py` so that
training throughput metrics (TFLOPs/GPU) are accurate.

**When to update:** Any time the new block has different FLOPs-per-token than standard self-attention
or standard MLP. Common cases:
- Linear attention variants (GDN, RetNet, RWKV) — replace the `O(s²)` attention term with the
  block's actual operation count
- MTP / speculative decoding heads — add FLOPs for the extra projection and norm layers
- SSM layers (Mamba) — different recurrence FLOPs than attention
- Novel MoE routing — may change the effective expert count

**How to update:**

1. Read the existing `transformer_flops()` function in `flop_utils.py` to understand the structure.
2. Add a conditional block gated on a config attribute (e.g., `experimental_attention_variant`,
   `mtp_num_layers`). Follow the existing MoE pattern for config validation — raise on invalid
   types, assert list lengths, and use direct attribute access instead of `getattr` with fallback
   defaults so that misconfigurations fail explicitly.
3. Compute the per-layer FLOPs for the new block and blend it with the standard attention term
   based on the layer pattern.
4. Add unit tests in `tests/unit_tests/training/utils/test_flop_utils.py` that verify:
   - New-block FLOPs differ from pure-attention baseline
   - Exact formula matches hand-computed expected values
   - Varying the block ratio (e.g., `linear_attention_freq`) changes FLOPs

Reference PR: [#2925 — GDN FLOPs calculator](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2925)
adds GDN support with both the calculator code and comprehensive tests.

## Phase 3: Recipe Support

Recipes provide pre-configured training settings for each model size.

**LLM recipes:** `src/megatron/bridge/recipes/<family>/<model>.py`
**VLM recipes:** `src/megatron/bridge/recipes/<family>/<model>.py`

Each recipe file defines functions for each model size + training mode:
- `<model>_<size>_sft_config()` — Full supervised fine-tuning
- `<model>_<size>_peft_config()` — LoRA/DoRA parameter-efficient fine-tuning
- `<model>_<size>_pretrain_config()` — Pretraining (LLM only, usually)

For detailed recipe patterns, see @skills/adding-model-support/recipe-patterns.md.

Recipes are the right API surface for model-size presets. Do not create or
export size-specific provider subclasses for recipes; either call
`AutoBridge.from_hf_pretrained(...).to_megatron_provider(load_weights=False)` to
derive the provider from HF config, or instantiate the base provider class with
explicit architecture fields inside the recipe function.

### Export checklist

1. Family `__init__.py` — import and add to `__all__`
2. Top-level `src/megatron/bridge/recipes/__init__.py` — wildcard import
3. `train_any_basic.py` — add to `config_map`, docstring, and `--model` choices

## Phase 4: Tests

### Unit tests (no GPU)

```text
tests/unit_tests/models/<model>/
├── __init__.py
├── test_<model>_bridge.py    # Mock HF config → verify provider mapping
└── test_<model>_provider.py  # (optional) Only if custom provider subclass exists
```

### Functional tests (GPU)

```text
tests/functional_tests/test_groups/models/<model>/
├── __init__.py
├── test_<model>_conversion.py  # Toy model HF↔Megatron roundtrip
└── test_<model>_provider.py    # compare_provider_configs (optional)
```

For detailed test patterns, see @skills/adding-model-support/tests-and-examples.md.

## Phase 5: Docs and Examples

### Examples

Model examples: `examples/models/<family>/<model>/`

```text
examples/models/<family>/<model>/
├── README.md
├── inference.sh         # Generation commands (real model, reasonable output)
├── slurm_sft.sh         # SFT training on SLURM
└── slurm_peft.sh        # PEFT training on SLURM
```

**Key deliverable requirement:** The README must include working import and export commands for
a real published model (e.g. `Qwen/Qwen3-8B`, not a toy) using
`scripts/conversion/convert.sh`. Add a model-specific conversion wrapper only when the model
requires preparation or verification that the shared CLI cannot express. The inference script
must produce reasonable output — for LLMs a coherent text continuation, for VLMs a plausible
image description. This is the acceptance bar: conversion runs cleanly and generation makes
sense.

### Documentation

Add a model page at `docs/models/<type>/<model>.md` covering:
- Supported variants and sizes
- Conversion commands
- Training examples (SFT, PEFT)
- Known limitations

## Verification Workflow

Run the authorized checks in the cluster development container. Do not hand
execution back to the user when cluster access and authorization are available.

### Discrepancy investigation

- Check import, persisted checkpoint reload, export, and roundtrip tensor
  coverage before interpreting forward differences. Report missing/extra keys,
  shapes, dtypes, and exact equality separately from forward metrics.
- Reload in a fresh process as well as within a conversion process. Pass the
  intended model-parallel overrides explicitly and verify effective TP/PP/EP
  process-group sizes; existing groups can conceal loader defaults.
- Exercise every requested modality path using identical processed inputs on
  both implementations: text-only, image-text, and an appropriate audio-containing
  case for models that support audio. Preserve input hashes, token IDs, logits,
  greedy continuations, code revisions, and runtime settings.
- Isolate each discrepancy to the earliest differing module with the same
  inputs and weights. Check mapping, layout, masks, positions, routing, and
  precision rather than assuming a small difference is numerical.
- Classify a discrepancy as numerical only after a controlled module-swap
  experiment: replace the suspected module in one execution path with the other
  implementation, keeping inputs, weights, and other settings fixed. Record
  baseline and swapped outputs and show that the discrepancy follows the
  implementation. Use a reverse swap or higher-precision comparison when needed
  to distinguish incorrect mathematics from floating-point effects. A swap that
  closes a gap alone does not prove the original computation was correct.
- Fix confirmed implementation bugs in Bridge and rerun the affected checks.
  Do not fix confirmed numerical differences by replacing reusable production
  blocks. Keep diagnostic swaps in validation code, report their measured
  effects, and leave inconclusive discrepancies unresolved.

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
./scripts/conversion/convert.sh import \
    --hf-model <org>/<model> \
    --megatron-path /workspace/<model> \
    --torch-dtype bfloat16

./scripts/conversion/convert.sh export \
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
uv run python -m pytest tests/unit_tests/models/<model>/ -v
uv run python -m pytest tests/functional_tests/test_groups/models/<model>/ -v --run-gpu
```

## Quick Decision Tree

```
User wants to add a model
│
├─ Has HF link? ─── No ──→ Ask for link (or config.json if private)
│
├─ Has vision/video or audio perception? ─── Yes ──→ Multimodal wrapper path
│   └─ Invoke supplied HF perception encoders directly (Gemma3 wrapper pattern)
│
└─ Text-only ──→ Language model path
    ├─ Standard GPT-style? ──→ Bridge with stock mappings
    └─ Custom layers? ──→ Bridge + local mapping subclasses / hook overrides
        ├─ Custom weight layout? ──→ Local mapping subclass in family dir
        └─ Custom import/export? ──→ Override bridge hooks (maybe_modify_*)
```
