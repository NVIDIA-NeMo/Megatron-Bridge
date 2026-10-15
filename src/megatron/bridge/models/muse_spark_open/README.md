# Muse Spark Open support

Supported checkpoint: **Muse-Spark-Open-816B-A42B**, model ID
`Muse-Spark-Open-816B-A42B`. The family API retains the `MuseSpark` class names.

This implementation uses MCore `HybridModel` with alternating attention and
MoE positions (`*E` per HF decoder block). It makes no MCore source or pin
changes. GQA, sliding/full attention, RoPE/NoPE, sigmoid output gating, latent
MoE projections, sigmoid top-k routing, grouped expert GEMMs, and distributed
dispatch use existing MCore implementations. QKV/gate conversion and output
logit scaling/soft-capping reuse existing Bridge code.

Shared with Muse Glimmer: the parameter-free Q/K-normalized attention base,
QKV/output-gate weight mapping, output multiplier and tanh soft-cap mixin, and
transformer config fields. Glimmer's affine branch post norms and dense MLP are
not equivalent to Spark's gated residuals and normalized latent MoE. Spark keeps
those compositions in `modeling_muse_spark_open/` and invokes HF perception code directly.

The model API is `MuseSparkBridge`, `MuseSparkModel`, `MuseSparkModelBuilder`, and
`MuseSparkModelConfig`, under `megatron.bridge.models.muse_spark_open`. Two small legacy
import modules retain compatibility with previously saved native config targets;
loading and resaving them writes the new names. Existing checkpoint and validation
artifact directories keep their original storage names.

The `modeling_muse_spark_open/` composition supplies per-head attention output
normalization, gated residual mixing, pre-expert normalization, input-scaled
post norms, and normalization of each expert output before its routing weight
is applied. Parameter-free norms use PyTorch RMSNorm; ordinary affine norms
use the existing Transformer Engine norm. Router logits receive the HF scale
before sigmoid selection. These additions represent missing computations,
not precision-only replacements.

Vision/video and waveform audio call the Transformers implementations directly.
The HF vision adapter is also reused. Perception modules live on the embedding
pipeline stage, with replicated parameters and the existing TP gradient hook.
The wrapper accepts full-sequence media placeholders and slices the merged
embeddings for context parallelism. Callers must not pre-slice those inputs.

## Loading configuration

Loading the checkpoint natively needs a Transformers build that includes
Muse Spark Open. Bridge registers the checkpoint's older `muse_spark` config alias
when such a build is installed; source files remain unchanged.

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("/path/to/vl-checkpoint", local_files_only=True)
config = bridge.get_model_config()
```

HF encoder imports are lazy. Constructing a multimodal embedding stage requires
the Transformers Muse Spark Open code. The original perception config is preserved on VL export.

## Text-only conversion

Use the shared `text_only=True` flag to select the language model and its separate
LM head. The returned config uses the same Hybrid decoder and builds no vision,
audio, or projection modules. Source weights are loaded lazily through an indexed
view; the original VL checkpoint is never rewritten.

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "/path/to/vl-checkpoint", text_only=True, local_files_only=True
)
config = bridge.get_model_config()
config.expert_model_parallel_size = 16  # Run with 16 GPU workers for this checkpoint.
models = bridge.get_model(config, wrap_with_ddp=False)
bridge.save_megatron_model(models, "/path/to/text-megatron", low_memory_save=True)
# This also works after bridge.load_megatron_model(...) in a fresh process.
bridge.save_hf_pretrained(models, "/path/to/text-hf")
```

The native config persists `hf_model_text_only=True`. `AutoBridge.from_auto_config`
can recover the text configuration using the original VL checkpoint as its HF
reference. For streaming export, retain a checkpoint-backed bridge created with
`from_hf_pretrained(..., text_only=True)`.

HF export has a flat `MuseSparkTextConfig`, `model.*` decoder keys, and
`lm_head.weight`. It retains tokenizer and generation settings and excludes
perception weights, configs, and processors. Text-only native models reject
perception tensors. Media token IDs remain in the vocabulary but no longer
trigger media insertion or placeholder replacement.

The Transformers text-only `MuseSparkForCausalLM` omits the VL output
multiplier and tanh soft-cap. Export includes `muse_spark_open_transformers_patch.py`, which
reuses the HF text model and applies those missing operations to its existing
LM head. It preserves parameter names and values, including the untied head,
and delegates HF backend capability detection to the original class. This patch
is used only by exported HF checkpoints, never by the Megatron forward pass.
Load the exported model with a Transformers build that includes Muse Spark Open and the exported local code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/text-hf", trust_remote_code=True, dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/path/to/text-hf")
```

This standalone HF model has no Bridge dependency. Loading the unwrapped Transformers
text class directly bypasses the output operations and changes logits.

## Current limits and validation

Text-only validation passes 49 focused tests and a persisted small-model
VL HF -> text-only Megatron -> text-only HF roundtrip. All 39 language tensors
match exactly, including after a fresh native EP4 reload. Fresh standalone HF
loading matches the source VL model's logits, loss, and four cached continuation
tokens exactly. Full EP16 streaming import/export checks all 1,119 language
tensors; only the known final-norm rounding described below changes values.
Native full-model outputs vary even on repeated identical inference. A controlled
expert-accumulation-only FP64 swap restores exact repeats and exact VL/text-only
wrapper agreement, isolating BF16 combination variation; production arithmetic
remains unchanged.
Available storage prevented writing an additional full native/HF checkpoint pair,
so full-model standalone text HF loading remains untested. The model verification
card, `examples/model_verification_cards/muse-spark-open-816b-a42b/card.yaml`,
records the native inference comparison.

The checkpoint is downloaded. The initial 17 focused unit checks passed. A small instance
of the actual HF architecture passes import, native save/reload, and persisted
HF export with exact equality for all 124 tensors, at EP=1 and EP=4. HF inference
from the small exported checkpoint exactly matches the original small HF
checkpoint for text, image-text, audio-text, and audio-image-text.

Full-checkpoint validation is recorded below. The small Bridge forward baseline
has logit differences and one divergent continuation. Diagnostic-only swaps of
attention and expert combination restore exact logits and continuations for all
four small cases. The expert-combination experiment isolates rounding before
versus after the FP32 weighted sum. A further experiment retains native attention
and swaps only Q scaling placement, RoPE arithmetic, and output-gate sigmoid
precision; together with the expert-combination swap, it also restores exact
small-model outputs.
These small-model experiments alone do not establish full-model parity; their
numerical changes are not installed by the production builder.

The first two real decoder layers, with complete perception encoders, also run
all four modalities. Their roundtrip preserves all values except the final
norm's BF16 centered-parameter conversion (1,205 of 8,192 values differ, maximum
0.015625); four residual gates are losslessly promoted to FP32, as in HF runtime.
Controlled swaps between existing centered and ordinary norm modules reproduce
the final-norm effect on identical activations. This numerical behavior remains
unchanged. Router correction biases retain their original FP32 values, and HF
vision rotary frequencies are restored using HF's initializer after dtype moves.
Reduced models are diagnostic fixtures and do not establish full-model quality.

The full checkpoint has completed import, native save, fresh-process EP16 reload,
and persisted HF export. All 2,000 exported tensors were checked: no missing or
extra keys, 124 lossless residual-gate dtype promotions, and only the final-norm
value changes described above. Its
initial HF export exhausted host memory because source shards interleave small
tensors from many layers with large expert tensors. The family bridge now emits
non-grouped tensors first, allowing the existing writer to release each expert
shard promptly. Small and full persisted export complete with this order.
Full original/exported HF inference now produces identical generated sequences
for text, image-text, audio-text, and audio-image-text, all reaching EOS. Swapping
only the exported final norm into original HF restores exact prefill logits,
isolating that roundtrip logit gap to the BF16 parameter conversion. Full native
Bridge at EP16 also completes all four cases with identical generated tokens.
Its prefill logits differ from HF. Full-depth module swaps, HF's own TP/EP
comparison, same-input MoE replays, and FP64 kernel experiments support numerical
effects; no remaining implementation bug was identified and production numerics
remain unchanged. This is not a bitwise-logit parity claim; the model verification
card records the measured agreement.

Unit checks cover schedule expansion, config serialization, mapping selection,
normalization order, residual gates, router operand dtypes, media embedding
placeholders, FP32 buffer retention, and calls into HF perception modules.
Run these in the Bridge development container:

```bash
uv run python -m pytest tests/unit_tests/models/muse_spark_open/test_muse_spark_open.py -v
```

The initial path requires Transformer Engine, expert tensor parallelism of 1,
dropless routing, and no CUDA graphs, EP communication overlap, virtual pipeline
stages, MTP, FP8/FP4, or fused/dynamic inference. Full-checkpoint import at
TP4/EP16, export at TP2/EP32, four-modality forward comparison at TP4/EP16 with
CP1, CP2, and CP4, and LoRA training at TP4/EP16, TP4/EP32, and TP4/CP2/EP16 are
recorded in the model verification card. Context parallelism must use
A2A or all-gather communication, and training with TP greater than 1 needs sequence
parallelism. Pipeline parallelism through six and eight explicit stages, full-parameter SFT
and pretraining of the complete model on 48 and 64 GB200 nodes, packed long-context SFT at
CP2 (32K) and CP4 (64K), packed Tulu 3 LoRA, and HF export of a fine-tuned checkpoint are
recorded there as well. Cached native generation and gradient
comparisons remain unvalidated. Image-text training reuses the Muse Glimmer collator; the
GB200 recipes are listed in `examples/models/muse_spark_open/README.md`.
