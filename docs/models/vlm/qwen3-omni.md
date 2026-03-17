# Qwen3-Omni Development Plan

Qwen3-Omni is a multimodal Qwen family model with text, image, video, and audio inputs. This page records the implementation plan for adding Qwen3-Omni support to Megatron Bridge on branch `omni3`.

The immediate goal is not to design a new multimodal stack from scratch. The goal is to port Qwen3-Omni incrementally by reusing the existing Qwen3-VL and Qwen2.5-Omni support in Megatron Bridge and following the integration shape used in Pai-Megatron-Patch.

## Reference Baseline

The reference implementation is the Qwen3-Omni support in `Pai-Megatron-Patch`, with the following characteristics:

- Language backbone is a Megatron/MCore GPT-style model with MoE support.
- Vision encoder and projector are custom multimodal modules derived from the Qwen3-VL path.
- Audio encoder is loaded from the Hugging Face Qwen3-Omni implementation.
- Multimodal embeddings are fused before the language decoder.
- Checkpoint conversion is handled by explicit HF <-> Megatron mappings for language, vision, and audio modules.

That reference also exposes important implementation constraints that should carry into the first Megatron Bridge milestone:

- Pipeline parallelism is primarily a language-backbone feature.
- Vision pipeline parallelism is not a required day-1 target.
- Virtual pipeline parallelism is not a required day-1 target.
- Flash decode and multimodal KV-cache support are not required day-1 targets.
- The first implementation should prefer training and checkpoint conversion over full inference feature parity.

## Target Integration Strategy

We should build Qwen3-Omni as an incremental extension of the existing Qwen3-VL support in Megatron Bridge rather than as a standalone model family.

Planned reuse:

- Reuse `src/megatron/bridge/models/qwen_vl/qwen3_vl_provider.py` as the starting point for provider design.
- Reuse `src/megatron/bridge/models/qwen_vl/qwen3_vl_bridge.py` as the starting point for HF config and weight mapping patterns.
- Reuse the existing `modelling_qwen3_vl` package where the vision stack is already implemented.
- Reuse the Bridge VLM data path where possible, then extend it for audio fields only where necessary.

Planned additions:

- A Qwen3-Omni provider that extends the Qwen3-VL model/provider contract with audio-specific config and runtime wiring.
- A Qwen3-Omni bridge that maps HF Qwen3-Omni configs and parameters to Megatron Bridge modules.
- A multimodal training step that supports image, video, and audio tensors together.
- Example recipes and conversion scripts for at least one MoE checkpoint.

## Delivery Roadmap

The implementation order should be strictly staged. We should not expand multimodal scope and parallel scope at the same time.

### Stage 0: Development Baseline

- Finalize the roadmap and success criteria.
- Reuse the local Qwen2.5-Omni implementation as the Bridge-native reference for package structure and testing style.
- Reuse the Pai-Megatron-Patch Qwen3-Omni implementation as the architecture and parallelism reference.

Success criteria:

- The team agrees on the staged plan below.
- The repository contains a single source-of-truth development document for Qwen3-Omni.

### Stage 1: Qwen3-Omni Minimal Bring-Up and Tests

Stage 1 is the first real milestone and should stay narrow.

Scope:

- HF -> Megatron checkpoint import for `Qwen3-Omni-30B-A3B-Instruct`.
- Megatron -> HF checkpoint export for the same model family.
- A minimal runnable Qwen3-Omni path that brings up the language backbone first.
- Test suite fully passing for the code introduced in this stage.

Implementation intent:

- Prioritize model construction, bridge registration, checkpoint mapping, and testability.
- Avoid broad multimodal optimization work in this stage.
- Allow temporary stubs or guarded paths for not-yet-enabled vision/audio runtime branches if needed, as long as tests and documented limitations are explicit.

Success criteria:

- Branch can construct a Qwen3-Omni model.
- Branch can import and export a real HF checkpoint at least for the implemented submodules.
- New unit and functional tests pass reliably.
- Documentation clearly states what is and is not active in Stage 1.

### Stage 2: Vision Support

After the minimal path is stable and tests are green, add vision support.

Scope:

- Image and video forward path.
- Vision token handling and multimodal position handling.
- Vision-related checkpoint mappings.
- Vision-aware training step coverage.

Success criteria:

- Synthetic and small real vision batches run forward and backward.
- Vision-related import/export mappings are validated.
- Tests for provider, bridge, and model path remain green.

### Stage 3: Audio Support

Audio comes after vision, not in parallel with it.

Scope:

- Audio tower wiring.
- Audio feature extraction and insertion into the multimodal embedding stream.
- Audio-related checkpoint mappings.
- Audio-aware training step and smoke tests.

Success criteria:

- Synthetic audio-containing batches run forward and backward.
- Audio import/export mappings are validated.
- Existing language and vision tests remain green.

### Stage 4: Parallelism Enablement

Only after language, vision, and audio paths are individually stable should we expand parallel support.

Scope:

- Bring up and validate `TP`, `PP`, `EP`, `ETP`, `SP`, and then `CP`.
- Validate model behavior under each supported parallel mode.
- Add focused tests and example configs for supported distributed layouts.

Success criteria:

- Supported parallel configurations are documented with explicit limits.
- Each claimed parallel mode has a runnable validation path.
- No unsupported parallel mode is silently accepted.

### Deferred

- Full flash-decode support.
- Multimodal inference KV-cache parity.
- Vision pipeline parallelism beyond what is strictly required.
- Virtual pipeline parallelism.
- RL integration.
- PEFT until the full-parameter path is stable.

## Parallelism Plan

Parallelism support should be staged after feature bring-up, not bundled into the first milestone.

### Bring-Up Order

1. Single-rank model construction and checkpoint conversion
2. Single-rank training smoke test
3. Language-path distributed support
4. Vision-path distributed validation
5. Audio-path distributed validation
6. Broader parallel combinations

### Target Support Order

- `TP`
- `PP` for the language backbone
- `EP`
- `ETP`
- `SP`
- `CP` only after multimodal RoPE and batch handling are verified

### Explicit Constraints

- Vision encoder should be treated as effectively non-pipelined until proven otherwise.
- Audio encoder should be treated as a first-stage multimodal tower until proven otherwise.
- VPP is out of scope for the first implementation passes.
- If CP creates correctness risk, it should be deferred instead of partially supported.

## Work Breakdown

### 1. Model and Provider Layer

- Create a new provider module under `src/megatron/bridge/models/qwen_vl/` or a dedicated `qwen_omni/` package after confirming which option better fits existing Bridge organization.
- Start from the Qwen3-VL MoE provider and add:
  - stage-1 language-path fields first
  - vision fields next
  - audio config fields last
  - token ids and freeze controls for language, vision, and audio modules
  - explicit sequence-parallel and pipeline constraints
- Decide whether Qwen3-Omni should reuse the existing Qwen3-VL model class with extension points or require a dedicated `Qwen3OmniModel`.

Exit criteria:

- A provider can be instantiated from HF config.
- A distributed model can be constructed without checkpoint loading.

### 2. Custom Modeling Layer

- Audit whether the current `modelling_qwen3_vl` package can absorb the Stage-1 minimal path cleanly.
- Add vision support before audio support.
- Add audio support only after the vision path is stable.
- If not, introduce a dedicated `modelling_qwen3_omni` package containing:
  - Omni model wrapper
  - audio feature extraction path
  - multimodal embedding merge logic
  - any deepstack integration needed by the decoder
- Keep the delta from `qwen3_vl` narrow and documented.

Exit criteria:

- Stage-1 forward path constructs and runs for the implemented minimal scope.
- Vision forward path works before audio is added.
- Audio forward path is added without regressing language and vision.
- Training mode remains compatible with `pre_process` and `post_process` semantics.

### 3. HF Bridge and Weight Mapping

- Register a new Megatron Bridge for the HF Qwen3-Omni class.
- Map top-level config into provider fields.
- Add mapping registry coverage in stages:
  - language embeddings, decoder, norms, lm head
  - vision patch embedding, blocks, merger, deepstack merger
  - audio conv, projection, attention, MLP, and final norm layers
- Reuse Qwen3-VL mappings where names are identical.

Exit criteria:

- Stage-1 import/export works for the implemented scope.
- Vision mappings are validated before audio mappings are added.
- Megatron -> HF export produces tensors with expected names and shapes for each enabled submodule.

### 4. Data and Training Step

- Review the existing Bridge VLM data pipeline and determine the minimum extension needed for Stage 1.
- Extend for vision first and audio last.
- Add audio-aware collate and batch handling only where existing abstractions are insufficient.
- Implement or extend a training step that:
  - prepares multimodal batch fields
  - builds multimodal position ids
  - passes image, video, and audio tensors into the model
- Keep the training-step API aligned with current Bridge training entry points.

Exit criteria:

- Stage-1 synthetic batch can run one forward and backward step.
- Vision batch path is stable before audio is enabled.
- The batch path remains compatible with distributed training setup.

### 5. Recipes and Examples

- Add an example directory for Qwen3-Omni, matching the existing Qwen3-VL example layout.
- Add commands for:
  - checkpoint import
  - checkpoint export
  - finetuning / SFT
- Document required environment assumptions, especially the HF transformer version if upstream Qwen3-Omni APIs are still moving.

Exit criteria:

- A user can follow a single README to run import and training.

### 6. Tests and Validation

- Add provider tests for config translation.
- Add bridge tests for HF config mapping and mapping registry coverage.
- Add conversion tests for language first, then vision, then audio mappings.
- Add at least one forward smoke test per delivery stage.
- Add distributed smoke tests only after the corresponding feature path is stable.

Recommended validation order:

1. Provider construction
2. Bridge construction and mapping registry checks
3. Single-rank forward
4. HF -> Megatron import
5. Megatron -> HF export
6. Small distributed training smoke test

## Risks

- The audio encoder is currently tied to Hugging Face Qwen3-Omni internals and may not fit cleanly into the Bridge modeling split.
- Multimodal RoPE and deepstack behavior may interact poorly with CP and PP if ported mechanically.
- Existing Bridge VLM utilities are image/video-centric; audio may require a broader batch contract change.
- The shortest implementation path may increase duplication with `qwen3_vl`; this must be managed carefully.

## Decision Points Before Coding

- Package layout: extend `qwen_vl` or create `qwen_omni`.
- Model reuse boundary: subclass existing Qwen3-VL model or fork a dedicated Omni model.
- Audio integration: wrap HF audio tower directly or port it into a Bridge-local modeling module.
- CP day-1 status: enabled with tests, or explicitly deferred.

## Recommended Delivery Sequence

1. Land the planning document and agree on the staged scope.
2. Implement the minimal Qwen3-Omni bring-up and get new tests fully green.
3. Add vision support and keep all prior tests green.
4. Add audio support and keep all prior tests green.
5. Enable and validate distributed parallel modes one by one.
6. Revisit performance tuning, broader model coverage, and deferred features.

## First Milestone Definition

The first milestone should be considered complete only when all of the following are true:

- Branch can import a real Qwen3-Omni HF checkpoint into Megatron format.
- Branch can construct and train the implemented minimal model path on a synthetic batch.
- New tests added for Stage 1 are all passing.
- Branch documents the current enabled scope and the known limitations.
- Vision, audio, and broader parallel support are explicitly marked as later stages.
