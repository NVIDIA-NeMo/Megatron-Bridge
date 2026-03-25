# Qwen3-Omni Thinker Training Plan

This document tracks the development plan for enabling Qwen3-Omni thinker-side training in Megatron Bridge on top of the existing conversion and inference support.

## Goal

The immediate engineering goal is to enable **Qwen3-Omni thinker training** with:

- text, image, video, and audio multimodal inputs
- end-to-end training forward and loss computation
- recipe-level configuration for local and cluster bring-up
- distributed validation that starts from single-rank training smoke and expands to sequence parallel and other parallel strategies

This plan is intentionally scoped to the **thinker** path only. `talker` and `code2wav` remain out of scope for this stage.

## Current Baseline

The current branch already provides:

- Hugging Face to Megatron checkpoint conversion
- Megatron to Hugging Face export
- thinker-side multimodal runtime support for text, image, video, and audio
- example conversion and inference scripts
- unit tests and toy functional conversion coverage
- local smoke validation for full-model multimodal inference and vertically trimmed conversion

## Current Limitations

The current implementation still has the following limitations:

- Megatron inference with `inference_params` is not implemented
- `packed_seq_params` is not implemented
- vision runtime does not support sequence parallel
- distributed parallel validation is not yet done beyond single-rank coverage
- local smoke tests still rely on user-provided multimodal assets for real-data validation

For training enablement, the most important gaps are:

- no training recipe for Qwen3-Omni
- no Omni-specific training step wired into Bridge training
- no Omni task encoder / dataset collation for thinker multimodal training
- no training smoke test
- no SP-aware vision training path

## Reference Implementation To Reuse

The primary references for this work are:

- `src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py`
- `src/megatron/bridge/recipes/qwen_vl/qwen3_vl.py`
- `src/megatron/bridge/recipes/qwen_vl/data/energon/task_encoder.py`
- `src/megatron/bridge/models/qwen_omni/modeling_qwen3_omni/thinker_model.py`

The working assumption is:

- reuse the existing Bridge training stack rather than introducing a parallel custom training entrypoint
- minimize new abstractions until the thinker training path is stable
- keep the first training milestone focused on correctness and bring-up, not performance tuning

## Development Checklist

### Phase 0: Training Design Freeze

- Define the first supported training mode:
  - thinker-only
  - supervised multimodal training
  - no talker/code2wav
  - no packed sequences
- Freeze the first validation target:
  - 1-rank training smoke
  - 1-step and few-step loss/backward validation
- Decide the first data path:
  - toy synthetic batch for CI/unit coverage
  - local real multimodal batch for bring-up

### Phase 1: Minimal Thinker Training Bring-up

- Add an Omni training step module, likely parallel to `qwen3_vl_step.py`
- Implement `get_batch_from_iterator()` for Omni thinker training inputs
- Implement `get_batch()` to produce:
  - tokens
  - labels
  - loss mask
  - attention mask
  - position ids
  - multimodal inputs for image/video/audio
- Wire the training forward path so thinker outputs can participate in standard masked next-token loss
- Ensure backward pass works without sequence parallel or packed sequences

Acceptance criteria:

- single-rank training forward works
- loss is finite
- backward completes
- optimizer step completes for a toy batch

### Phase 2: Recipe and Dataset Wiring

- Add a Qwen3-Omni recipe module under `src/megatron/bridge/recipes/`
- Reuse the Qwen3-VL recipe structure where possible
- Add a minimal Omni dataset/task-encoder path for thinker training
- Reuse Energon patterns only if they reduce code duplication; do not force Energon into the first milestone if it slows bring-up
- Add a local training example script for cluster bring-up

Acceptance criteria:

- recipe builds a valid `ConfigContainer`
- recipe can construct model, optimizer, scheduler, and dataset config
- one local training command can run a smoke step end to end

### Phase 3: Test Coverage for Training

- Add unit tests for Omni training recipe builders
- Add a toy training-step test that validates:
  - batch construction
  - multimodal forward
  - loss computation
  - backward pass
- Add a minimal functional smoke test for thinker training on toy data

Acceptance criteria:

- recipe unit tests pass
- toy training-step test passes
- functional training smoke passes locally

### Phase 4: Sequence Parallel Support

- Audit the current blocker in thinker runtime:
  - vision runtime explicitly rejects `sequence_parallel=True`
- Decide the sequence-parallel contract for multimodal embedding construction:
  - where embedding fusion happens
  - when scatter to sequence-parallel region happens
  - whether vision/audio features remain replicated before fusion
- Implement SP-safe handling for image/video inputs
- Validate that audio path remains layout-compatible under SP

Acceptance criteria:

- thinker training works with `sequence_parallel=False`
- thinker training works with `sequence_parallel=True` for supported multimodal inputs
- no runtime `NotImplementedError` remains for the supported SP path

### Phase 5: Additional Parallelism Validation

- Validate TP bring-up
- Validate PP bring-up
- Validate CP bring-up
- Validate EP/ETP bring-up for MoE thinker configurations
- Document the supported and unsupported parallel combinations

Acceptance criteria:

- each claimed parallel mode has at least one smoke validation path
- unsupported combinations fail explicitly and predictably

## Suggested Implementation Order

The recommended execution order is:

1. thinker training step
2. training recipe
3. toy training smoke test
4. local real-data training smoke
5. sequence parallel support
6. additional parallelism validation

This keeps correctness work ahead of performance work and avoids mixing training bring-up with distributed debugging too early.

## Expected File Additions

The most likely new files are:

- `src/megatron/bridge/models/qwen_omni/qwen3_omni_step.py`
- `src/megatron/bridge/recipes/qwen_omni/qwen3_omni.py`
- `tests/unit_tests/recipes/qwen_omni/test_qwen3_omni_recipes.py`
- `tests/functional_tests/models/qwen_omni/test_qwen3_omni_training.py`

Possible follow-up additions if Energon is used in the first training milestone:

- `src/megatron/bridge/recipes/qwen_omni/data/energon/task_encoder.py`
- related unit tests for task encoding

## Risks To Watch Closely

- sequence-parallel support for vision embeddings is currently the clearest functional blocker
- audio feature layout must remain consistent with the reused HF encoder contract
- multimodal batch collation can become the hidden source of training instability if we over-generalize too early
- recipe work can expand quickly if we try to support all cluster topologies before the first single-rank training smoke is stable

## Non-Goals For This Stage

The following remain out of scope for this stage:

- talker training
- code2wav training
- inference-path completion with `inference_params`
- packed-sequence support
- trying to optimize every parallel combination before the first training milestone is stable
