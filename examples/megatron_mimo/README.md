# MegatronMIMO Examples

MegatronMIMO is the Megatron-Bridge example path for multimodal training with
heterogeneous parallelism. The core idea is simple: a vision encoder and an LLM
do not need to inherit the same parallelism plan. Each module can use the
layout that fits its own shape, sequence length, and memory profile, then the
modules are connected into one end-to-end training graph.

For the system design and performance results, see
[Heterogeneous Parallelism for Multimodal Large Language Model Training](https://arxiv.org/abs/2605.27678).
The paper covers colocated execution on shared GPUs, non-colocated execution on
disjoint GPU sets, and convergence parity against homogeneous baselines.

MegatronMIMO is still under active development. This directory is a landing
page for model-specific example recipes in Megatron Bridge. Each subdirectory
documents what is currently supported for that model and what is not.

## Model Examples

Current example coverage:

- [Qwen3.5-VL](qwen35_vl/README.md): HF to MegatronMIMO conversion and non-colocated HF-data SFT.
- [LLaVA](llava/README.md): heterogeneous MIMO training (Vicuna-7B + CLIP, with
  an optional Whisper audio encoder), HF→Megatron conversion, and parallelism
  test sweeps.


Model-specific READMEs and scripts document supported layouts, data paths,
conversion commands, training launchers, and known limitations.

## General Guidance

- Start with the model-specific README before running scripts.
- Treat each model's supported/unsupported section as the source of truth.
- Prefer the model subdirectories for runnable workflows. Legacy flat scripts in
  this directory may remain temporarily while examples are being reorganized.
- Keep standard non-MIMO baselines in the corresponding `examples/models/...`
  directory when comparing performance or correctness.
