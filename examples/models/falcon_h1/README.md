# Falcon H1 Examples

This directory contains conversion and inference examples for
[`tiiuae/Falcon-H1-0.5B-Instruct`](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct),
the smallest public Falcon H1 instruction checkpoint.

Falcon H1 uses custom Hugging Face model code, so the example scripts pass
`--trust-remote-code`.

## Workspace Configuration

All scripts use `WORKSPACE` as the base directory for converted checkpoints.
Default: `/workspace`.

```bash
export WORKSPACE=/your/shared/workspace
```

Directory structure:
- `${WORKSPACE}/models/Falcon-H1-0.5B-Instruct` - imported Megatron checkpoint
- `${WORKSPACE}/models/Falcon-H1-0.5B-Instruct-hf-export` - exported Hugging Face checkpoint

## Checkpoint Conversion

[conversion.sh](conversion.sh) imports the Hugging Face checkpoint to Megatron,
exports it back to Hugging Face format, copies tokenizer assets into the export
directory, and runs the round-trip checker with `TP=1, PP=1, EP=1, ETP=1`.
This single-process layout works for the 0.5B model.

```bash
bash examples/models/falcon_h1/conversion.sh
```

The round-trip check should complete with all converted parameters matching.
Tokenizer assets are copied from the source Hugging Face checkpoint so the
exported Hugging Face directory can be used as a standalone model path for
generation.

## Inference

[inference.sh](inference.sh) runs greedy text generation from:
- the Hugging Face checkpoint
- the imported Megatron checkpoint, when `${WORKSPACE}/models/Falcon-H1-0.5B-Instruct/iter_0000000` exists
- the exported Hugging Face checkpoint, when `${WORKSPACE}/models/Falcon-H1-0.5B-Instruct-hf-export` exists

```bash
bash examples/models/falcon_h1/inference.sh
```

Default prompt:

```text
What is artificial intelligence?
```

Expected correctness signal: the generated text should be a coherent English
answer about AI systems or machine intelligence. Repeated symbols, unrelated
fragments, or obvious gibberish indicate a conversion or Falcon H1 multiplier
issue.
