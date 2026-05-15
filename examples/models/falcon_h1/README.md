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
directory, and runs the multi-GPU round-trip checker. The default parallelism is
`TP=1, PP=1` and `NPROC_PER_NODE=1`, which is sufficient for the 0.5B model.

```bash
bash examples/models/falcon_h1/conversion.sh
```

To run the same check with a different parallelism layout:

```bash
TP=2 PP=1 NPROC_PER_NODE=2 bash examples/models/falcon_h1/conversion.sh
```

The round-trip check should complete with all converted parameters matching.

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

## cw Validation

On cw, use a one-node GPU allocation or an equivalent sbatch job, set shared
cache locations such as `HF_HOME` and `UV_CACHE_DIR`, run `uv sync`, then run:

```bash
uv run python -m pytest tests/unit_tests/models/falcon_h1 -v
WORKSPACE=/shared/workspace/falcon_h1 bash examples/models/falcon_h1/conversion.sh
WORKSPACE=/shared/workspace/falcon_h1 bash examples/models/falcon_h1/inference.sh
```

For PR validation, attach the Slurm job id and log path. The inference log should
show a coherent completion for the default prompt after both HF direct loading
and Megatron checkpoint loading.
