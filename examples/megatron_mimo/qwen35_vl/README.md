# Qwen3.5-VL MegatronMIMO Examples

This directory contains Qwen3.5-VL examples for **MegatronMIMO**, a
multimodal training path that lets the vision encoder and language model use
different parallelism layouts.

If you are new to this feature, start with one of the tutorials below. The
tutorials explain the training mode, show the commands to run, and state what
has been validated.


## Tutorials

| Goal | Tutorial | Status |
|---|---|---|
| Fine-tune Qwen3.5-VL 27B with non-colocated MegatronMIMO SFT | [`qwen35-vl-non-colocated-sft.md`](../../../tutorials/megatron_mimo/qwen35-vl-non-colocated-sft.md) | Available |
| Fine-tune with colocated MegatronMIMO training | Coming soon | Planned |


## What This Example Covers

The current Qwen3.5-VL example focuses on one validated workflow:

- Dense Qwen3.5-VL models.
- Two MegatronMIMO components: `language` and `images`.
- Hugging Face to MegatronMIMO checkpoint conversion.
- Full-parameter SFT on Hugging Face conversation data.
- Non-colocated training, where the language model and image encoder run on
  disjoint rank ranges.

The reported performance and loss-parity evidence currently applies to the
validated 27B non-colocated workflow in the tutorial. Re-measure performance and
check loss behavior if you adapt the scripts to a different model size, dataset,
or parallelism layout.


## Files

These files are used by the tutorials:

| File | Purpose |
|---|---|
| `conversion.sh` | Converts dense HF Qwen3.5-VL checkpoints to MegatronMIMO format and exports back to HF for a round-trip check. |
| `finetune_qwen35_vl.py` | MegatronMIMO SFT runner for HF conversation data. |
| `slurm_sft.sh` | Multi-node Slurm launcher for the validated 27B non-colocated SFT layout. |


## Relationship to Standard Qwen3.5-VL

MegatronMIMO changes the parallelism layout and launcher for multimodal
training. The standard non-MIMO Qwen3.5-VL examples remain available at
`examples/models/qwen/qwen35_vl/`.

Use the standard examples when you want the regular Megatron-Bridge Qwen3.5-VL
path. Use this directory when you want to try MegatronMIMO layouts for
Qwen3.5-VL.


## More Coverage Coming

Future examples and tutorials will expand coverage for:

- Qwen3.5-VL MoE variants.
- MTP training.
- Packed sequences.
- Energon datasets.
- Evaluation or matched validation/test artifacts.
- MIMO throughput/FLOPs logging.

See the tutorial for the exact command sequence and the validated 27B
non-colocated setup.
