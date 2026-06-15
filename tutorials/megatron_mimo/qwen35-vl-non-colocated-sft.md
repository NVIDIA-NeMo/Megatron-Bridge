# Heterogeneous Parallelism for Qwen3.5-VL with MegatronMIMO

This tutorial shows how to fine-tune dense Qwen3.5-VL with *MegatronMIMO*,
using *heterogeneous parallelism*. In the non-colocated setup covered here,
the vision encoder and the language model run on disjoint GPU sets and can use
different declared TP/PP/DP layouts.

You will:

1. Convert a Hugging Face Qwen3.5-VL checkpoint into the MegatronMIMO format.
2. Launch the validated 27B non-colocated SFT job on a multi-node cluster.
3. Review the reported validation evidence: throughput gain and loss parity
   against the standard Megatron-Bridge baseline.

**Scope.** This tutorial is hands-on for the MegatronMIMO path. It covers
**dense** Qwen3.5-VL, two components (`language` + `images`), and
**non-colocated** full-parameter SFT on Hugging Face conversation data. The
hands-on workflow uses the validated 27B layout. Performance is a first-class
part of the tutorial: the results section reports a controlled comparison
against the standard Megatron-Bridge Qwen3.5-VL path on the Hugging Face VLM
datasets currently supported for this path:
[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2),
[RDR](https://huggingface.co/datasets/quintend/rdr-items), and
[MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA).

Familiarity with standard Qwen3.5-VL SFT in Megatron-Bridge is helpful, but the
hands-on steps below focus on what changes for MegatronMIMO. For the system
design, colocated execution, and the full results, see the paper:
[Heterogeneous Parallelism for Multimodal Large Language Model Training](https://arxiv.org/abs/2605.27678).


## 1. Setup

You need a Megatron-Bridge environment (the
[NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
already provides one), Hugging Face access to the `Qwen/Qwen3.5-*` models, and
a shared filesystem visible on every node and inside the container.

Set a workspace and an experiment root:

```bash
export WORKSPACE=/path/to/shared/workspace
export EXPERIMENT_ROOT=${WORKSPACE}/qwen35_vl_mimo
```

The examples use this directory layout:

```
${EXPERIMENT_ROOT}/
  models/mimo/         # converted MegatronMIMO checkpoints
  results/mimo/        # Slurm run outputs
```

The 27B multi-node step needs 3 8-GPU nodes with the default launcher settings.


## 2. Convert a Hugging Face checkpoint

Conversion imports the HF weights into the MegatronMIMO format and, as a
round-trip check, exports them back to HF. For the validated 27B multi-node job,
declare the per-component conversion layout explicitly:

```bash
MIMO_MODEL_ROOT=${EXPERIMENT_ROOT}/models/mimo
WORKSPACE=${MIMO_MODEL_ROOT} \
MODEL_NAME=Qwen3.5-27B \
LANGUAGE_TP=4 \
LANGUAGE_DP=1 \
LANGUAGE_RANK_OFFSET=0 \
VISION_TP=1 \
VISION_DP=1 \
VISION_RANK_OFFSET=4 \
  bash examples/megatron_mimo/qwen35_vl/conversion.sh
```

The component names are fixed for Qwen3.5-VL:

- `language` routes the language-model weights.
- `images` routes the vision-encoder weights.

This writes the MegatronMIMO checkpoint to
`${MIMO_MODEL_ROOT}/Qwen3.5-27B-mimo`, which is the default
`PRETRAINED_CHECKPOINT` for the Slurm launcher.

The conversion layout above uses 5 ranks, while the training layout below uses
17 ranks. That is expected: the MegatronMIMO checkpoint can be loaded into a
different TP/PP/DP layout for training. In practice, use the smaller conversion
layout to create the checkpoint, then let the Slurm training job declare the
validated 17-rank layout.


## 3. Launch 27B non-colocated SFT

Before launching the run, it helps to separate three terms that are used below:

- **Model abstraction.** The standard Qwen3.5-VL training loop already supported
  in Megatron-Bridge is non-MIMO: a regular integrated Megatron model where the
  vision encoder, projector, and language model are internal submodules.
  MegatronMIMO uses the MIMO model abstraction: a graph of computational modules
  connected by activation edges, with support for multiple encoders per
  modality.
- **Placement.** Colocated means module rank sets overlap; non-colocated means
  module rank sets are disjoint.
- **Layout.** Homogeneous means there is one declared parallel topology, or
  equivalent declared component topologies. Heterogeneous means at least two
  declared component topologies differ.

Using those terms, the standard baseline used for validation is **non-MIMO +
colocated + homogeneous**. The run below is **MIMO + non-colocated +
heterogeneous**: `language` and `images` run on disjoint rank sets and declare
different TP/PP/DP layouts.

The validated 27B layout is:

```
ranks 0-15   language   TP=4  PP=2  DP=2     (rank_offset=0)
rank  16     images     TP=1  PP=1  DP=1     (rank_offset=16)
world_size = 17
```

This layout comes from a baseline sweep of standard Megatron-Bridge runs for
the 27B setup: smaller 8-GPU candidates did not fit, while wider TP or deeper
PP alternatives completed but delivered lower active tokens/s/GPU. For the MIMO
comparison, we keep that selected LLM layout unchanged and change only the
image-encoder placement. The image encoder fits on one rank, so non-colocated
MIMO keeps it off the language ranks and lets its work overlap the language
model's critical path.

The validated 27B job is launched with
`examples/megatron_mimo/qwen35_vl/slurm_sft.sh`. The script declares the
17-rank MIMO layout, validates the allocation, and launches the language and
image ranks with an MPMD `srun`.

All knobs live in a single `USER CONFIGURATION` block at the top of the script,
and each one is environment-overridable at submit time. The MIMO layout is set
by these defaults (already the validated 17-rank layout):

```bash
MIMO_LANGUAGE_TP=4   MIMO_LANGUAGE_PP=2   MIMO_LANGUAGE_DP=2   MIMO_LANGUAGE_OFFSET=0
MIMO_IMAGES_TP=1     MIMO_IMAGES_PP=1     MIMO_IMAGES_DP=1     MIMO_IMAGES_OFFSET=16
```

After pointing `WORKSPACE` at the shared filesystem that holds your converted
checkpoint, submit with the defaults:

```bash
WORKSPACE=/path/to/shared/workspace \
  sbatch examples/megatron_mimo/qwen35_vl/slurm_sft.sh
```

Override hyperparameters inline without editing the file:

```bash
sbatch --export=ALL,SEQ_LENGTH=2048,TRAIN_ITERS=100 \
  examples/megatron_mimo/qwen35_vl/slurm_sft.sh
```

Before submitting, check these run settings:

- `PRETRAINED_CHECKPOINT` defaults to
  `${EXPERIMENT_ROOT}/models/mimo/Qwen3.5-27B-mimo`, matching the 27B conversion
  command in Section 2.
- Run outputs land under `${EXPERIMENT_ROOT}/results/mimo/${RUN_NAME}`.
- `MICRO_BATCH_SIZE` is the global microbatch across the language DP group. With
  language DP=2, `MICRO_BATCH_SIZE=2` gives a language-local microbatch of 1,
  matching standard 27B SFT.

<details>
<summary>Cluster allocation note</summary>

The `#SBATCH` defaults request 3 nodes of 8 GPUs and pack the 17 active ranks as
8 + 8 + 1 tasks per node. This suits clusters that allocate whole 8-GPU nodes
exclusively. The only hard requirement is that the allocation provides at least
17 GPUs; adjust the node count, GPUs-per-node, partition, account, and container
settings for your own cluster.

</details>


### Monitoring the run

At startup the job prints a banner with the resolved layout: language and image
TP/PP/DP, active ranks, allocated GPUs, and batch sizes. Check that it matches
the 17-rank layout above; if it does not, cancel the job and fix the launcher
settings before rerunning.

During training, each iteration logs its `lm loss` and step time to the Slurm
output file (`qwen35vl_mimo_sft_<jobid>.out` in the directory you submitted
from, at `LOG_INTERVAL=1`). A healthy run shows a finite `lm loss` that trends
down and a per-iteration step time that stabilizes after the first few warmup
iters. Set `WANDB_API_KEY` to also stream loss and step-time curves to Weights &
Biases. Run artifacts land under `${EXPERIMENT_ROOT}/results/mimo/${RUN_NAME}`.

MIMO tokens/s/GPU is not logged today (heterogeneous FLOPs accounting is not yet
wired in — see [Limitations](#6-limitations-and-references)), so use the
per-iteration step time as the throughput signal.

## 4. Reported validation results — performance

The numbers below are reported validation evidence from a controlled comparison
between the standard baseline and MIMO on Qwen3.5-VL 27B (bf16). You do not
need to run the standard baseline to use the MIMO workflow above.

**Comparison setup.** The standard Megatron-Bridge baseline is TP=4 PP=2 DP=2
(16 GPUs), the strongest standard layout found by a parallelism sweep. In the
terminology from Section 3, that baseline is **non-MIMO + colocated +
homogeneous**. MIMO uses the same language layout plus a single image rank
(language TP=4 PP=2 DP=2 + image TP=1 PP=1 DP=1, 17 active ranks), so the
reported MIMO run is **MIMO + non-colocated + heterogeneous**. Both paths use
Qwen3.5-VL 27B, bf16, the same supported HF VLM dataset path, and matched
language-local microbatch size. The performance sweep ran 20 iterations; timing
is the mean step time over iterations 3–20.

**Metric.** We report **active tokens/s/GPU** — throughput divided by the number
of ranks actually doing work (16 for the baseline, 17 for MIMO) — and the
wall-clock step-time reduction.

The results use the Hugging Face VLM dataset makers currently supported by
Megatron Bridge for this path. All three are real datasets with one image per
sample: CORD-v2 is receipt parsing with variable image resolutions, RDR is
image captioning with a fixed 768x768 image shape, and MedPix-VQA is medical VQA
with variable image resolutions and short question/answer text.

MIMO-win deltas (higher is better) across the three datasets:

| seq / GBS | CORD-v2 wall Δ | CORD-v2 t/s/GPU Δ | RDR wall Δ | RDR t/s/GPU Δ | MedPix wall Δ | MedPix t/s/GPU Δ |
|---|---:|---:|---:|---:|---:|---:|
| 2048 / 32 | 28.1% | +31.0% | 27.6% | +29.9% | 23.8% | +23.6% |
| 2048 / 64 | 30.5% | +35.5% | 35.0% | +44.7% | 32.7% | +39.8% |
| 4096 / 32 | 23.5% | +23.0% | 23.7% | +23.3% | 23.7% | +23.3% |
| 4096 / 64 | 27.7% | +30.2% | 27.9% | +30.6% | 26.2% | +27.5% |

These deltas are measured for the validated 27B non-colocated setup. They are
not a general guarantee for arbitrary models, encoders, datasets, or layouts;
re-measure if you adapt the workflow to a different configuration.


## 5. Reported validation results — loss parity

A faster path is only useful if it trains the same model. In the reported
validation run, the standard baseline and MIMO start from the same converted
Qwen3.5-VL 27B checkpoint and produce matching loss trajectories.

**Setup.** MedPix-VQA, seq=2048, GBS=32, 400 training iterations, from the
pretrained checkpoint. The standard baseline uses MBS=1 and MIMO uses MBS=2, so
both paths see a language-local microbatch of 1. Both load model weights only;
optimizer and RNG state start fresh.

The plot below overlays the `lm loss` curves from the standard baseline and
MIMO runs. The curves track each other over the 400-iteration window, with no
systematic drift between the two paths.

![Qwen3.5-VL 27B loss parity between the standard baseline and MegatronMIMO](figures/qwen35_vl_loss_parity_400iter.png)

In the plot legend, **Standard Qwen3.5-27B** is the standard Megatron-Bridge
baseline (non-MIMO + colocated + homogeneous), and **MegatronMIMO Qwen3.5-27B**
is the MIMO run (MIMO + non-colocated + heterogeneous).

This is the expected loss-parity result for the validated 27B setup: MIMO uses a
different rank layout, but it trains from the same checkpoint and follows the
same loss trajectory as the standard Megatron-Bridge path.


## 6. Limitations and references

The current Qwen3.5-VL MegatronMIMO example targets one workflow well — dense,
two-component, non-colocated full-parameter SFT. The following are not yet
supported or validated, so you do not spend time on paths that will not work:

- **MoE variants** — only dense Qwen3.5-VL is wired up.
- **MTP** — the example disables Multi-Token Prediction layers.
- **Packed sequences** — MIMO packed-sequence behavior is unvalidated.
- **Energon datasets** — use the HF conversation provider.
- **Colocated layouts** — only non-colocated (disjoint ranks) is covered here.

### References

- Paper: [Heterogeneous Parallelism for Multimodal Large Language Model Training](https://arxiv.org/abs/2605.27678)
- MegatronMIMO examples: [`examples/megatron_mimo/`](../../examples/megatron_mimo/README.md)
- Qwen3.5-VL MIMO scripts: [`examples/megatron_mimo/qwen35_vl/`](../../examples/megatron_mimo/qwen35_vl/README.md)
- Standard (non-MIMO) Qwen3.5-VL examples: `examples/models/qwen/qwen35_vl/`
