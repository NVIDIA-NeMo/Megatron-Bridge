# Router replay diagnostic

Small numerical differences can change a discrete MoE top-k selection and
amplify downstream logit differences. This standalone diagnostic records the
experts selected by **Megatron**, then forces those expert sets in **HF**.
HF still computes its own gate logits, sigmoid probabilities, normalization,
scaling, expert outputs, and shared-expert contribution. No Megatron routing
weights or hidden states are copied into HF.

This is **not a parity acceptance test**. Always retain the natural comparison;
an improved replay cosine does not establish correct conversion or prove that
all remaining differences are harmless. Verify weights separately using the
round-trip tools. The report intentionally has no configurable pass threshold.

## Scope

- Nemotron-H sigmoid MoE routing, including the language backbone of Nemotron-H
  multimodal models. Other model families/router formulas are not supported.
  HF modules must expose `route_tokens_to_experts` and expert-bias buffers;
  language backbones named either `backbone` or `model` are recognized.
- Uncached, unpacked, batch-size-one full sequences; TP=PP=CP=ETP=1, no sequence
  parallelism, and exactly EP ranks. Routes are captured before expert dispatch
  for **every token at every language MoE layer**, excluding MTP. EP ranks must
  agree on inputs and selected expert IDs.
- Text, one image, or one video. Media uses the existing native-processor
  inference adapter, including post-resize rectangular grids and tubelet counts.
  Training preprocessing is not changed.
  Media requires the newer native `vision_model`/`vision_projector` API with
  `pixel_values`/`pixel_values_videos`; legacy Omni composites requiring
  `image_flags` and audio inputs are outside this tool's scope.
- Separate processes for Megatron recording and HF replay, so both models need
  not reside in GPU memory simultaneously. HF can shard over visible GPUs.

## Run

Use an installed Bridge GPU environment. Start with an already converted
Megatron checkpoint and its corresponding HF checkpoint/export. For post-SFT
verification, **both phases must name the exported HF directory**, not the
original pretrained source.

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  -m examples.conversion.compare_hf_and_megatron.router_replay record \
  --hf-model-path /path/to/hf-export \
  --megatron-model-path /path/to/megatron-checkpoint \
  --ep 8 --prompt 'The capital of France is' --positions all \
  --output routes.pt

uv run python -m examples.conversion.compare_hf_and_megatron.router_replay replay \
  --hf-model-path /path/to/hf-export \
  --routes routes.pt --hf-device-map auto --output replay.json
```

Use `--trust-remote-code` in both phases when the checkpoint requires trusted
custom Python code. Pin Hub sources with the same `--hf-revision` immutable
commit in both commands. Local checkpoints must remain immutable between runs.
No export is performed by this script and no checkpoint is modified.

`--positions` selects logit rows: `last` (default), `all`, or zero-based indices
such as `12,13,14`. Every selected row contains the complete HF vocabulary;
Megatron vocabulary padding is removed. Routes always cover the entire sequence
regardless of this selection. To examine teacher-forced continuations, include
them in the text prompt and select the corresponding prediction positions.
All-position logits can require substantial CPU memory and disk space.

For media, add `--image-path image.png` or `--video-path video.mp4` to **record**.
Replay consumes the exact stored processor output, not a second preprocessing
pass. Media prompts use the processor's user-turn template and generation
prefix; text prompts are literal tokenizer input. Video requiring HF pruning
must explicitly opt into `--disable-hf-video-pruning` for an **unpruned** matched
reference; this intervention is stored in the artifact. This does not verify
the checkpoint's pruned-video behavior.

For multi-GPU HF video, keep `vision_model` and `vision_projector` entirely on
the same GPU. Native video methods can bypass Accelerate dispatch hooks. If
`auto` splits them, pass a JSON device-map file via `--hf-device-map`; use module
names from your checkpoint and assign both vision components to the same GPU.
The script rejects an incompatible placement instead of modifying model code.
Use only existing optional media dependencies and do not install packages just
to run a text-only diagnostic.

## Controls and interpretation

The script fails if any control fails:

1. Megatron recording must leave selected full-vocabulary logits bitwise equal
   to an unobserved forward.
2. HF recording must likewise preserve its unobserved logits.
3. Replaying HF's own selections must reproduce its natural logits bitwise.
4. Removing the replay wrappers must restore the original HF logits bitwise.
5. Inputs, token order, router dimensions, layer coverage, unique/in-range IDs,
   and applied expert sets must match. Router expert-bias buffers cannot change.

Controls cover the selected logit positions, not every intermediate activation.
When replacing an expert set, retained experts keep native HF enumeration order
and new experts are appended in ascending-ID order. This avoids perturbing
summation order in self-replay; Megatron's dense routing map does not encode
an ordered top-k list.

The JSON report includes natural and replayed raw cosine, centered cosine,
top-1 agreement, absolute logit differences, probability total variation, and
per-layer route-set mismatch counts. These counts mean **token rows with any
different expert**, not the fraction of individual expert assignments.
Replay override counts can differ from natural mismatch counts because earlier
interventions change the hidden states entering later routers.

The recording includes tensor fingerprints, exact inputs, routes, audit-only
weights, selected logits, model identifiers, and topology. Treat these files as
sensitive if prompts or media are private. Existing output files are never
overwritten. The format is diagnostic, not a stable checkpoint API.
