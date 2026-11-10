# TrainingEngine: not implemented yet

Everything below is either rejected with `NotImplementedError` at the point of use or marked with a
`TODO(<tag>)` comment in the code. The tag names the item here. Supported today: construction by model
name or from a recipe, THD packing with CP sharding, `forward_only` logprobs and values,
`forward_backward` with the predefined losses (cross entropy, importance sampling, PPO clip, value
loss) and custom losses, gradient accumulation across calls, `optim_step`, synchronous and asynchronous
checkpoints, HF export (directory and streaming), `cpu()`/`cuda()`, rollout routing replay for MoE
models, several engines sharing one runtime.

## Construction and recipes

- **Registry coverage.** `recipes/registry.py` holds one entry: Qwen3.5-35B-A3B on GB300 up to 131072
  tokens (TP1 CP8 EP16, 16 GPUs minimum). Other models, other GPU families (GB200, H100), and smaller
  context buckets with less context parallelism (fewer GPUs) need entries and a gate run each.
- **Frozen engines need a provider config.** `trainable=False` requires the recipe's model to be a
  `ModelProviderMixin` (`engine.py`, `_build_frozen_model`).
- **Megatron FSDP v2** (`TODO(mfsdp-v2)`, `recipes/base.py`): FSDP wrappers are rejected; the engine must
  zero and scale gradients through the optimizer and run the cross-replica reduction on demand.

## Compute

- **Pipeline parallelism** (`TODO(pp)`): `forward_only` and `forward_backward` raise for PP > 1, so
  `broadcast_output` has no effect yet. Needs the output-stage loss and the summary broadcast.
- **Gradient-reduce overlap** (`TODO(overlap)`): `ddp.overlap_grad_reduce=True` is rejected in
  `forward_backward`. Bucket reductions have to be deferred across accumulation calls.
- **Dense `bshd` layout** (`TODO(bshd)`): `batch_layout="bshd"` raises in packing and CP sharding; only
  THD packing works.
- **Dynamic context parallelism** (`TODO(dynamic-cp)`): one CP degree per call; a per-microbatch degree
  would need planner, shard and unshard changes.
- **Packing algorithm choice** (`TODO(packing)`): first-fit decreasing only.
- **Router replay with sequence parallelism** (`TODO(replay-sp)`): replay raises when TP > 1 with
  sequence parallelism, because each TP rank's router sees a slice of the sequence.
- **Full-logits path** (`TODO(logits)`): losses that declare `requires_logits`,
  `forward_only(return_entropy=True)`, `target_entropy` and `chunked_vocab_apply` raise;
  `ModelOutput.logits` and `ForwardOutput.entropy` are always `None`. Needs the model called without
  labels, TP-sharded logits in `ModelOutput`, a chunked TP-aware entropy, and an input-aligned MTP mask.
  This is also what on-policy distillation with a teacher top-k distribution needs.
- **Detached MTP heads** (`TODO(detached-mtp)`): accepted, but the head's gradient is divided by the
  primary token count like every other parameter instead of the pooled MTP-target count. The exact
  version needs a Megatron-LM option to skip the module's per-rank ratio.

## Open investigations

- **Logprobs depend on the input order when packs are padded.** With `pad_to_max_length=True`,
  `examples/engine/score_packed_parquet.py` fails its reversed-order check: sequences that are last in
  a pack in one order and not in the other get different logprobs from the first context-parallel cut
  onward (about half their tokens differ by 1e-3 or more, worst case above 20), while per-sequence loss
  over supervised tokens moves by at most 0.002. With `pad_to_max_length=False` both orders match.
  The padding is attached to the last sequence of a pack, so that sequence is cut across ranks at
  different points. Not settled: whether this is bf16 rounding amplified by discrete expert routing, or
  an error in how a padded last sequence is handled. To do: histogram the differences on supervised
  tokens, print the values at the worst token, compare the routing decisions of the two orders, and
  repeat in fp32 or on a small model. Until then, use `pad_to_max_length=False` for scoring that must
  not depend on batch composition.

## Multimodal models

`qwen35_vl_35b_a3b_rl_16gpu_gb200_bf16` builds the full Qwen3.5 vision-language model and runs it on
text input. Checked so far on text input: construction, and `forward_only` giving the same mean NLL as
the text recipe. Missing:

- **Image and video input** (`TODO(multimodal)`): `Datum` has no pixel or grid fields, and
  `check_microbatches` rejects image and video placeholder tokens in `input_ids`. Needs the pixel
  data and grid sizes on the Datum, grid positions for visual tokens (the engine passes the same
  position on all three MRoPE axes, which is only right for text), and the per-rank vision mask the
  model requires for CP-sharded packed input.
- **Value head** (`TODO(multimodal)`): `output_head="value"` raises; the head has to replace the
  language model's output layer inside the wrapper.
- **Lookup by model.** The recipe is not in the registry: a multimodal checkpoint passed to
  `TrainingEngine(model, ...)` still resolves to the text recipe. Select the multimodal recipe by name.
- **Not run yet for this recipe:** `forward_backward` and `optim_step`, with and without the vision
  side frozen (`model.freeze_vision_model`, `model.freeze_vision_projection`), routing replay,
  checkpoints, `cpu()`/`cuda()`, and export (weight names against the multimodal HF layout).

## LoRA

- `lora=` on construction and `forward_only(disable_adapters=True)` raise (`TODO(lora)`). Agreed design:
  Bridge `LoRA` applied through a pre-wrap hook after the base weights load, targets = Bridge's default
  four linears plus the GDN `in_proj`/`out_proj`, router excluded, default rank 32 / alpha 64, the
  expert-adapter gradient sync for expert parallelism, adapter-only checkpoints, merged weights in the
  streaming export, and the reference model in the examples becoming the policy with adapters off.
  Adapter-only refit of the inference engine is a follow-up.

## State, export and placement

- **Saving with an open accumulation window** (`TODO(save-mid-window)`): `save_state` requires the
  window to be closed.
- **Export of a value-head engine** (`TODO(value-export)`): the HF causal-LM layout has no slot for the
  head; a critic layout is undefined.
- **Export with `overlap_param_gather`** (`TODO(export)`) and **export of a text-only model through a
  multimodal source** (`TODO(export)`) raise.
- **`cpu()`/`cuda()` with** `overlap_param_gather` (`TODO(overlap)`), MXFP8 grad-buffer reuse
  (`TODO(mxfp8)`), `nccl_ub` (`TODO(nccl-ub)`), `optimizer_cpu_offload` (`TODO(hdo)`) or the
  precision-aware optimizer (`TODO(precision-aware)`) raise.
- **Checkpoint save strategy** (`TODO(checkpoint-config)`): torch_dist with CPU shared-memory staging is
  seeded by hand in `checkpoint.py`; it should come from a Bridge `CheckpointConfig` knob after measuring
  the alternatives.

## Input and output contract

- **Rank contract.** Every rank passes the same global list of Datums. Submission from designated ranks
  (`None` elsewhere) and a bounded-memory streaming provider are reserved and rejected.
- **Entropy and logits** in the outputs depend on the full-logits path above.

## Examples

- The vllm rollout, sleep/wake and weight-refresh helpers live in `examples/engine/vllm_rollout.py`,
  not in the package; a user writing their own loop copies that file.
- The RL examples assume the DAPO-Math parquet and a Megatron checkpoint; no HF-hub-only quickstart.
- Convergence runs with routing replay at 128K and with LoRA have not been done; the runs so far are
  API checks.
