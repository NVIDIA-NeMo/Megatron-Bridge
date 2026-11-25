# TrainingEngine: not implemented yet

Everything below is either rejected with `NotImplementedError` at the point of use or marked with a
`TODO(<tag>)` comment in the code. The tag names the item here. Supported today: construction by model
name or from a recipe, THD packing with CP sharding, `forward_only` logprobs and values,
`forward_backward` with the predefined losses (cross entropy, importance sampling, PPO clip, value
loss) and custom losses, gradient accumulation across calls, `optim_step`, synchronous and asynchronous
checkpoints, HF export (directory and streaming), `cpu()`/`cuda()`, rollout routing replay for MoE
models, LoRA on the policy (adapter-only checkpoints, merged or adapter-only export), several engines
sharing one runtime.

## Construction and recipes

- **Registry coverage.** `recipes/registry.py` holds one entry: Qwen3.5-35B-A3B on GB300 up to 131072
  tokens (TP1 CP8 EP16, 16 GPUs minimum). Other models, other GPU families (GB200, H100), and smaller
  context buckets with less context parallelism (fewer GPUs) need entries and a gate run each.
- **Megatron FSDP v2** (`TODO(mfsdp-v2)`, `recipes/base.py`): FSDP wrappers are rejected; the engine must
  zero and scale gradients through the optimizer and run the cross-replica reduction on demand.
- **GTP weight rematerialization** (`TODO(gtp)`, `recipes/base.py`): a recipe that shards weights
  beyond tensor parallelism (`tensor_parallel_num_weight_shards`,
  `expert_tensor_parallel_num_weight_shards`) is rejected. It has never been run through the engine:
  forward, backward, checkpoints, export and `cpu()`/`cuda()` are untested with sharded weights, and
  Bridge configures GTP process-wide while the engine builds a policy and a frozen model in one process.

## Compute

- **Pipeline parallelism** is supported for `forward_only`, `forward_backward`, `optim_step`,
  `save_state`, `load_state`, `export`, `cpu()`/`cuda()` and router replay, with the value head and
  with LoRA (scores and the loss summary come from the last stage; `broadcast_output` copies them to
  the other stages, and `optim_step` shares the window totals). Content-sized packs
  (`pad_to_max_length=False`) need `model.variable_seq_lengths=True` on the recipe, so stages can
  exchange packs of different lengths. A value head and LoRA adapters are initialized differently than
  without PP, because Megatron seeds each pipeline stage separately.
- **Virtual pipeline stages** (`model.virtual_pipeline_model_parallel_size`) are supported for the same
  calls. Megatron's interleaved schedule sets a rule for the microbatch count of a call, on every
  data-parallel replica: at least `model.microbatch_group_size_per_vp_stage` (default: the pipeline
  size), with a remainder of 0 or at least the pipeline size when divided by it. The engine does not
  add packs to reach such a count; `forward_only` and `forward_backward` raise with the nearest valid
  counts, and the caller passes Datums that pack into one (`preprocess` returns the microbatches
  without running the model). Not run: `overlap_p2p_comm`, a group size other than the default, and
  the checks of `examples/engine/router_replay_gate.py` that record the policy's own routing (they
  need one pack per call; replaying a generation backend's routing is checked).
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

- **Two engines in one process do not get bit-identical Gated DeltaNet gradients.** Two trainable engines
  built from the same recipe and checkpoint and given the same `forward_backward` calls (Qwen3.5 text
  recipe, Triton 3.8.0, deterministic kernels) agree bit for bit on every gradient except those of three
  Gated DeltaNet parameters: `conv1d.weight` (about 150 of 524,288 entries per layer), and single entries
  of `A_log` and `dt_bias`. The differences are at rounding scale (about 1e-6 against gradients of about
  5e-2), are there with one and with two pipeline stages, with and without the load-balancing loss, and
  are the same before and after gradient finalization. Moving an engine with `cpu()`/`cuda()` does not
  change them. Where such an entry has a near-zero gradient, one or two weights out of about 36 billion
  take a different Adam update, and the next update then differs broadly (gradient norms apart by about
  0.5%, a third of the weights updated differently). Since only these parameters' own gradients differ,
  the forward pass and the gradients passed to earlier layers are identical; the variation is in how
  these weight gradients are computed. Not settled: which kernel (the layer's convolution runs through
  `F.conv1d` or a custom causal convolution, and cuDNN deterministic mode is not set), whether it is
  random or depends on which engine was built second (the same weight difference has recurred across
  jobs), and why one differing weight changes the next update that much. To do: repeat with
  `torch.backends.cudnn.deterministic=True`, find which convolution branch runs, and compare the model's
  bf16 weights of the two engines after the first update. `examples/engine/movement_gate.py` allows for
  this: a looser gradient limit, and the second engine takes the first one's weights before the second
  update. Comparisons between separate jobs (scoring, one update, resume) have reproduced exactly.
- **Results are not repeatable on Triton 3.7.1.** The Gated DeltaNet kernel has a race on Triton 3.7.1
  that changes results from call to call; Triton 3.8.0 does not show it. An environment whose Triton is
  3.7.1 needs 3.8.0 ahead of it on the Python path.
- **A job using the `ncclep` MoE dispatcher does not exit.** Scoring through the engine with
  `model.moe_flex_dispatcher_backend="ncclep"` gives the same logprobs as `hybridep`, with and without
  pipeline parallelism, but after `engine.close()` and `torch.distributed.destroy_process_group()` every
  rank prints `NCCL WARN Error: corrupted comm object detected` / `NCCL error 4 at nccl_ep.cc` and the
  process hangs with idle GPUs. Not settled: what still holds the NCCL expert-parallel handle after
  `close()`, and whether the engine should release it or the caller must change its shutdown order.
  `ncclep` training, checkpoints and export have not been run. `hybridep` jobs exit normally.

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

- **LoRA on the critic** (`TODO(lora-critic)`): `lora=` with `output_head="value"` raises. Bridge's LoRA
  freezes every base parameter, the newly built value head included, and the adapter-only checkpoint
  keeps trainable tensors only, so the head has to be unfrozen after LoRA is applied and recorded with
  the adapters. The value-head and LoRA hooks would also each load the base, so a LoRA critic needs a
  single load (base weights, then the value head, then LoRA). `ppo_vllm.py` needs a critic option too.
- **Multi-LoRA**: an engine trains one adapter set. Bridge's `MultiLoRA` (`peft/multi_lora.py`) already
  wraps each target with several adapter slots, picks the slot per token segment
  (`set_tokens_per_adapter_slot`, with slot routing through the MoE dispatcher) and manages the slots
  (`init_adapter_slot`, `load_adapter`). The engine still needs a config for several adapters, an adapter
  per `Datum` with micro-batches ordered by adapter, and per-adapter optimizer state, checkpoints and
  export.
- **User-supplied adapters** (for RL as a service): users run one base model with different adapters
  and manage those adapters in their own way. The adapter code stays Bridge's; a user's own adapter
  implementation would need a fork of Bridge. Today an engine trains one adapter set, fresh or resumed
  from its own checkpoint, and `export(adapter_only=True)` already hands it out. Missing: loading
  adapter weights the user supplies into a running engine and swapping them for another set without
  rebuilding the base (Bridge's `MultiLoRA` already loads and clears adapters per slot), with a check
  that their rank, alpha, targets and expert layout match the `LoraConfig`; optimizer state kept per
  adapter across swaps; and an HF PEFT import for adapters trained elsewhere (Bridge loads Megatron
  adapter checkpoints with `load_peft_adapter_checkpoint` and exports HF adapters, but has no HF adapter
  import).
- **Adapter-only refit** (`TODO(lora-refit)`): the examples refresh vllm with merged weights,
  `bf16(W + s B A)`, which drops update elements below half a bf16 step of W. Refreshing vllm with the
  adapters instead needs a refresh that loads adapter tensors without reloading the weights, the expert
  adapters in a layout vllm loads (vllm 0.29 applies LoRA to Qwen3.5, GDN projections and MoE experts
  included, with per-expert adapters by default and outer-shared ones with `enable_moe_shared_loras`;
  Bridge's adapter export has the per-expert, 3D and shared-outer forms), and its decode cost and
  train/inference gap measured against merged refit.
- **Unmerged export of outer-shared expert adapters** (`TODO(lora-outer-shared-export)`):
  `export(adapter_only=True)` raises for `expert_adapters="outer_shared"`. Bridge's shared-outer adapter
  export names each expert's own factor after that expert's HF tensor, and a checkpoint that packs its
  experts (Qwen3.5) has one such name for all of them, so the PEFT conversion keeps one expert. It needs
  the stacked form the other layouts get: one `[experts, ...]` tensor per packed name.
- **Adapter conversion planning** is redone on every merged streaming export: Bridge rebuilds the
  adapter tasks, and only the base conversion tasks are cached.
- **FP32 adapters** (`lora_dtype`): Bridge's `LoRA.lora_dtype` reaches only adapters on plain
  `torch.nn.Linear`. Adapters on Megatron layers are built in the model's `params_dtype`, and
  `Float16Module` casts every tensor not marked `keep_in_fp32`, so FP32 adapters need Bridge's adapters to
  support them first (FP32 weights marked `keep_in_fp32`, and a cast of the BF16 input in their forward).
  `LoraConfig` does not expose it until then.
- **LoRA+** (a per-group `max_lr` ratio on `lora_B`) is not reachable from the engine, and
  `apply_adam_params` scales by `lr_mult` only, so wiring it needs the ratio rule too.
- **Routing state under LoRA**: `LoraConfig.freeze_routing` (on by default; Bridge's
  `LoRA.freeze_routing`) holds MoE routing at the base model's. With it off, as in Bridge's default LoRA,
  a router's expert bias keeps updating (Bridge saves it with the adapters), and so does quantile
  balancing's `qb_beta`, which Bridge's adapter checkpoint does not keep: on such models
  `forward_only(disable_adapters=True)` then scores the base weights with the moved state, and
  `export(adapter_only=True)` leaves that state out.
- **Vision-language freeze flags**: LoRA freezes every base weight whatever the recipe's `freeze_*`
  flags say; `LoraConfig.skip_frozen_modules` (on by default; Bridge's `LoRA.skip_frozen_modules`) keeps
  the adapters off the parts they freeze. Training the unfrozen parts in full is not reachable: Bridge's
  `VLMLoRA`, which does that, reads a `vision_projection` module that Bridge's Qwen3-VL model does not
  have.
- **Not run on GPU yet:** LoRA on the vision-language recipe.

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
