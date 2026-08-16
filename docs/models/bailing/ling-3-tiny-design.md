# Ling 3.0 Tiny/Flash Bridge Design

Status: Bridge implementation complete for Ling 3.0 Tiny and Flash HF-to-DCP workflows

Last updated: August 16, 2026

## Decision

Add Hugging Face to Megatron distributed-checkpoint conversion for the public
[`inclusionAI/Ling-3.0-tiny`](https://huggingface.co/inclusionAI/Ling-3.0-tiny) and
[`inclusionAI/Ling-3.0-flash`](https://huggingface.co/inclusionAI/Ling-3.0-flash)
checkpoints through one model-aware Megatron Bridge implementation.

Tiny and Flash share the `BailingMoeV3ForCausalLM`/`bailing_hybrid` architecture,
but they are not size-only aliases: Flash changes the layer schedule, uses direct-Q
MLA, and adds one MTP layer. The bridge therefore validates the public config and
selects variant-specific mappings while keeping model-specific changes out of
Megatron Core.

The converter must instantiate the official Megatron Core `HybridModel`, load the
Hugging Face tensors through Bridge mappings, and save the model through its real
`sharded_state_dict()`. It must not synthesize torch distributed-checkpoint metadata
or expert offsets independently of the target model.

## Goals

- Import the public Ling 3.0 Tiny or Flash Hugging Face checkpoint into a native
  Megatron `torch_dist` distributed checkpoint (DCP).
- Implement bidirectional Hugging Face and Megatron weight mappings so exact
  round-trip verification is possible.
- Preserve the official KDA, gated MLA, dense MLP, grouped MoE, and shared-expert
  parameter semantics.
- Support topology-independent DCP reload, including expert-parallel resharding.
- Fail closed when the source config or checkpoint contains an unsupported Ling 3.0
  variant.

## Non-goals

- New Megatron Core KDA, gated-MLA, HybridModel, or Flash implementation work.
- Training recipes, performance tuning, or optimizer-state conversion.
- Hand-written `ShardedTensor`, `ShardedTensorFactory`, or DCP common-state metadata.

The full Flash checkpoint conversion and strict DCP reload are complete. Flash
HF round-trip weight parity, logit parity, and one-step training smoke remain
runtime gates because the public artifact is approximately 254 GB.

## Pinned Source Artifact

| Property | Value |
|----------|-------|
| Hugging Face model | `inclusionAI/Ling-3.0-tiny` |
| Revision | `a2ee06c0f2de5b171701aee7f73f70a1da75483b` |
| Weight shards | 32 BF16 safetensors shards |
| Indexed tensor keys | 9,283 |
| Weight bytes | 15,787,992,416 |
| `config.json` SHA256 | `9750d847957913f665a13c0b5a6537199e33c6f3ec970d9fcb55a0e5076d4012` |
| Safetensors index SHA256 | `84ef9fe8ef967eeb0545deb1d23c0ce54e86e6b18fa943a7903d7354a79f9cf9` |

The source audit found no MTP/next-token-prediction tensors and no quantized FP8/FP4
weight layout requiring dequantization. It also found no linear projection bias
tensors; the expected semantic tensors `dt_bias` (KDA) and `expert_bias` (MoE router)
are present and are mapped explicitly.

## Architecture Contract

The Bridge implementation supports the following Tiny configuration and rejects
incompatible values rather than guessing:

| Property | Tiny value |
|----------|------------|
| HF architecture | `BailingMoeV3ForCausalLM` |
| HF model type | `bailing_hybrid` |
| Logical decoder blocks | 24 |
| Hidden size | 1,536 |
| Vocabulary size | 157,184 |
| Logical layer group size | 4 |
| Attention schedule | three KDA blocks followed by one MLA block |
| Dense blocks | first logical block only |
| MoE blocks | remaining 23 logical blocks |
| Routed experts | 128 |
| Experts per token | 8 |
| MTP depth | 0 |
| MLA Q LoRA rank | 256 |
| MLA KV LoRA rank | 512 |
| MLA output gate | head-wise |

Each Hugging Face logical block is split into one attention position and one
MLP/MoE position in `HybridModel`. The resulting 48-position main pattern is:

```text
K-KEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+EKEKEKE+E
```

The pattern must not contain an MTP separator or MTP suffix.

### Flash contract

The public Flash configuration is mapped to 42 logical blocks and an 84-position
main pattern. It has two dense blocks followed by 40 MoE blocks, with MLA at the
end of each group of six. Flash uses direct-Q MLA (`q_lora_rank=null`) and one
physical MTP layer. The MTP pattern is `+E`, so the finalized Hybrid pattern is:

```text
<84-position-main-pattern>/+E
```

The public Flash dense and routed widths are `intermediate_size=6144` and
`moe_intermediate_size=768`, respectively. The Bridge validates both values so a
different dense MLP width cannot silently produce a non-public Flash variant.

The public modeling code includes `expert_swiglu_limit_list` metadata, but its
expert implementation uses the common `moe_intermediate_size=768` shape and does
not read those arrays. They therefore do not require a separate weight-shape
mapping in this bridge.

## Dependency Preconditions

The Bridge Megatron Core checkout must include the official Ling Tiny prerequisites:

- HybridModel KDA layer support and projection sharding metadata.
- Kimi Delta Attention with the neutral delta-rule path used by Tiny.
- Head-wise gated MLA.
- Direct-Q MLA and generic MTP support used by Flash.
- The Tiny/Flash HybridModel layer specification.

The Bridge submodule is pinned to the official Megatron-LM draft PR commit
(`f62b8bf20ee5a03c2fd77a28362e568a0451257e`) through `.dev.commit`. This commit
includes the Ling-V3 Tiny HybridModel support, KDA/MLA training and FLOPs
accounting, and the Tiny functional regression. `.main.commit` remains on the
production MCore pin until the draft PR is merged.

The conversion and tests must run in the Bridge container/locked environment. A
separate Ling runtime with an older Transformers version is not a supported Bridge
development environment. The active `fla` import path must also be asserted before
runtime parity tests to prevent an unrelated installation from shadowing the intended
dependency.

## Bridge Implementation

### Files

```text
src/megatron/bridge/models/bailing/
├── bailing_moe3_bridge.py
├── bailing_moe3_mappings.py
├── bailing_moe3_provider.py
├── bailing_moe3_spec.py
└── __init__.py
```

Tiny uses the standard low-rank MLA path. Flash selects the Bridge-local
`BailingMoe3DirectQMLASelfAttention` adapter: it retains MCore's inherited MLA
forward/backward/sharding implementation, but bypasses the current MCore Q/KV
norm resolver so Flash gets a plain `q_proj` plus standalone KV RMSNorm. No file
under `3rdparty/Megatron-LM/` is modified.

Ling 3.0 Tiny and Flash use the stock `HybridModel` implementation. The current
MCore draft exposes several MLA fields only as dynamic attributes, so the Bridge
adds a small serializable `BailingMoe3HybridProvider` shim that declares those
fields; this is required for `run_config.yaml`-based DCP reload and does not alter
MCore.
Model-specific weight-layout logic stays in the Bailing family directory and must not
introduce Ling-specific conditionals into the shared conversion infrastructure.

### Registration

Register one architecture-level bridge:

```python
@MegatronModelBridge.register_bridge(
    source="BailingMoeV3ForCausalLM",
    target=HybridModel,
    provider=BailingMoe3HybridProvider,
    model_type="bailing_hybrid",
)
class BailingMoeV3Bridge(MegatronModelBridge):
    ...
```

Use string-based source registration so importing Megatron Bridge does not require the
custom Hugging Face modeling class. Loading the pinned local Hugging Face artifact may
use `trust_remote_code=True`; conversion should still read weights through Bridge's lazy
safetensors `StateDict` and must not instantiate a complete Hugging Face model on every
rank.

### Configuration Translation

`provider_bridge()` should call the base conversion path, then configure the returned
`BailingMoe3HybridProvider` with the Tiny-specific HybridModel, KDA, MLA, MoE,
router, shared expert, precision, and layer-pattern fields. The provider shim declares
the MLA fields that the current MCore draft exposes only dynamically, so the generated
`run_config.yaml` is self-contained for DCP reload.

It must validate at least:

- `num_nextn_predict_layers == 0`.
- Bias-free attention and MLP projections.
- Head-wise MLA gating.
- Direct, non-LoRA KDA projections.
- The expected 24-layer, group-of-four attention schedule.
- Exactly one dense logical block followed by MoE blocks.
- Consistent expert count, top-k, and shared-expert configuration.

The mapping registry should read config-dependent values from `self.hf_config`; it
should not override conversion-task construction to store a second config copy.

## Layer and Weight Mapping

The Hugging Face layer index cannot be used directly as the Megatron layer index.
Parse the generated HybridModel pattern first and build:

```text
attention_positions[logical_layer]
mlp_positions[logical_layer]
```

Generate concrete mappings for each of the 24 logical layers. This keeps logical to
physical layer translation local to the model family and avoids changing wildcard
semantics in the shared mapping registry.

| Hugging Face source | Megatron target | Mapping behavior |
|---------------------|-----------------|------------------|
| Word embedding | `embedding.word_embeddings` | TP-aware direct mapping |
| Final norm | decoder final norm | Replicated mapping |
| LM head | `output_layer` | TP-aware direct mapping |
| Input and post-attention norms | attention and MLP/MoE leaf norms | Logical-to-physical layer mapping |
| KDA Q/K/V/F/G projections | `self_attention.in_proj` | Local five-section TP mapping in Q/K/V/G/Gate target order; HF F maps to target G and HF G maps to target Gate |
| KDA Q/K/V convolutions | `self_attention.conv1d` | Local three-section TP mapping |
| KDA `A_log` and `dt_bias` | Same semantic parameters | Preserve FP32 values |
| KDA output norm/projection | `out_norm` and `out_proj` | Replicated and row-parallel mappings |
| KDA beta projection | `beta_proj` | Column-parallel mapping |
| MLA Q down/norm/up | Q down/up modules | Replicated/column mappings matching the target module types |
| MLA KV down/norm/up | KV down/up modules | Replicated/column mappings matching the target module types |
| MLA gate/output | `linear_gate` and `linear_proj` | Column/row mappings |
| Dense gate/up/down | dense FC1/FC2 | Existing gated-MLP mapping; do not hand-code the target SwiGLU layout |
| MoE router and bias | router weight/expert bias | Replicated mappings |
| Routed expert gate/up/down | grouped expert FC1/FC2 | EP-aware gated-expert mappings |
| Shared expert gate/up/down | shared FC1/FC2 | Gated/row mappings |

KDA input and convolution mappings require model-local mapping subclasses. Their TP
logic must shard and gather each semantic section independently before concatenating
the local target tensor. Splitting the fully concatenated tensor once is incorrect for
`TP > 1`.

The routed-expert mappings must rely on the instantiated target model and Bridge's
existing EP-aware conversion behavior for global expert IDs. They must not calculate
DCP global offsets themselves.

Because the physical pattern contains the non-standard `K` KDA and `+` MLA symbols,
Bridge's hybrid FLOPs estimator counts KDA and MLA layers separately from standard
attention and GDN. The estimator uses the direct-projection KDA decomposition (fused
Q/K/V/F/G, per-head beta, short convolution, delta-rule state update, and output
projection) and the MLA projection/output-gate/dense-attention decomposition. A
pattern containing `K` still requires an MCore pin with KDA support.

## DCP Conversion Flow

The canonical first conversion uses `TP=1`, `PP=1`, `EP=1`, and `CP=1`:

1. Load the pinned Hugging Face config and lazy safetensors accessor.
2. Build and validate the Tiny `HybridModelProvider` configuration.
3. Instantiate the real Megatron Core `HybridModel` without random weight
   initialization.
4. Load Hugging Face tensors through the registered mappings.
5. Save a model-only checkpoint through `AutoBridge.save_megatron_model()` with the
   low-memory save path enabled.
6. Record the source revision in the provider/run config. Tokenizer metadata is an
   optional post-save artifact and must not gate the model-only DCP write.

The output intentionally contains no optimizer or RNG state. A one-rank writer is a
debugging and memory-management choice, not a fixed checkpoint topology. The saved DCP
must subsequently reload under expert parallelism.

A proposed real-model command, after implementation, is:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 \
  examples/models/bailing/convert_ling3_tiny.py \
  --hf-path <pinned-ling-3-tiny-directory> \
  --output <megatron-dcp-directory> \
  --low-memory-save
```

Prefer the shared conversion CLI if it can express all required validation and model
parallel settings. Add the wrapper only if the shared CLI cannot do so without a public
API change.

## Verification Plan

Verification is gated in the following order.

### 1. Source and config audit

- Verify source revision and config/index hashes.
- Verify all 32 indexed shards and all 9,283 keys are present.
- Verify the config resolves to 24 logical blocks and the exact 48-position pattern.
- Verify the source has no MTP or quantization companion tensors, and distinguish the
  expected KDA/MoE bias semantics from unsupported linear projection biases.

### 2. CPU unit tests

Add focused tests under `tests/unit_tests/models/bailing/` for:

- AutoBridge registration by architecture and model type.
- Tiny config-to-provider translation.
- Exact HybridModel pattern construction.
- Fail-closed validation for MTP, bias, unsupported gate granularity, and incompatible
  layer schedules.
- Logical-to-physical layer index resolution.
- KDA five-section and convolution three-section sentinel round trips.
- Dense, routed-expert, and shared-expert gate/up ordering.

Section sentinel tests should fill each source projection with a distinct value and
require exact recovery, preventing shape-correct but semantically reordered mappings.

### 3. Toy/distributed conversion

The current implementation uses CPU unit fixtures for the semantic TP sentinels and
the full Tiny checkpoint for the distributed path. A future small functional fixture
can preserve the same KDA, MLA, dense, MoE, and shared-expert structures with fewer
layers and experts. The distributed validation already covers:

- HF to Megatron import.
- Native DCP save and strict reload.
- Megatron to HF export.
- Exact tensor round trip (`max_diff == 0`).
- A two-rank TP2 topology on the real checkpoint, plus an eight-rank EP8 DCP reload.

### 4. Full checkpoint conversion

- Consume every one of the 9,283 Hugging Face tensors exactly once.
- Reject missing, unexpected, duplicated, or unconsumed tensors.
- Save through the actual model's sharded state dictionary.
- Strictly reload with the writer topology.
- Export back to 9,283 Hugging Face keys and require bitwise equality for pure
  concatenation/splitting mappings.

The previously audited direct conversion produced 6,207 unsharded Megatron tensors;
use this as a diagnostic invariant, not as a replacement for target-model validation.

### 5. Topology resharding

Strictly load the produced DCP with `TP=1`, `PP=1`, and `EP=8`. Each rank should own
16 of the 128 routed experts, and export must reconstruct experts 0 through 127 without
duplicates or gaps. The current validation has completed the EP8 DCP reload and the
TP2 HF export/DCP reload; a separate EP8 HF export remains an optional follow-up.

### 6. Numerical parity

Run the pinned Hugging Face implementation and Megatron model on identical token IDs
and BF16 weights on a common KDA algorithm. Require:

- Identical greedy next-token selection.
- Full-logit cosine similarity of at least `0.99`.
- Recorded maximum and mean absolute differences.

Tokenizer behavior should be excluded from the core comparison by reusing the same
token IDs. A short generation smoke test should additionally produce coherent text.

### 7. Training smoke test

Run one forward, backward, optimizer, save, and reload step. Loss and gradients must be
finite. This verifies that the converted DCP can serve as a training starting point;
it is not a convergence or performance claim.

## Acceptance Criteria

Ling 3.0 Tiny Bridge support is complete for the current MCore draft when the following
criteria hold:

- Focused unit tests and real full-checkpoint distributed conversion pass.
- Every full-checkpoint source tensor is covered exactly once.
- Native DCP strict load succeeds without factory-key or expert-offset repairs.
- EP8 reshard preserves all 128 experts; EP8 HF export is an optional follow-up
  release gate when that topology-specific artifact is required.
- HF to DCP to HF weights round-trip exactly where the mapping is lossless.
- Full-model logit parity meets the stated threshold.
- One training step and checkpoint reload complete with finite values.
- The model documentation and real conversion example contain commands validated in
  the Bridge runtime.

Run the focused remote tests and the repository-mandated pre-commit checks before a
commit. Do not run the full test suite.

## Implementation Status (2026-08-13)

The Bridge implementation has been exercised against the remote HF artifact and the
official MCore draft pin:

- The pinned source audit found all 32/32 shards, 9,283 indexed keys, and the recorded
  config/index hashes. The actual shard-byte sum is 15,787,992,416 bytes. The source
  has zero MTP tensors and zero quantization companion tensors; its 24 `dt_bias` and
  23 `expert_bias` tensors are expected, not projection-bias violations.
- All 9,283 HF tensors produced 6,207 non-empty conversion tasks; no task was missing
  a source tensor. Real HF-to-HybridModel loading completed with the expected KDA, MLA,
  dense, routed-expert, and shared-expert shapes.
- The full real conversion wrote a native DCP through the target model's sharded
  state dict. Its `run_config.yaml` records the pinned HF revision, and strict TP1
  reload succeeded without modifying MCore source.
- The 12 focused Ling Tiny unit tests pass in the isolated remote Bridge runtime;
  the two TP sentinel tests verify independent KDA section sharding/gathering. The
  remote container's normal full Bridge import is currently blocked by unrelated
  missing Transformers symbols (`Ernie4_5_VL` and `Qwen3VLProcessor`); this is an
  environment issue, not a Ling mapping failure.
- EP8 reload of the full DCP succeeded with 16 local experts per rank and distinct
  first-expert checksums across all eight ranks. A real TP2 run also completed HF
  export (`6207/6207`, 32 safetensors shards), TP2 DCP save, and same-layout TP2
  DCP reload.
- TP1 DCP→HF export consumed all 6,207 conversion tasks and reproduced all 9,283
  source tensors exactly: 9,283 keys compared, zero mismatches, maximum absolute
  difference `0.0`. Only normalized HF config metadata differs between the source and
  exported directory; weights are bitwise identical.
- Common chunk-KDA numerical parity at sequence length 128 gave cosine `0.9978705`,
  maximum absolute difference `2.125`, mean absolute difference `0.1506272`, and the
  same greedy next token (`220`). The pinned HF implementation uses fused recurrent
  KDA for sequence lengths up to 64, while this MCore draft has chunk-KDA only;
  short-sequence recurrent parity is therefore a MCore capability boundary, not a
  Bridge mapping claim.
- A one-step forward/backward/update/save/reload smoke completed with finite loss
  `14.243135`, gradient norm `82.00872`, nonzero tracked update `0.0119629`, and
  identical tracked sums before/after reload (`-3799.5263671875`).
- The hybrid FLOPs branch and exact formula tests pass on the remote MCore pin;
  KDA and MLA physical layers are both included in Tiny/Flash throughput accounting,
  including ragged dense-attention sequence statistics and head-wise MLA gating.

The implementation is ready for the Tiny and Flash conversion workflows against the
official MCore draft pin.
Production support still needs the normal Bridge environment's missing Transformers
symbols resolved. Flash still needs HF round-trip weight parity, logit parity, and a
GPU forward/backward/save/reload smoke with MTP enabled. The Bridge workflow uses the
official MCore draft revision recorded above as its development submodule pin.

## Pull Request Structure

1. Pin the Bridge submodule to the official MCore draft revision through `.dev.commit`;
   keep `.main.commit` unchanged until the MCore draft PR is merged.
2. Submit the Ling 3.0 Tiny Bridge implementation, mappings, provider shim, focused tests, example,
   model documentation, and recorded real-checkpoint validation.

The Bridge implementation keeps architecture-level names such as
`BailingMoeV3Bridge` and dispatches Tiny/Flash by validated HF configuration. It does
not add a size-specific provider class.

## Remaining Flash Runtime Gates

The full Flash HF-to-DCP conversion and strict DCP reload are complete. Remaining
work is operational parity validation in the AIStudio environment:

- run the focused Bridge tests and the direct-Q/MTP model-construction smoke;
- run HF round-trip export and verify every source tensor is bitwise covered;
- run direct-Q/MTP logit parity;
- run one forward/backward/save/reload smoke with MTP enabled.

These gates use the recorded official MCore draft pin.
