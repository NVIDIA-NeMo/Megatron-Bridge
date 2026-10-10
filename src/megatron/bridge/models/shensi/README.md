# Shensi

User-facing documentation lives under `docs/models/shensi/`; this README covers the
implementation.

## Layout

```
shensi/
├── configuration_shensi.py   HF-side config mirror plus the Megatron-Core transformer config
├── modeling_shensi.py        Shensi modules: indexed/KDA attention, streams, AttnRes, ROSA, MLP/MoE
├── shensi_bridge.py          HF <-> Megatron config mapping and parameter mappings
├── shensi_hybrid.py          HybridModel glue: decoder stack and pipeline pattern
├── shensi_builder.py         model-config / model-builder path
├── shensi_provider.py        provider: ShensiConfig -> ShensiModel
├── __init__.py
└── README.md
```

## What Shensi adds

The latest Shensi model is a hybrid decoder. Each layer runs one of two attention paths and one
of two feed-forward paths:

* **indexed attention** — DSA-style sparse attention over a low-rank KV latent. A HISA indexer
  selects keys block-first (mean-pooled blocks, `index_block_topk`) and then token-first inside
  the retained blocks (`index_topk_tokens`); the query's own key replaces the weakest selected
  slot (self-position exclusion removes its value direction afterwards), and a per-head sigmoid
  gate scales the output. Every indexed layer runs its own indexer unless `indexer_types` marks
  it `"shared"`, in which case it reuses the previous full indexer's top-k. Training the indexer
  goes through a KL auxiliary loss against the attention distribution over the selected
  candidates (`indexer_loss_coeff`, default 0.01, DeepSeek-V3 lightning-indexer style): the
  selection is non-differentiable, the scores are not. The KV latent is `kv` -> `kv_up`; the
  first indexed layer's `kv_emb` is *added* to the latent (it is tied across indexed layers).
* **linear attention** — Kimi Delta Attention (KDA) from Megatron-Core, with two-stage low-rank
  decay/output gates, a short convolution and a gated output norm.

Around the sublayers:

* **hyper-connections** (`ShensiHyperConnection`) — every sublayer collapses a bank of
  `hc_mult` residual streams with a single `pre` projection + softmax, writes the sublayer output
  back with per-stream `post` weights and optional depthwise-conv features, and keeps the block
  residual on the fixed streams only (the residual anchors). When `hc_active_streams < hc_mult`
  the routed streams also carry per-stream learned decay and low-rank adapters;
  the default (`hc_active_streams == hc_mult`) runs the unconditioned identity-4 form.
  `ShensiStreamEmbeddings` creates the streams from the embeddings (per-stream token tables plus
  depthwise convs) and the bank's `stream_basis` stores them in a low-rank basis between layers.
* **attention residuals** (`ShensiAttentionResidual`) — one `g_proj` emits the channel-wise
  decay/erase/write gates, low-rank depth routing (`q_proj`/`k_proj`) reads the completed block
  summaries, and a two-element `scale` carries the routing temperature and the ReZero gate. Only the depth correction is written into the hyper-connection
  streams; the block prefix stays in the bank (`ShensiAttnResBank`, low-rank basis, exact token
  embeddings in row 0).
* **ROSA** (`ShensiROSAEmbedding`) — suffix-automaton next-token predictions computed once per
  forward and injected into every stream at the configured layers. The automaton itself is
  parameterless; its embedding/value/gate train through the LM loss and, by default
  (`rosa_loss_coef = 0.05`, following the RWKV-v8 ROSA training recipe), through an auxiliary
  next-token cross-entropy on the gated ROSA feature using the tied token embedding as
  classifier (the RWKV-v8 "onlyemb" objective).
  The auxiliary term is attached with Megatron-Core's aux-loss autoscaler, so it changes
  gradients only and does not alter the reported LM loss. It requires
  `rosa_embed_dim == hidden_size` (the tied-embedding case). The automaton itself stays hard
  (no STE): `rosa_soft`/SUFA is a separate differentiable operator and is not part of this port.
* **dense MLP / latent MoE** — dense layers use ReLU² with clamped gate/up paths, a
  previous-token shift and a token-conditioned DeepEmbed scale. Sparse layers route through a
  low-rank latent space (`latent_size`), normalize there, and project back. Routers and expert
  pools are shared per attention-residual block; the ERC loss couples the router rows with the
  expert weights and is attached once per tied pool through Megatron-Core's auxiliary-loss
  autoscaler, so `erc_loss_coef` trains them without changing the reported LM loss.
* **selective fp32 keeping** — the checkpoint's numerically sensitive parameters
  (norms, gates, bank/stream bases, KDA `A_log`/`dt_bias`) stay in fp32 while the model trains in
  bf16.

## Pipeline behavior

The stream loop lives in `ShensiDecoderStack`. Between layers only the compressed streams, the
running residual (`prefix_sum`) and the encoded block bank are kept; at a pipeline stage boundary
these (together with the embeddings and ROSA ids) ride the stage payload as extra sequence rows
(`pack_shensi_payload` / `unpack_shensi_payload`). The payload shape depends on the microbatch, so
pipeline runs enable `variable_seq_lengths` and switch the MoE token dispatcher to `alltoall`
(the `allgather` dispatcher rejects variable sequence lengths).

TP > 1 enables sequence parallelism, which Megatron-Core requires for MoE training. The
token-wise stream/attention-residual path stays on the local chunk; the hyper-connection
convolutions gather the full sequence for their causal window and scatter the local chunk back,
and the indexed attention gathers/scatters around its HISA top-k. Context parallelism is rejected:
the attention-residual bank and the HISA indexer rank over the full sequence.

The per-stream token table is a Megatron-Core `VocabParallelEmbedding`: it uses the padded
vocabulary layout of `embedding.word_embeddings`, is sharded across TP, and follows the
sequence-parallel reduce-scatter when SP is on. The bridge pads the reference (unpadded) rows on
import and crops them on export, so checkpoints stay interchangeable. At the default geometry
this table is ~0.5 B parameters, so sharding divides its weights and gradient accumulation by TP.
Boundary modules follow Megatron-Core's pipeline convention as well: `ShensiStreamEmbeddings`
exists only on the first stage and `hc_head`/`output_attn_res`/`norm` only on the last stage, so
non-terminal stages do not carry their parameters.

The ROSA token table is tied to the input embedding; on the last stage it reuses the shared output
table. Both go through Megatron-Core's `VocabParallelEmbedding` for the vocab-parallel lookup. A
tied ROSA layer therefore lives on the first or last pipeline stage (a middle stage
raises a clear error). Setting `rosa_embed_dim != hidden_size` uses untied local tables instead.

A shared indexer may sit on the pipeline stage after its full indexer: the full indexer's top-k
indices ride the payload as byte-exact rows, so arbitrary pipeline splits keep the same selected
tokens.

## Native reuse

The implementation is rebased on upstream Megatron-Bridge and tracks the latest Megatron-Core.
Every bridge-side component was audited against the closest native implementation in this repo
before it stayed custom:

- The container is MCore's `HybridModel`; the layer is a `TransformerLayer`; MTP is MCore's native
  `MultiTokenPredictionLayer` and `get_gpt_mtp_block_spec` (the same entry point GLM-5 uses for its
  hybrid MTP stack).
- KDA is MCore's `KimiDeltaAttention`; dense/MoE linears come from
  `get_backend_from_config` (TE when installed, local otherwise).
- Latent MoE is MCore's `MoELayer` with `moe_latent_size` and
  `moe_use_norm_before_up_proj`; only the ERC loss remains bridge-side (MCore has no
  expert-specialization objective).
- The dense branch is MCore's `MLP` (fused gate/up `linear_fc1`, `linear_fc2`, and its gated
  clamp path) subclassed only to add the previous-token shift and the DeepEmbed scale. The
  activation is `megatron.core.activations.squared_relu`, set on the module's private config copy,
  so routed experts keep the reference's SiLU. The DeepEmbed scale is per-channel, not the scalar
  `per_token_scale` MCore's `MLP.forward` accepts, which is why the four-line activation step
  stays in the subclass.
- Linears and norms come straight from Megatron-Core's backend factories:
  `get_backend_from_config(config).linear()` (TE `parallel_mode="duplicated"`),
  `.column_parallel_linear()`, `.row_parallel_linear()` and `.layer_norm(rms_norm=True)`
  (`TENorm`). Shensi therefore requires Transformer Engine — the local backend has no
  replicated linear and its torch norm asserts sequence parallelism off — and `finalize()`
  refuses a non-TE backend with a clear error.
- The per-stream token table and the tied ROSA table are MCore `VocabParallelEmbedding`s; the
  bridge's `_StreamTokenEmbeddingMapping` only adds the padded-vocabulary pad/crop around
  `ColumnParallelMapping`.
- Both auxiliary loss terms use Megatron-Core's own autoscalers: the indexer KL goes through
  `DSAIndexerLossAutoScaler` (the same one MCore's DSA indexer uses, scaled by the schedule) and
  its KL reduction is MCore's `indexer_kl_per_row`; the ERC/ROSA terms use
  `MoEAuxLossAutoScaler`. The routed-stream balance term is the same Switch-style form as
  MCore's router loss.
- The block bank's encode/decode projections are plain `F.linear` calls over its basis
  parameters (cast to the activation dtype), not a custom kernel.
- The HISA indexer follows MCore's `DSAIndexer` module shape (`ShensiIndexerSubmodules`,
  `forward_before_topk`/`forward`, replicated projections with TP gradient averaging); the
  projection names follow the reference (`linear_wq_b`, `linear_key`, `linear_key_up`,
  `linear_weights_proj`). MCore's own `index_kpool` is a different
  operator: softmax-weighted pools with a learned compress gate and APE, while the reference
  mean-pools blocks and forces the first/last causal blocks before rescoring. That selection, the
  low-rank cached keys and the shared token table therefore stay bridge-side.
- Hyper-connections use MCore's decomposition (`aggregate` / `apply_h_post` / `apply_h_res` /
  `forward`), but MCore's `HyperConnectionModule` and `fused_mhc_kernels` implement the
  manifold-constrained variant — sigmoid `h_pre`/`h_post` and a Sinkhorn-projected `h_res`; the
  ERNIE 5.5 model in this repo follows exactly that design. The reference's Shensi form instead
  uses a softmax `pre`, top-k routed streams with learned decay and low-rank adapters, and
  depthwise-conv write-back features, so only the stage structure is shared.
- The attention residual stays bridge-side: MCore's `wide_residual`/`streamwise_residual_ops` are
  per-layer streamwise sigmoid residuals with learned scalar factors, and Kimi K3's
  `attn_res_aggregate` is a scalar read over block snapshots; the reference's `g_proj`
  decay/erase/write gates plus `q_proj`/`k_proj` depth routing and the low-rank block bank have
  no native equivalent.
- ROSA and the pipeline stream payload stay bridge-side: no suffix-automaton or stage-payload
  equivalent exists in MCore or in the other bridge models (Kimi K3's `pack_stage_boundary` is a
  simpler prefix+bank concatenation, not a multi-row stream/block/top-k payload).

## Training objectives

The bridge adds the training glue the reference (an inference implementation) does not carry:

| Objective | Where | Weight | Notes |
|-----------|-------|--------|-------|
| next-token cross entropy | Megatron-Core `LanguageModule` on the LM head | 1.0 | the only reported loss; masked by `masked_next_token_loss` |
| router load balancing | Megatron-Core `MoELayer` (`aux_loss`) | `moe_aux_loss_coeff` <- reference `router_aux_loss_coef` (default 1e-3) | enabled whenever the reference weight is non-zero, even if `output_router_logits` is off: a routed MoE otherwise starves experts |
| expert-router coupling (ERC) | `ShensiMoELayer.compute_erc_loss` (one term per shared expert pool) | `erc_loss_coef / erc_num_groups` | `erc_loss_func` matches the reference formula; attached through `MoEAuxLossAutoScaler` |
| indexer KL | `ShensiIndexedAttention._attach_indexer_loss` | `indexer_loss_coeff` (default 0.01) | target is the attention distribution over the selected candidates; attached through `DSAIndexerLossAutoScaler`; `0` removes the indexer graph entirely |
| ROSA next-token CE | `ShensiROSAEmbedding._rosa_token_loss` | `rosa_loss_coef / num_rosa_layers` (default 0.05) | RWKV-v8 "onlyemb" objective with the tied embedding as classifier |
| routed-stream load balancing | `ShensiHyperConnection._attach_router_balance_loss` | `hc_router_aux_loss_coef` (default 0.01) | Switch-style load x gate-probability term so every routed-stream adapter stays reachable |
| MTP | Megatron-Core `MultiTokenPredictionLayer` | `mtp_loss_scaling_factor` (MCore default 0.1) | only when `mtp_num_layers > 0` |

All auxiliary terms except MTP (which is native) attach through Megatron-Core's auxiliary-loss
autoscalers, so the reported LM loss stays the pure next-token loss. With
`calculate_per_token_loss`, the ERC and ROSA terms are pre-multiplied by the token count to match
the token-scaled main loss; the router balancing terms are already token means.

For numerical stability keep the optimizer master weights and moments in fp32 (Megatron-Core's
default) and reduce gradients in fp32 (`grad_reduce_in_fp32=True`): bf16 AdamW state is a known
source of step-2 NaNs on some accelerators. Megatron-Core's `moe_layer_recompute=True` is the
middle ground between no recompute and the full per-iteration checkpoint when only the MoE
activation term needs to shrink.

The two vocabulary-sized auxiliary projections are streamed: the ROSA next-token cross entropy
projects 1024 tokens at a time in the model dtype (measured at the reference vocabulary: peak
4.2 GiB -> 2.7 GiB and 99 -> 82 ms per ROSA layer versus the full-batch fp32 projection), and the
indexer KL target gathers its candidates 512 queries at a time instead of materializing the
whole `[b, heads, s, topk]` tensor.

## Memory levers

At the reference geometry the dominant term is the weight + Adam state (~19.5 B per trainable
parameter: bf16 weights, fp32 master, two momentums, fp32 grad accumulation). The levers, in
order of impact:

- **Distributed optimizer** (`use_distributed_optimizer=True`) shards the optimizer state and
  gradients across the data-parallel group — the 44 GB state at 2.26B parameters becomes
  `44 GB / DP`.
- **Vocabulary-parallel stream table**: the per-stream token table is ~0.5 B parameters at the
  default geometry. It is a `VocabParallelEmbedding`, so TP sharding divides its weights and
  gradient accumulation by TP (it then follows the sequence-parallel reduce-scatter, so the
  lookup keeps working under SP). The bridge's `_StreamTokenEmbeddingMapping` pads the reference
  rows on import and crops them on export, which keeps one checkpoint layout across TP sizes.
- **Pipeline-stage module placement**: `stream_embeddings` is built only on the first stage and
  `hc_head`/`output_attn_res`/`norm` only on the last stage (the decoder stack consumes them at
  those boundaries only). A middle pipeline stage therefore does not carry the ~0.5 B-parameter
  stream table or the output-side modules at all.
- **Activation recompute** (`recompute_granularity="full"` — or `"selective"`, which this stack
  maps to the same per-iteration region because Megatron-Core's selective recompute only wraps
  its own submodules — `recompute_method="block"`, `recompute_num_layers=1`) checkpoints one
  stream-loop iteration at a time. MCore's built-in full recompute cannot drive this stack (each
  layer consumes freshly decoded streams and a re-read attention-residual bank), so the stack
  implements its own per-iteration checkpoint region; a unit test pins that all three settings
  produce identical losses and every gradient.
- **Short-module recompute**: even without the stack-level option, training checkpoints the two
  bridge-owned modules that retain the largest tensors for backward — the hyper-connection
  aggregate (an fp32 copy of the flattened streams) and the attention residual (fp32 gate and
  block tensors). Measured at the reference geometry over 38 calls: attention residual
  14.7 GiB -> 1.3 GiB and 72.6 -> 39.2 ms per call, hyper-connection 4.2 GiB -> 2.3 GiB and
  42.2 -> 3.5 ms per call — recomputing is both smaller and faster than reading the fp32 copies
  back. The visible-block view (committed rows + in-progress prefix) is built inside that
  recompute region, so the concatenated tensor is never retained either.
- **Short-module compile** (`shensi_compile_modules=True`, default off): the same two modules
  run through one shared `torch.compile` graph — the first layer pays the compile cost and the
  rest reuse it, inside the recompute region when that is enabled. Measured per call at the
  small geometry: attention residual 4.5 -> 3.5 ms and 1199 -> 1011 MiB peak; the hyper-connection
  forward/backward traces without graph breaks (the balancing loss rides the Megatron-Core aux
  autoscaler). Leave it off during bring-up so the eager path stays debuggable.
- **Indexer training** is an auxiliary term, not extra state: the HISA indexer trains through a
  KL loss against the attention distribution (`indexer_loss_coeff`, default 0.01) because its
  top-k output is non-differentiable. Set `indexer_loss_coeff=0` to drop the graph the indexer
  keeps for that loss when memory is the binding constraint.
- **Fused cross entropy** (`cross_entropy_loss_fusion=True`, default): the native fused path
  overwrites the logits in place instead of allocating a logits-sized fp32 temporary for the
  CE (the bf16 logits + fp32 CE peak at the reference geometry is ~1.5 GB).
- **Fused attention**: indexed attention runs through
  `F.scaled_dot_product_attention` (with the single-head KV latent broadcast to the query
  heads). Measured at 1k tokens, 64 heads, head dim 128, bf16: the previous explicit
  matmul + fp32 softmax + matmul path peaked at 1091 MiB / 6.3 ms per layer, the fused kernel at
  67 MiB / 1.6 ms. The indexer's KL target is computed from gathered candidates in query chunks,
  so it no longer materializes the full `[b, heads, s, kv]` attention matrix either.
- **Allocator fragmentation**: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the L0
  launcher does) so the varying MoE grouped-GEMM shapes do not strand reserved blocks.

## RL integration

`AutoBridge.get_model_config()` / `get_model()` and the provider path with
`apply_overrides_and_finalize()` both build Shensi, and `export_hf_weights(models,
current_pp_stage_only=True)` streams each pipeline stage's slice for weight refit. The model
rejects packed sequences (`packed_seq_params`) and context parallelism with explicit errors — the
HISA indexer, the attention-residual bank and the ROSA automaton rank over one contiguous sequence
per sample — and `position_ids` defaults to `arange(seq_len)` so logprob-only forwards work.

## Distributed launch

The model has no node-local assumptions: TP/PP groups and the data-parallel group come from
Megatron-Core's `parallel_state`, and rank 0 of the job owns rendezvous and logging.

```bash
# Single machine, single GPU
torchrun --standalone --nproc_per_node=1 <training entrypoint>

# Single machine, N GPUs
torchrun --standalone --nproc_per_node=N <training entrypoint>

# Multi-node: run the same command on every node, with node 0 as MASTER_ADDR.
# NODE_RANK is 0..NODES-1, GPUS_PER_NODE is the local process count.
torchrun --nnodes=$NODES --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE <training entrypoint>
```

Set the model-parallel sizes in the recipe (`tensor_model_parallel_size`,
`pipeline_model_parallel_size`, `data_parallel_size` follows from the world size). Checkpoints
must live on storage visible to every node, or use Megatron-Core's distributed checkpointing.

Each MTP depth mirrors the last decoder layer, including its attention-residual plan: the
decoder stack's post-process step publishes the completed block summaries, and the MTP heads read
them in the same forward, so their attention-residual routing trains over the same block set as
the layer they mirror (without it the routing sees a single candidate and its q/k/scale
parameters never earn a gradient). When that layer reuses a preceding full indexer
(`indexer_types == "shared"`), the MTP head has no predecessor to take top-k indices from, so it
falls back to dense causal attention and keeps the same mirrored parameters.

## Field mapping

`shensi_derived_field_values` maps the HF config onto Megatron-Core: `deepemb_kv_rank →
kv_lora_rank`, `latent_size → moe_latent_size`, `moe_intermediate_size → moe_ffn_hidden_size`,
`linear_num_heads/head_dim → linear_*`, the HISA indexer fields, the per-layer plans
(`layer_types`, `mlp_layer_types`, `indexer_types`, `attn_res_block_layer_types`) and the
hyper-connection/ROSA/ERC fields. `output_router_logits=True` selects Megatron-Core's
`aux_loss` router load balancing (with `router_aux_loss_coef → moe_aux_loss_coeff`); otherwise
the router aux loss stays off, matching HF. Fields without a Megatron-Core counterpart are
listed in `SHENSI_ONLY_FIELDS`.

## Usage

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("<hf checkpoint>")
provider = bridge.to_megatron_provider(load_weights=False)
model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.load_hf_weights(model)
```

## Tests

```sh
uv run python -m pytest -q tests/unit_tests/models/shensi --noconftest --import-mode=importlib
uv run python -m pytest -q tests/functional_tests/test_groups/models/shensi
```

The unit tests pin the attention-residual/bank math against the HF reference, the stream payload
layout and its gradient path, and the ERC gather. The functional tests build a tiny Shensi
checkpoint (`make_tiny_hf_model`) and cover the round trip (HF → Megatron → HF), the forward pass
and the provider/MTP configuration.
