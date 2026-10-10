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
  selects keys block-first (mean-pooled blocks) and then token-first inside the retained blocks;
  the query's own key is forced into the selection (XSA removes its value direction afterwards),
  and the per-head sigmoid gate scales the output. Full indexers recur every
  `index_topk_freq` indexed layers; the intervening indexed layers reuse the previous top-k.
  Training the indexer goes through a KL auxiliary loss against the attention distribution over
  the selected candidates (`indexer_loss_coeff`, default 0.01, DeepSeek-V3 lightning-indexer
  style): the selection is non-differentiable, the scores are not.
* **linear attention** — Kimi Delta Attention (KDA) from Megatron-Core, with two-stage low-rank
  decay/output gates, a short convolution and a gated output norm.

Around the sublayers:

* **hyper-connections** (`ShensiHyperConnection`) — every sublayer collapses a bank of
  `hc_mult` residual streams with a single `pre` projection + softmax, writes the sublayer output
  back with per-stream `post` weights and optional depthwise-conv features, and keeps the block
  residual on the fixed streams only (the residual anchors). When `hc_active_streams < hc_mult`
  the routed streams also carry per-stream learned decay and a low-rank `ShensiRoutedAdapter`;
  the default (`hc_active_streams == hc_mult`) runs the unconditioned identity-4 form.
  `ShensiStreamExpansion` creates the streams from the embeddings and `ShensiStreamCompression`
  stores them in a low-rank basis between layers.
* **attention residuals** (`ShensiAttentionResidual`) — channel-wise decay/erase/write gates from
  a low-rank `gate_down`/`gate_up` pair, low-rank depth routing (`q_proj`/`k_proj`) over the
  completed block deltas, and a ReZero `residual_gate` that interpolates between the plain
  residual and the routed branch. Only the depth correction is written into the hyper-connection
  streams; the block prefix stays in the bank (`ShensiAttnResBank`, low-rank basis, exact token
  embeddings in row 0).
* **ROSA** (`ShensiROSALayer`) — suffix-automaton next-token predictions computed once per
  forward and injected into every stream at the configured layers. The automaton itself is
  parameterless; its embedding/value/gate train through the LM loss and, with
  `rosa_loss_coef > 0` (default 0, matching the HuggingFace loss; set it to follow the
  RWKV-v8 ROSA training recipe), through an auxiliary next-token cross-entropy on the gated
  ROSA feature using the tied token embedding as classifier (the RWKV-v8 "onlyemb" objective).
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
* **selective fp32 keeping** (`fp32_keep`) — the checkpoint's numerically sensitive parameters
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

The ROSA token table is tied to the input embedding; on the last stage it reuses the shared output
table. Both go through Megatron-Core's `VocabParallelEmbedding` for the vocab-parallel lookup. A
tied ROSA layer therefore lives on the first or last pipeline stage (a middle stage
raises a clear error). Setting `rosa_embed_dim != hidden_size` uses untied local tables instead.

A shared indexer may sit on the pipeline stage after its full indexer: the full indexer's top-k
indices ride the payload as byte-exact rows, so arbitrary pipeline splits keep the same selected
tokens.

## Native reuse

The implementation is rebased on upstream Megatron-Bridge and tracks the latest Megatron-Core.

- The container is MCore's `HybridModel`; the layer is a `TransformerLayer`; MTP is MCore's native
  `MultiTokenPredictionLayer` and `get_gpt_mtp_block_spec`.
- KDA is MCore's `KimiDeltaAttention`; dense/MoE linears come from
  `get_backend_from_config` (TE when installed, local otherwise).
- Latent MoE is MCore's `MoELayer` with `moe_latent_size` and
  `moe_use_norm_before_up_proj`; only the ERC loss remains bridge-side.
- The dense branch is MCore's `MLP` (fused gate/up `linear_fc1`, `linear_fc2`, and its gated
  clamp path) subclassed only to add the previous-token shift and the DeepEmbed scale, which the
  checkpoint's separate `gate_proj`/`up_proj` tensors reach through `GatedMLPMapping`.
- Linears and norms come straight from Megatron-Core's backend factories:
  `get_backend_from_config(config).linear()` (TE `parallel_mode="duplicated"`),
  `.column_parallel_linear()`, `.row_parallel_linear()` and `.layer_norm(rms_norm=True)`
  (`TENorm`). Shensi therefore requires Transformer Engine — the local backend has no
  replicated linear and its torch norm asserts sequence parallelism off — and `finalize()`
  refuses a non-TE backend with a clear error.
- Both auxiliary loss terms use Megatron-Core's own autoscalers: the indexer KL goes through
  `DSAIndexerLossAutoScaler` (the same one MCore's DSA indexer uses, scaled by the schedule)
  and the ERC/ROSA terms through `MoEAuxLossAutoScaler`.
- The HISA indexer follows MCore's `DSAIndexer` module shape (`ShensiIndexerSubmodules`,
  `forward_before_topk`/`forward`, `linear_wq_b`/`linear_wk`/`linear_weights_proj` naming,
  replicated-projection gradient averaging); only the selection (mean-pooled blocks, then
  tokens) and the reference's low-rank cached keys / shared token table replace kpool+RoPE.
- Hyper-connections follow MCore's `HyperConnectionModule` decomposition
  (`aggregate` / `apply_h_post` / `apply_h_res` / `forward`) with the reference `pre`/`post`/
  `route` parameterization instead of the sinkhorn-projected mapping.
- The attention-residual bank, ROSA, the DeepEmbed/token-shift MLP and the pipeline stream
  payload stay bridge-side: MCore ships only the `ResidualConnection` interface, and no
  suffix-automaton or stage-payload equivalent exists.

## Memory levers

At the reference geometry the dominant term is the weight + Adam state (~19.5 B per trainable
parameter: bf16 weights, fp32 master, two momentums, fp32 grad accumulation). The levers, in
order of impact:

- **Distributed optimizer** (`use_distributed_optimizer=True`) shards the optimizer state and
  gradients across the data-parallel group — the 44 GB state at 2.26B parameters becomes
  `44 GB / DP`.
- **Activation recompute** (`recompute_granularity="full"`, `recompute_method="block"`,
  `recompute_num_layers=1`) checkpoints one stream-loop iteration at a time, dropping the
  per-layer activations (~2 GB at the reference geometry) at a ~30% recompute cost. MCore's
  built-in full recompute cannot drive this stack (each layer consumes freshly decoded streams
  and a re-read attention-residual bank), so the stack implements its own per-iteration
  checkpoint region; a unit test pins that losses and every gradient match the eager run.
  Run this leg without the distributed optimizer and APEX fused wgrad: their grad-buffer
  bookkeeping assumes one backward per forward.
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
