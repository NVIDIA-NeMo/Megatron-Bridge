# Shensi

Shensi is a hybrid sparse-attention Mixture-of-Experts language model. Each decoder layer runs
either indexed attention (DeepSeek-style sparse attention with a HISA indexer) or gated linear
attention (Kimi Delta Attention); each feed-forward branch is a dense ReLU² MLP with a
token-conditioned DeepEmbed scale or a latent MoE. Every sublayer is wrapped in
hyper-connection streams, layers group into attention-residual blocks whose low-rank bank is
carried with the running residual, and ROSA suffix-automaton predictions are injected at
selected layers.

## Supported Variants

| Variant | Notes |
|---------|-------|
| Shensi | Indexed / linear attention plan, hyper-connection streams, attention-residual bank, latent MoE |

## Architecture Notes

- The per-layer attention plan is derived from the attention-residual write schedule: write
  layers run indexed attention, read layers run linear attention. Indexed layers use a low-rank
  KV latent and a HISA indexer that ranks mean-pooled blocks before refining tokens; full
  indexers run on every indexed layer; `indexer_types` can mark a layer `"shared"` so it reuses
  the previous full indexer's top-k. The
  top-k selection is non-differentiable, so the indexer trains through a KL auxiliary loss
  against the attention distribution over the same candidates (`indexer_loss_coeff`, DeepSeek-V3
  lightning-indexer style); the auxiliary term changes gradients only, not the reported loss.
- Hyper-connection streams are expanded from the embeddings, collapsed with a single softmax
  `pre` projection, kept in a low-rank basis between layers, and compressed/expanded at pipeline
  stage boundaries. The fixed streams are the residual anchors; routed streams can carry learned
  decay and low-rank adapters when `hc_active_streams < hc_mult`.
- Attention-residual blocks keep a low-rank bank of completed block deltas. Every sublayer
  applies channel-wise decay/erase/write gates and low-rank depth routing over the bank, gated by
  a ReZero scalar; only the depth correction is written into the hyper-connection streams.
- The leading layers are dense; the remaining layers use a latent MoE that routes in
  `latent_size` and normalizes before projecting back. Router and expert pools are shared per
  attention-residual block, and the ERC loss couples the router with the expert weights (attached
  once per pool through Megatron-Core's auxiliary-loss autoscaler). `output_router_logits=True`
  maps to the router load-balancing auxiliary loss.
- ROSA predictions come from a parameterless suffix automaton; the learned ROSA
  embedding/value/gate are trained by the LM loss and, by default (`rosa_loss_coef = 0.05`), by
  an auxiliary next-token cross-entropy through the tied token embedding (the RWKV-v8 "onlyemb"
  objective). The automaton stays hard (no STE); `rosa_soft`/SUFA is a separate operator.
- MTP depths mirror the last decoder layer; when that layer reuses a shared indexer, the MTP head
  falls back to dense causal attention.
- TP > 1 enables sequence parallelism (required by Megatron-Core's MoE training); the stream
  convolutions gather the full sequence for their causal window. PP runs with
  `variable_seq_lengths` and the `alltoall` token dispatcher. Context parallelism is rejected
  because the attention-residual bank and HISA indexer rank over the full sequence.
- A shared indexer may cross a pipeline-stage boundary: the full indexer's top-k indices travel
  in the stage payload, so arbitrary PP splits keep the same selected tokens.
- The ROSA token table is tied to the input embedding (or the shared output table on the last
  stage) through Megatron-Core's `VocabParallelEmbedding`, so tied ROSA layers live on the first
  or last PP stage; middle stages raise an error.
- The model rides `HybridModel` through two equivalent paths: `ShensiModelProvider` (provider)
  and `ShensiModelConfig` + `ShensiModelBuilder` (model-config / model-builder).
- Native MCore pieces are reused where the semantics match: `TransformerLayer`, native
  `MultiTokenPredictionLayer`, `KimiDeltaAttention`, latent MoE with
  `moe_use_norm_before_up_proj`, and the backend norm/linear factories (`TENorm`, TE
  `parallel_mode="duplicated"`), which makes Transformer Engine a hard requirement.
  HISA indexed attention, hyper-connections, the attention-residual bank, ROSA, DeepEmbed and the
  pipeline stream payload remain bridge-side because MCore's DSA kpool, mHC and AttnRes
  aggregate use different parameterizations.

## Distributed launch

Use `torchrun` with one process per GPU. For a single machine use
`--standalone --nproc_per_node=<gpus>`; for multiple machines run the same command on every node
with `--nnodes=<nodes> --node_rank=<rank> --master_addr=<node0> --master_port=<port>` and
`--nproc_per_node=<gpus-per-node>`. TP/PP/DP sizes come from the recipe, and checkpoints must be
on shared storage (or use distributed checkpointing).

## Training and RL integration

Checkpoint import/export and the model-construction entry points follow the standard bridge
flow, so an RL framework can drive Shensi without model-specific glue:

```python
from megatron.bridge import AutoBridge

# HF -> Megatron checkpoint (one call), then build the training config.
bridge = AutoBridge.from_hf_pretrained("<hf checkpoint>", trust_remote_code=True)
model_config = bridge.get_model_config()            # ShensiModelConfig (builder path)
models = bridge.get_model(wrap_with_ddp=False)      # one ShensiModel per pipeline stage

# Or the provider path with validated overrides:
provider = bridge.to_megatron_provider(load_weights=False)
provider.apply_overrides_and_finalize(
    dtype=torch.bfloat16,
    overrides={"tensor_model_parallel_size": 2, "recompute_granularity": "full",
               "recompute_method": "block", "recompute_num_layers": 1},
)
```

Refit back to Hugging Face for inference workers:

```python
for name, weight in bridge.export_hf_weights(models, cpu=True, current_pp_stage_only=True):
    send_to_inference_workers(name, weight)
```

`current_pp_stage_only=True` streams each pipeline stage's slice on that stage only; combining one
representative stream from every PP stage reconstructs the complete export.

**Limits to know when wiring an RL loop:**

- Shensi requires Transformer Engine (`finalize()` rejects a non-TE backend) and rejects context
  parallelism: the HISA indexer, the attention-residual bank and the ROSA automaton rank over one
  contiguous sequence per sample.
- Packed sequences (`packed_seq_params`, THD layout) are rejected with an explicit error rather
  than silently mis-ranking across segments; pass a right-padded 2D attention mask instead.
- `position_ids` is optional (defaults to `arange(seq_len)`), so logprob-only forwards can pass
  `input_ids` alone. Output logits are `[batch, seq, vocab]`.
- With `recompute_granularity="full"`, run the recompute leg without the distributed optimizer and
  APEX fused wgrad: their grad-buffer bookkeeping assumes one backward per forward.
- The per-stream token table is vocabulary-parallel and uses Megatron's padded vocabulary layout;
  checkpoint conversion pads the reference rows on import and crops them on export. Boundary
  modules are built only on the pipeline stages that consume them, so middle stages do not carry
  the stream table or the output-side modules.

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/shensi`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/shensi)
- Implementation notes (layout, knobs, validation): [`src/megatron/bridge/models/shensi/README.md`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/shensi/README.md)
