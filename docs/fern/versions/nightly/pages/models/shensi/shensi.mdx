# Shensi

Shensi is a sparse-attention Mixture-of-Experts language model. Every decoder layer picks one of
three attention paths from a per-layer plan — compressed sparse attention (CSA, compression ratio
4), heavily compressed attention (HCA, 128) or a sliding window — and the residual stream is not a
single tensor but a pool of streams that the sublayers read from and write back into. Layers group
into attention-residual blocks whose state may cross pipeline stages, the leading layers are
token-embedding-gated dense MLPs, and routed experts use a low-rank bottleneck with an ERC
auxiliary loss. The hybrid attention rides Megatron-Core's `dsv4_hybrid` path.

## Supported Variants

| Variant | Notes |
|---------|-------|
| Shensi | CSA/HCA/sliding-window attention plan, mHC streams, attention-residual blocks, hash-routed leading layers |

## Architecture Notes

- The per-layer attention plan is driven by compression ratios (0 = sliding window, 4 = CSA,
  128 = HCA); the DSA lightning indexer sits on the CSA layers.
- Residual streams are multi-stream hyper-connections (mHC); attention-residual blocks read and
  write them, and block state is handed across pipeline stages when a block spans stages.
- Routed experts are low-rank (`routed_expert_hidden_size`). The leading layers use hash routing
  (token-id based), which takes no load-balancing loss; the remaining layers use the learned router.
- The model rides `HybridModel` through two equivalent paths: `ShensiModelProvider` (provider) and
  `ShensiModelConfig` + `ShensiModelBuilder` (model-config / model-builder).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/shensi`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/shensi)
- Implementation notes (layout, knobs, validation): [`src/megatron/bridge/models/shensi/README.md`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/shensi/README.md)
