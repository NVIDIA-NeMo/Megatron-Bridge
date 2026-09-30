# Known Issues

Known issues are maintained in the [GitHub release notes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases).
See the **Known Issues** section for your version, or check the
[latest release notes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/latest).

## 26.08

- **SPCX payload corruption at high expert parallelism:** In the affected NCCL 2.30.5/SPCX stack, stale or uninitialized `ncclIbRequest.send.onlyWriteImm` state can misclassify grouped variable-split BF16 all-to-all sends, losing or corrupting payloads. MCore's MoE `alltoall` dispatcher can trigger this transport issue at high EP. Disable the external network plugin and select NCCL's internal IB transport with `NCCL_NET_PLUGIN=none NCCL_NET=IB`, or use HybridEP. See [issue #5462](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5462).
- **Step-3.7-Flash checkpoint parity remains unverified.**
- **Large-checkpoint round-trip validation may time out:** All ranks export, but rank 0 then lazily loads and exhaustively compares the original Hugging Face tensors. Peers can reach synchronization early enough to exceed the default 600-second NCCL process-group timeout while rank 0 is still progressing. Increase `--distributed-timeout-minutes` beyond the worst-case verification duration. If exhaustive validation is impractical, run separate `import` and `export --distributed-save` workflows instead of `roundtrip`. See [PR #5491](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5491).
