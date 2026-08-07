# Nsight Systems analysis pitfalls

Read this checklist before finalizing a diagnosis.

1. **Summed kernels are not wall time.** Use interval unions across streams and
   clip them to the iteration window.
2. **The longest stream is not step time.** The trainer's slowest-rank step and
   the iteration anchor define the wall clock.
3. **Rank 0 is not automatically representative.** PP stages, expert groups,
   and stragglers can have different work.
4. **Different rank fingerprints are not comparable implementations.** Compare
   like model parts or explain the structural difference.
5. **Distributed communication is not necessarily NCCL-only.** HybridEP,
   DeepEP, NVSHMEM, and other backends may use their own dispatch, combine,
   RDMA, and synchronization kernels.
6. **Dispatcher time is not pure network transfer.** It can contain packing,
   routing, synchronization, control work, arrival jitter, and device-side
   waiting. Report its components and union without relabeling the union as
   network time.
7. **Exposed is not blocking.** A collective or copy may occupy an otherwise
   empty interval without delaying the next compute kernel.
8. **An occupant is not necessarily the producer.** Attribute a gap only after
   resolving the dependency followed by the waiting compute kernel.
9. **CUDA event handles are reused.** Join on `eventSyncId`, not `eventId`.
10. **CUDA event timestamps may be zero.** Reconstruct completion from the last
   device operation before the event record on its stream.
11. **Host synchronization duration is not GPU stall.** Intersect host sync
    intervals with device-idle time.
12. **A copy needs engine evidence.** Check achieved bandwidth, occupancy,
    direction, pinned memory, stream, and blocking dependency.
13. **Default-stream async copies may synchronize.** Verify the actual stream
    and pinned/pageable kinds.
14. **Metric samples are device-wide.** Report exclusivity and sample count;
    do not attach a separate-pass sample to an individual timing-pass launch.
15. **SMs Active is occupancy context.** Use SM Issue and Tensor Active for
    compute-throughput evidence.
16. **NVLink request metrics understate payload traffic.** Use response
    user-data throughput and make no peer claim from device-aggregate metrics.
17. **Metric ids are unstable.** Resolve by metric name in each report.
18. **Unsampled is not zero.** Short operators may receive no samples.
19. **Profiler permission failure is not workload failure.** Continue timing
    analysis when GPU metrics are unavailable.
20. **Regex taxonomy is fallible.** Inspect material matched and unclassified
    names; preserve fused operators rather than forcing them into one primitive.
21. **CUDA graphs break launch assumptions.** Graph kernels may have no
    individual CUDA API row or call-stack link.
22. **Runtime correlation is not source attribution.** A kernel can correlate
    perfectly with its CUDA Runtime API launch while having no Python or
    framework call-stack link. Report the two rates separately.
23. **Cross-host clocks are not automatically aligned.** Rebase capture starts,
    refine against many matched collective ends, and report residual
    uncertainty. A longer duration can be earlier start skew rather than late
    completion.
24. **A sampled-rank verdict is not a global verdict.** Two representative
    ranks can agree while an uncaptured participant is the true straggler.
25. **Capture overhead can change the workload.** Keep the ROI short, compare
    same-run profiled and unprofiled step time, and separate metrics from
    timing. The `PROFILER_OVERHEAD` table is not end-to-end slowdown.
26. **Profiler boundary steps are not steady state.** Start/stop and buffer
    flush can inflate a logged step outside the selected device ROI.
27. **Capture warnings constrain the claim.** Missing CUDA, NVTX, OS-runtime,
    scheduling, metric, or hardware-trace data must appear in capture quality.
28. **Dataloading scope changes the conclusion.** Match the user's step
    definition before attributing GPU idle to training code.
29. **MFU numerator errors survive perfect timing analysis.** Recheck the model-
    specific FLOP formula and precision denominator separately.
30. **One failed implementation does not disprove a technique.** Classify the
    failure as mechanism, implementation, environment, or verification-gate
    failure and retain the technique's measured ceiling.
31. **Opportunity ceilings do not automatically add.** Add them only for
    disjoint intervals and independent fixes.
32. **Trace phenomena are not source root causes.** Require a source/config
    mechanism and actionable code anchor, or mark the claim inferred.
