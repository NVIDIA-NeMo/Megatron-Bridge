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
5. **NCCL duration is not pure network transfer.** It can include arrival
   jitter and device-side waiting.
6. **Exposed is not blocking.** A collective or copy may occupy an otherwise
   empty interval without delaying the next compute kernel.
7. **An occupant is not necessarily the producer.** Attribute a gap only after
   resolving the dependency followed by the waiting compute kernel.
8. **CUDA event handles are reused.** Join on `eventSyncId`, not `eventId`.
9. **CUDA event timestamps may be zero.** Reconstruct completion from the last
   device operation before the event record on its stream.
10. **Host synchronization duration is not GPU stall.** Intersect host sync
    intervals with device-idle time.
11. **A copy needs engine evidence.** Check achieved bandwidth, occupancy,
    direction, pinned memory, stream, and blocking dependency.
12. **Default-stream async copies may synchronize.** Verify the actual stream
    and pinned/pageable kinds.
13. **Metric samples are device-wide.** Report exclusivity and sample count;
    do not attach a separate-pass sample to an individual timing-pass launch.
14. **SMs Active is occupancy context.** Use SM Issue and Tensor Active for
    compute-throughput evidence.
15. **NVLink request metrics understate payload traffic.** Use response
    user-data throughput and make no peer claim from device-aggregate metrics.
16. **Metric ids are unstable.** Resolve by metric name in each report.
17. **Unsampled is not zero.** Short operators may receive no samples.
18. **Profiler permission failure is not workload failure.** Continue timing
    analysis when GPU metrics are unavailable.
19. **Regex taxonomy is fallible.** Inspect material matched and unclassified
    names; preserve fused operators rather than forcing them into one primitive.
20. **CUDA graphs break launch assumptions.** Graph kernels may have no
    individual CUDA API row or call-stack link.
21. **Cross-host clocks are not automatically aligned.** Rebase capture starts,
    refine against collective ends, and report residual uncertainty.
22. **Capture overhead can change the workload.** Keep the ROI short, compare
    profiled and unprofiled step time, and separate metrics from timing.
23. **Dataloading scope changes the conclusion.** Match the user's step
    definition before attributing GPU idle to training code.
24. **MFU numerator errors survive perfect timing analysis.** Recheck the model-
    specific FLOP formula and precision denominator separately.
25. **One failed implementation does not disprove a technique.** Classify the
    failure as mechanism, implementation, environment, or verification-gate
    failure and retain the technique's measured ceiling.
26. **Opportunity ceilings do not automatically add.** Add them only for
    disjoint intervals and independent fixes.
27. **Trace phenomena are not source root causes.** Require a source/config
    mechanism and actionable code anchor, or mark the claim inferred.
