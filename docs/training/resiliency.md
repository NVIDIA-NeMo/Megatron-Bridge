# Resiliency

Megatron Bridge incorporates resilient training features from the
[NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext).
This extension provides fault-tolerant capabilities that help minimize downtime
due to failures and interruptions during training.

This page is the stable overview for what each resiliency feature is, when to
use it, and which constraints are durable. For operational setup, config knobs,
parameter tables, code anchors, and verification commands, see
`skills/resiliency/SKILL.md`.

## Feature Overview

| Feature | Purpose | Maturity | Cluster |
|---|---|---|---|
| Fault tolerance | Hang detection + automatic job restart | Production | Slurm only |
| NVRx straggler detection | Identify slow GPUs | Production | Any |
| Preemption | Graceful shutdown before time limit | Production | Slurm only |
| Async checkpoint save | Non-blocking checkpoint writes | Production | Any |
| Local checkpointing | Fast local save with replication | Production | Any |
| Re-run state machine | NaN / spiky loss attribution | Experimental | Any |
| In-process restart | Restart within the same process | Experimental | Any |

## Fault Tolerance

The fault tolerance feature detects hangs during training and automatically
restarts the workload. It uses section-based monitoring with different timeout
thresholds for setup, training steps, and checkpointing operations.

### When to Use It

Fault tolerance is a good fit when:

- training on unreliable hardware or at very large scale
- transient faults (network glitches, GPU errors) are common
- you want automatic recovery without manual intervention

### Stable Constraints

- Requires Slurm and `ft_launcher` (not `torchrun`)
- Checkpoint directory must be configured and accessible
- Uses `nvidia-resiliency-ext` RankMonitorClient
- Not compatible with NSys profiling

The system supports both in-job restarts (within the same Slurm allocation) and
new job launches on failure, with configurable limits for each.

## Straggler Detection

NVRx straggler detection monitors GPU performance across ranks and identifies
slow-performing nodes. It calculates both relative and individual performance
scores, and can optionally terminate training if performance falls below
configurable thresholds.

### When to Use It

Straggler detection is useful when:

- training at scale where one slow node degrades overall throughput
- you want visibility into per-rank GPU performance
- you need to identify persistent hardware issues

### Stable Constraints

- Requires `nvidia-resiliency-ext`
- Overhead is minimal but can be tuned via `profiling_interval`
- Does **not** stop training by default; `stop_if_detected` must be
  explicitly set to `True` for automatic termination

## Preemption

Preemption handling provides graceful shutdown when a training job receives a
termination signal (default: SIGTERM). It saves a checkpoint before exiting to
preserve training progress.

### When to Use It

Preemption is important when:

- running on shared clusters with job time limits
- higher-priority jobs may preempt your allocation
- you want to minimize lost work on job termination

### Stable Constraints

- The `PreemptionPlugin` is Slurm-specific
- Direct configuration via `exit_signal_handler` works on any cluster
- Signal detection happens at the end of each training step

## Async Checkpoint Save

Async checkpoint save overlaps checkpoint I/O with training compute using
persistent background workers. Training continues immediately after scheduling
the save rather than blocking until the write completes.

### When to Use It

Async save is valuable when:

- checkpoint save time is a significant fraction of step time
- you are using `torch_dist` checkpoint format

### Stable Constraints

- Requires `ckpt_format="torch_dist"`
- Other formats (zarr, fsdp_dtensor) do not support async save
- The persistent checkpoint worker must be enabled

## Local Checkpointing

Local checkpointing saves checkpoint data to node-local storage first, then
replicates across a configurable number of nodes. This avoids the latency of
writing to shared network storage during the critical path.

### When to Use It

Local checkpointing is useful when:

- shared-storage checkpoint writes are the bottleneck in your checkpoint interval
- you want faster recovery from node failures without depending on network filesystem availability
- training at scale where network-storage contention is common

### Stable Constraints

- Node-local storage must have sufficient capacity for at least one checkpoint
- Replication degree must be configured to survive the expected failure rate
- Requires compatible checkpoint format (see `skills/resiliency/SKILL.md`)

## Re-run State Machine

The re-run state machine is an experimental feature for attributing unexpected
results (NaN loss, spiky loss) to transient errors, persistent hardware faults,
or correct-but-unexpected results. It works by re-running computations on the
same and different GPUs.

### When to Use It

Consider the re-run state machine when:

- you need automated NaN detection and attribution
- you want to distinguish hardware faults from training instability

### Stable Constraints

- Alpha-level feature; full integration is limited
- Three modes: `disabled`, `validate_results`, `report_determinism_stats`
- Uses specific exit codes (16, 17) to control job behavior

## In-Process Restart

In-process restart provides automatic fault recovery by restarting the training
function within the same OS process. This avoids the overhead of launching new
jobs, starting containers, and creating new CUDA contexts.

### When to Use It

In-process restart is appropriate when:

- software faults (exceptions, deadlocks) are more common than hardware faults
- restart latency matters and you want to avoid full job relaunch
- you can accept the experimental status and compatibility constraints

### Stable Constraints

- Requires PyTorch >= 2.5.1 and NCCL >= 2.26.2
- Not compatible with NeMo-Run or Slurm preemption plugins
- Requires specific environment variables (`NCCL_NVLS_ENABLE=0`, etc.)
- The PyTorch NCCL watchdog timeout must exceed `hard_timeout`
- Supports both node-level and rank-level restart granularity

In-process restart is not suitable for hardware-level failures such as switch
failures or network partitions. For comprehensive fault tolerance, combine it
with job-level fault tolerance.

## Practical Caveats

1. No single resiliency feature covers all failure modes. The recommended
   approach is to layer features (e.g., fault tolerance + straggler detection +
   async checkpoint).
2. Not all recipes enable resiliency features by default. Check and enable
   explicitly.
3. Two straggler detectors exist in the codebase (NVRx and legacy MCore).
   Use the NVRx version; do not enable both.

## Related Docs

- `docs/training/checkpointing.md`
- `docs/performance-guide.md`
- `skills/resiliency/SKILL.md`
- `skills/resiliency/card.yaml`
- [NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext)
- [In-Process Restart Guide](https://nvidia.github.io/nvidia-resiliency-ext/inprocess/index.html)
