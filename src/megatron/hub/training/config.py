# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import signal
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Optional, Union

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.hub.core.utils.common_utils import get_world_size_safe
from megatron.hub.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.hub.models import GPTModelProvider, T5ModelProvider
from megatron.hub.training.tokenizers.config import TokenizerConfig
from megatron.hub.training.utils.config_utils import _ConfigContainerBase as Container


@dataclass(kw_only=True)
class RNGConfig:
    """Configuration settings for random number generation."""

    seed: int = 1234
    """Random seed used for python, numpy, pytorch, and cuda."""

    te_rng_tracker: bool = False
    """Use the Transformer Engine version of the random number generator.
    Required for CUDA graphs support."""

    inference_rng_tracker: bool = False
    """Use a random number generator configured for inference."""

    data_parallel_random_init: bool = False
    """Enable random initialization of params across data parallel ranks"""


@dataclass(kw_only=True)
class DistributedInitConfig:
    """Configuration settings for distributed training initialization."""

    # ---------------- Distributed config. ----------------

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    """Which backend to use for distributed training."""

    distributed_timeout_minutes: int = 10
    """Timeout minutes for torch.distributed."""

    align_grad_reduce: bool = True
    """If not set, all PP stages will launch gradient reduces simultaneously.
    Otherwise, each PP stage will independently launch as needed.
    """

    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    """local rank passed from distributed launcher."""

    lazy_init: bool = False
    """If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead.
    Also turns on --use-cpu-initialization flag. This is for external DDP manager."""

    use_torch_fsdp2: bool = False
    """Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel.
    It is still not in a stable release stage, and may therefore contain bugs or other
    potential issues."""

    nccl_communicator_config_path: Optional[str] = None
    """Path to the yaml file with NCCL communicator configurations. The number of min/max thread
    groups and thread group cluster size of each communicator can be configured by setting
    `min_ctas`, `max_ctas`, and `cga_cluster_size`."""

    use_tp_pp_dp_mapping: bool = False
    """If set, distributed ranks initialize order is changed from tp-dp-pp to tp-pp-dp.
    Make sure EP and CP aren't used with this option enabled.
    """

    use_gloo_process_groups: bool = True
    """If set, create Gloo process groups for communications."""


@dataclass
class RerunStateMachineConfig:
    """Configuration for the rerun state machine used for result validation or stats."""

    error_injection_rate: int = 0
    """Rate at which to inject unexpected results, e.g. 1000 means
    once every 1000 result validations"""

    error_injection_type: Literal["correct_result", "transient_error", "persistent_error"] = "transient_error"
    """Type of error to inject. """

    rerun_mode: Literal["disabled", "validate_results", "report_stats"] = "disabled"
    """Use re-run engine to validate results (default) or to emit stats
    on variability of computations due to non-deterministic algorithms."""


@dataclass(kw_only=True)
class DataloaderConfig:
    """Base configuration for data loading."""

    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = None
    """Single pass vs multiple pass data loader"""

    num_workers: int = 8
    """Dataloader number of workers."""

    data_sharding: bool = True
    """Disable data sharding."""

    pin_memory: bool = True
    """Whether to pin memory during data loading for faster GPU training."""

    persistent_workers: bool = False
    """Whether to keep data loading workers persistent across epochs."""


@dataclass
class GPTDatasetConfig(MCoreGPTDatasetConfig, DataloaderConfig):
    """Configuration specific to GPT datasets, inheriting from MCore and base DataloaderConfig."""

    def __post_init__(self) -> None:
        """Post-initialization checks for GPT dataset config."""
        super(MCoreGPTDatasetConfig, self).__post_init__()

        assert self.reset_position_ids is not None, "reset_position_ids must be defined."
        assert self.reset_attention_mask is not None, "reset_attention_mask must be defined."
        assert self.eod_mask_loss is not None, "eod_mask_loss must be defined."


@dataclass
class MockGPTDatasetConfig(GPTDatasetConfig):
    """Modifies GPTDatasetConfig to enforce necessary options for creating a mock dataset."""

    blend: None = field(init=False, repr=False, default=None)
    blend_per_split: None = field(init=False, repr=False, default=None)


@dataclass(kw_only=True)
class FinetuningDatasetConfig(DataloaderConfig):
    """Configuration specific to finetuning datasets, inheriting from DataloaderConfig."""

    dataset_root: Optional[Union[str, Path]] = None
    seq_length: int
    seed: int = 1234
    memmap_workers: int = 1
    max_train_samples: Optional[int] = None
    packed_sequence_specs: Optional[PackedSequenceSpecs] = None
    dataset_kwargs: Optional[dict[str, Any]] = None
    do_validation: bool = True
    do_test: bool = True


@dataclass(kw_only=True)
class SchedulerConfig:
    """Configuration settings for the learning rate scheduler and weight decay."""

    # ---------------- Learning rate config. ----------------
    lr_decay_style: Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"] = "linear"
    """Learning rate decay function."""

    lr_wsd_decay_style: Literal["exponential", "linear", "cosine"] = "exponential"
    """Decay style for the annealing phase of WSD"""

    lr_decay_iters: Optional[int] = None
    """number of iterations to decay learning rate over, If None defaults to `--train-iters`"""

    lr_wsd_decay_iters: Optional[int] = None
    """number of iterations for the annealing phase in the wsd schedule"""

    lr_warmup_fraction: Optional[float] = None
    """fraction of lr-warmup-(iters/samples) to use for warmup (as a float)"""

    lr_warmup_iters: int = 0
    """number of iterations to linearly warmup learning rate over."""

    lr_warmup_init: float = 0.0
    """Initial value for learning rate warmup. The scheduler starts warmup from this value."""

    override_opt_param_scheduler: bool = False
    """Reset the values of the scheduler (learning rate, warmup iterations, minimum learning rate,
    maximum number of iterations, and decay style from input arguments and ignore values from
    checkpoints. Note that all the above values will be reset."""

    use_checkpoint_opt_param_scheduler: bool = False
    """Use checkpoint to set the values of the scheduler (learning rate, warmup iterations,
    minimum learning rate, maximum number of iterations, and decay style from checkpoint
    and ignore input arguments."""

    # ---------------- Regularization config. ----------------

    start_weight_decay: Optional[float] = None
    """Initial weight decay coefficient for L2 regularization."""

    end_weight_decay: Optional[float] = None
    """End of run weight decay coefficient for L2 regularization."""

    weight_decay_incr_style: Literal["constant", "linear", "cosine"] = "constant"
    """Weight decay increment function."""

    lr_warmup_steps: Optional[int] = field(init=False, default=None)
    lr_decay_steps: Optional[int] = field(init=False, default=None)
    wd_incr_steps: Optional[int] = field(init=False, default=None)
    wsd_decay_steps: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        """Post-initialization checks for scheduler config."""
        if self.start_weight_decay is not None:
            assert self.start_weight_decay >= 0.0, "start_weight_decay should be positive."
            assert self.end_weight_decay >= self.start_weight_decay

        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, "both override and use-checkpoint are set."


@dataclass(kw_only=True)
class TrainingConfig:
    """Configuration settings related to the training loop and validation."""

    # ---------------- Training config. ----------------

    micro_batch_size: Optional[int] = None
    """Batch size per model instance (local batch size). Global batch size is local batch size times
    data parallel size times number of micro batches."""

    global_batch_size: Optional[int] = None
    """Training batch size. If set, it should be a multiple of micro-batch-size times
    data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size
    as the global batch size. This choice will result in 1 for number of micro-batches."""

    rampup_batch_size: Optional[list[int]] = None
    """Batch size ramp up with the following values: <start batch size>, <batch size increment>,
    <ramp-up samples>
    For example:
        rampup-batch-size = [16, 8, 300000]
        global-batch-size 1024
    will start with global batch size 16 and over (1024 - 16) / 8 = 126 intervals will increase
    the batch size linearly to 1024. In each interval we will use approximately
    300000 / 126 = 2380 samples.
    """

    decrease_batch_size_if_needed: bool = False
    """If set, decrease batch size if microbatch_size * dp_size does not divide batch_size.
    Useful for KSO (Keep Soldiering On) to continue making progress if number of healthy GPUs
    (and corresponding dp_size) does not support current batch_size. Old batch_size will be
    restored if training is re-started with dp_size that divides batch_size // microbatch_size."""

    empty_unused_memory_level: Literal[0, 1, 2] = 0
    """Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation.
    0=off, 1=moderate, 2=aggressive.
    """

    check_weight_hash_across_dp_replicas_interval: Optional[int] = None
    """Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked."""

    train_sync_interval: Optional[int] = None
    """Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU."""

    train_iters: Optional[int] = None
    """Total number of iterations to train over all training runs.
    Note that either train-iters or train-samples should be provided.
    """

    exit_interval: Optional[int] = None
    """Exit the program after the iteration is divisible by this value."""

    exit_duration_in_mins: Optional[int] = None
    """Exit the program after this many minutes."""

    exit_signal_handler: bool = False
    """Dynamically save the checkpoint and shutdown the training if SIGTERM is received"""

    exit_signal: int = signal.SIGTERM
    """Signal for the signal handler to detect."""

    exit_signal_handler_for_dataloader: bool = False
    """Use signal handler for dataloader workers"""

    manual_gc: bool = False
    """Disable the threshold-based default garbage collector and trigger the garbage collection
    manually. Manual garbage collection helps to align the timing of the collection across ranks
    which mitigates the impact of CPU-associated jitters. When the manual gc is enabled, garbage
    collection is performed only at the start and the end of the validation routine by default."""

    manual_gc_interval: int = 0
    """Training step interval to trigger manual garbage collection.
    When the value is set to 0, garbage collection is not triggered between training steps.
    """

    manual_gc_eval: bool = True
    """When using manual garbage collection,
    disable garbage collection at the start and the end of each evaluation run.
    """

    # ---------------- Validation config. ----------------

    eval_iters: int = 100
    """Number of iterations to run for evaluation validation/test for."""

    eval_interval: Optional[int] = 1000
    """Interval between running evaluation on validation set."""

    skip_train: bool = False
    """If set, bypass the training loop, optionally do evaluation for validation/test, and exit."""


@dataclass(kw_only=True)
class CheckpointConfig:
    """Configuration settings for model checkpointing (saving and loading)."""

    # ---------------- Checkpointing config. ----------------

    save: Optional[str] = None
    """Output directory to save checkpoints to."""

    save_interval: Optional[int] = None
    """Number of iterations between persistent checkpoint saves."""

    save_optim: bool = True
    """Do not save current optimizer."""

    save_rng: bool = True
    """Do not save current rng state."""

    load: Optional[str] = None
    """Directory containing a model checkpoint."""

    load_optim: bool = True
    """Do not load optimizer when loading checkpoint."""

    load_rng: bool = True
    """Do not load rng state when loading checkpoint."""

    non_persistent_save_interval: Optional[int] = None
    """Number of iterations between non-persistent saves."""

    non_persistent_ckpt_type: Optional[Literal["global", "local", "in_memory", "None"]] = None
    """Type of non-persistent model checkpoints.
    "global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed.
    "local" - [TBD] Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk).
    "in_memory" - [TBD] A special kind of local checkpoint that avoids serialization.
    None - No non-persistent checkpointing (default option)."""

    non_persistent_global_ckpt_dir: Optional[str] = None
    """Directory containing global non-persistent model checkpoints."""

    non_persistent_local_ckpt_dir: Optional[str] = None
    """Directory containing local non-persistent model checkpoints."""

    non_persistent_local_ckpt_algo: Literal["fully_parallel", "atomic"] = "fully_parallel"
    """Algorithm for local non-persistent checkpointing."""

    finetune: bool = False
    """Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0.
    Assumed when loading a release checkpoint."""

    pretrained_checkpoint: Optional[str] = None
    """Directory containing a pretrained model checkpoint for finetuning."""

    ckpt_step: Optional[int] = None
    """Checkpoint step to load model from."""

    use_checkpoint_args: bool = False
    """Override any command line arguments with arguments from the checkpoint"""

    exit_on_missing_checkpoint: bool = False
    """If 'load' is set, but checkpoint is not found (e.g., path typo), then exit instead of random initialization."""

    ckpt_format: Literal["torch_dist", "zarr"] = "torch_dist"
    """Checkpoint format to use."""

    ckpt_convert_format: Optional[Literal["torch", "torch_dist", "zarr"]] = None
    """Checkpoint format for conversion."""

    ckpt_convert_save: Optional[str] = None
    """Save directory for converted checkpoint."""

    fully_parallel_save: bool = True
    """Disable applying full save parallelization across DP for distributed checkpoints.
    Depending on ckpt format might decrease the number of files in the checkpoint.
    Makes DistributedOptimizer checkpoint non-reshardable."""

    async_save: bool = False
    """Apply async checkpointing save. Currently works only with `torch_dist` distributed checkpoint format."""

    use_persistent_ckpt_worker: bool = True
    """Use a persistent background worker for async checkpoint saves. When enabled, creates a dedicated
    worker thread/process for handling async saves. When disabled, uses temporal workers that are
    created and destroyed for each save operation."""

    fully_parallel_load: bool = False
    """Apply full load parallelization across DP for distributed checkpoints."""

    ckpt_assume_constant_structure: bool = False
    """If the model and optimizer state dict structure is constant throughout a *single training job,
    it allows for different checkpointing performance optimizations."""

    dist_ckpt_strictness: Literal[
        "assume_ok_unexpected",
        "log_unexpected",
        "log_all",
        "raise_unexpected",
        "raise_all",
        "return_unexpected",
        "return_all",
        "ignore_all",
    ] = "assume_ok_unexpected"
    """Determine handling of key mismatch during checkpoint load. Check StrictHandling docs for flags meaning.
    NOTE: This flag controls only distributed checkpoint load from storage, not loading state dict into the model."""

    replication: bool = False
    """If set, replication of local checkpoints is enabled. Needs to be enabled on all ranks."""

    replication_jump: Optional[int] = None
    """Specifies `J`, the spacing between ranks storing replicas of a given rank's data. Replicas
    for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`, etc. This flag has an
    effect only if --replication is used. and must be consistent across all ranks."""

    replication_factor: int = 2
    """Number of machines storing the replica of a given rank's data."""


@dataclass(kw_only=True)
class LoggerConfig:
    """Configuration settings for logging, including TensorBoard and WandB."""

    # ---------------- Logging config. ----------------

    log_interval: int = 100
    """Report loss and timing interval."""

    log_params_norm: bool = False
    """If set, calculate and log parameters norm."""

    log_throughput: bool = False
    """If set, calculate and log throughput per GPU."""

    log_progress: bool = False
    """If set, log progress (in terms of number of processed tokens and number of floating-point operations)
    to progress.txt file in checkpoint directory.
    """

    timing_log_level: Literal[0, 1, 2] = 0
    """Granularity level to measure and report timing.
    0: report only iteration time and make sure timing does not introduce extra overhead.
    1: report timing for operations that are executed very limited times (basically once) during each iteration
        (such as gradient all-reduce)
    2: report timing for operations that migh be executed numerous times during each iteration.
    Note that setting the level to 1 or 2 might cause increase in iteration time.
    """

    timing_log_option: Literal["max", "minmax", "all"] = "minmax"
    """Options for logging timing:
    max: report the max timing across all ranks
    minmax: report min and max timings across all ranks
    all: report timings of all ranks.
    """

    tensorboard_dir: Optional[str] = None
    """Write TensorBoard logs to this directory."""

    tensorboard_log_interval: int = 1
    """Report to tensorboard interval."""

    tensorboard_queue_size: int = 1000
    """Size of the tensorboard queue for pending events and summaries
    before one of the 'add' calls forces a flush to disk.
    """

    log_timers_to_tensorboard: bool = False
    """If set, write timers to tensorboard."""

    log_loss_scale_to_tensorboard: bool = True
    """Disable loss-scale logging to tensorboard."""

    log_validation_ppl_to_tensorboard: bool = False
    """If set, write validation perplexity to tensorboard."""

    log_memory_to_tensorboard: bool = False
    """Enable memory logging to tensorboard."""

    log_world_size_to_tensorboard: bool = False
    """Enable world size logging to tensorboard."""

    wandb_project: Optional[str] = None
    """The wandb project name. Ignore wandb by default."""

    wandb_exp_name: Optional[str] = None
    """The wandb experiment name."""

    wandb_save_dir: Optional[str] = None
    """Path to save the wandb results locally."""

    wandb_entity: Optional[str] = None
    """The wandb entity name."""

    logging_level: int = logging.INFO
    """Set default logging level"""

    filter_warnings: bool = True
    """Filter out warning messages"""

    modules_to_filter: Optional[list[str]] = None
    """List of modules to filter out from the logs"""

    set_level_for_all_loggers: bool = False
    """Set the logging level for all loggers. If False, only level for NeMo loggers will be set."""


@dataclass(kw_only=True)
class ProfilingConfig:
    """Configuration settings for profiling the training process."""

    # ---------------- Profiling config. ----------------

    use_nsys_profiler: bool = False
    """Enable nsys profiling. When using this option, nsys options should be specified in
    commandline. An example nsys commandline is
    `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true
    --capture-range=cudaProfilerApi --capture-range-end=stop`.
    """

    profile_step_start: int = 10
    """Global step to start profiling."""

    profile_step_end: int = 12
    """Global step to stop profiling."""

    use_pytorch_profiler: bool = False
    """Use the built-in pytorch profiler. Useful if you wish to view profiles in tensorboard."""

    profile_ranks: list[int] = field(default_factory=lambda: [0])
    """Global ranks to profile."""

    record_memory_history: bool = False
    """Record memory history in last rank."""

    memory_snapshot_path: str = "snapshot.pickle"
    """Specifies where to dump the memory history pickle."""

    record_shapes: bool = False
    """Record shapes of tensors."""

    def __post_init__(self) -> None:
        """Validate profiling configuration."""
        assert not (self.use_pytorch_profiler and self.use_nsys_profiler), (
            "Exactly one of pytorch or nsys profiler should be enabled, not both, when ProfilingConfig is active."
        )


@dataclass
class FaultToleranceConfig:
    """Configuration settings related to fault tolerance mechanisms (NVIDIA internal use)."""

    enable_ft_package: bool = False
    """If set, Fault Tolerance package is enabled. Note: This feature is for Nvidia internal use only."""

    calc_ft_timeouts: bool = False
    """If set, FT package will try to automatically compute the timeouts.
    Note: This feature is for Nvidia internal use only.
    """

    simulate_fault: bool = False
    """Sets a simulated fault for fault tolerance. NOTE: This if for fault tolerance testing only."""

    simulated_fault_type: Literal["rank_hung", "rank_killed", "random"] = "random"
    """How the simulated fault should behave. 'random' will randomly choose one of the other two options."""

    simulated_fault_rank: Optional[int] = None
    """Rank on which simulated fault should occur."""

    simulated_fault_base_delay: int = 0
    """Base delay before simulated fault thread is started. A small random delay is added to this."""


@dataclass
class StragglerDetectionConfig:
    """Configuration settings for detecting and logging GPU stragglers."""

    log_straggler: bool = False
    """If set, tracks and logs straggler per GPU."""

    enable_straggler_on_startup: bool = True
    """If set, StragglerDetector is disabled on startup."""

    straggler_ctrlr_port: int = 65535
    """Port number to toggle StragglerDetector on/off at runtime"""

    straggler_minmax_count: int = 1
    """Number of ranks to report with high/low estimated throughput"""

    disable_straggler_on_startup: bool = False
    """If set, StragglerDetector is disabled on startup."""


@dataclass
class NVRxStragglerDetectionConfig:
    """Configuration settings for NVIDIA Resiliency Extension straggler detection."""

    enabled: bool = False
    """Enable NVRx straggler detection."""

    report_time_interval: float = 300.0
    """Interval [seconds] of the straggler check."""

    calc_relative_gpu_perf: bool = True
    """Calculate relative GPU performance scores."""

    calc_individual_gpu_perf: bool = True
    """Calculate individual GPU performance scores."""

    num_gpu_perf_scores_to_print: int = 5
    """How many best and worst perf scores to print (0 - does not print periodically,
    but only if stragglers are detected)."""

    gpu_relative_perf_threshold: float = 0.7
    """Threshold for relative GPU performance scores."""

    gpu_individual_perf_threshold: float = 0.7
    """Threshold for individual GPU performance scores."""

    stop_if_detected: bool = False
    """Set to True, to terminate the workload if stragglers are detected."""

    enable_logging: bool = True
    """Set to True, to log GPU performance scores."""

    profiling_interval: int = 1
    """Profiling interval passed to straggler.Detector.initialize."""

    logger_name: str = "megatron_hub.NVRxStragglerDetection"
    """Logger name for straggler detection messages."""

    def __post_init__(self) -> None:
        """Validate NVRx straggler detection configuration."""
        if self.enabled:
            if not (self.calc_relative_gpu_perf or self.calc_individual_gpu_perf):
                raise ValueError(
                    "At least one of calc_relative_gpu_perf or calc_individual_gpu_perf must be True "
                    "when NVRx straggler detection is enabled."
                )
            if self.report_time_interval <= 0:
                raise ValueError("report_time_interval must be positive.")
            if not (0.0 <= self.gpu_relative_perf_threshold <= 1.0):
                raise ValueError("gpu_relative_perf_threshold must be between 0.0 and 1.0.")
            if not (0.0 <= self.gpu_individual_perf_threshold <= 1.0):
                raise ValueError("gpu_individual_perf_threshold must be between 0.0 and 1.0.")


# ---------------- Container config (standalone top-level config) ----------------
@dataclass(kw_only=True)
class ConfigContainer(Container):
    """Top-level container holding all configuration objects."""

    rng: RNGConfig = field(default_factory=RNGConfig)
    rerun_state_machine: RerunStateMachineConfig = field(default_factory=RerunStateMachineConfig)
    train: TrainingConfig
    model: GPTModelProvider | T5ModelProvider
    optimizer: OptimizerConfig
    ddp: DistributedDataParallelConfig = field(default_factory=DistributedDataParallelConfig)
    scheduler: SchedulerConfig
    dataset: GPTDatasetConfig | FinetuningDatasetConfig
    logger: LoggerConfig
    tokenizer: TokenizerConfig
    checkpoint: CheckpointConfig
    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    ft: Optional[FaultToleranceConfig] = None
    straggler: Optional[StragglerDetectionConfig] = None
    nvrx_straggler: Optional[NVRxStragglerDetectionConfig] = None
    profiling: Optional[ProfilingConfig] = None

    def get_data_parallel_size(self, world_size: int) -> int:
        """Calculate the data parallel size based on the model configuration."""
        model_cfg = self.model
        encoder_model_size = (
            getattr(model_cfg, "encoder_tensor_model_parallel_size", 0)
            * getattr(model_cfg, "encoder_pipeline_model_parallel_size", 0)
            * model_cfg.context_parallel_size
        )
        decoder_model_size = (
            model_cfg.tensor_model_parallel_size
            * model_cfg.pipeline_model_parallel_size
            * model_cfg.context_parallel_size
        )
        total_model_size = encoder_model_size + decoder_model_size
        assert world_size % total_model_size == 0, f"""
        world size ({world_size}) is not divisible by total_model_size ({encoder_model_size=} + {decoder_model_size=})
        """
        return world_size // total_model_size

    def validate(self) -> None:
        """Performs validation checks on the combined configuration.

        Calculates dependent values like data_parallel_size and scheduler steps.
        Ensures compatibility between different configuration settings.
        """
        # Re-run post-inits of sub-configs
        for f in fields(self):
            sub_cfg = getattr(self, f.name)
            if hasattr(sub_cfg, "__post_init__"):
                sub_cfg.__post_init__()

        # Run validations

        # Distributed
        world_size = get_world_size_safe()
        self.data_parallel_size = self.get_data_parallel_size(world_size)

        self.model.use_cpu_initialization = self.model.use_cpu_initialization or self.dist.lazy_init

        # Make sure all functionality that requires Gloo process groups is disabled.
        if not self.dist.use_gloo_process_groups:
            if self.optimizer.use_distributed_optimizer:
                # If using distributed optimizer, must use distributed checkpointing.
                # Legacy checkpointing uses Gloo process groups to collect full distributed
                # optimizer state in the CPU memory of DP rank 0.
                assert self.checkpoint.ckpt_format == "torch_dist"

        # Scheduler
        if self.scheduler.lr_decay_iters is None:
            self.scheduler.lr_decay_iters = self.train.train_iters
        self.scheduler.lr_decay_steps = self.scheduler.lr_decay_iters * self.train.global_batch_size
        self.scheduler.wd_incr_steps = self.train.train_iters * self.train.global_batch_size
        self.scheduler.wsd_decay_steps = None
        if self.scheduler.lr_wsd_decay_iters is not None:
            self.scheduler.wsd_decay_steps = self.scheduler.lr_wsd_decay_iters * self.train.global_batch_size
        if self.scheduler.lr_warmup_fraction is not None:
            self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_fraction * self.scheduler.lr_decay_iters
        else:
            self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_iters * self.train.global_batch_size

        if (
            isinstance(self.dataset, FinetuningDatasetConfig)
            and self.dataset.packed_sequence_specs is not None
            and self.dataset.packed_sequence_specs.packed_sequence_size > 0
            and self.train.micro_batch_size > 1
        ):
            packed_sequence_size = self.dataset.packed_sequence_specs.packed_sequence_size
            raise ValueError(
                "Micro batch size should be 1 when training with packed sequence, but your micro batch size "
                f"is {self.train.micro_batch_size}. \nThe following config is equivalent to your current setting for "
                f"a packed dataset. Please update your config to the following: \n"
                f"Set micro batch size to 1 (currently {self.train.micro_batch_size})\n"
                f"Set global batch size to {self.train.global_batch_size // self.train.micro_batch_size} "
                f"(currently {self.train.global_batch_size}) \n"
                f"Set packed sequence length to {packed_sequence_size * self.train.micro_batch_size} "
                f"(currently {packed_sequence_size}) \n"
                f"For details please visit "
                f"https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html"
            )
