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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.models import GPTModelProvider, T5ModelProvider
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.deepep import validate_deepep
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, get_mixed_precision_config
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.training.utils.config_utils import _ConfigContainerBase as Container
from megatron.bridge.utils.common_utils import (
    get_world_size_safe,
    print_rank_0,
    warn_rank_0,
)


@dataclass
class DistributedDataParallelConfig(MCoreDistributedDataParallelConfig):
    """Megatron Core DistributedDataParallelConfig with deferred post-init.

    This class inherits from Megatron Core's DistributedDataParallelConfig but defers the
    execution of post_init() until finalize() is explicitly called. This allows
    for field modifications after construction but before computed fields are calculated.
    """

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core DistributedDataParallelConfig.__post_init__()
        to compute derived fields based on the current field values.
        """
        super().__post_init__()


@dataclass
class OptimizerConfig(MCoreOptimizerConfig):
    """Megatron Core OptimizerConfig with deferred post-init.

    This class inherits from Megatron Core's OptimizerConfig but defers the
    execution of post_init() until finalize() is explicitly called. This allows
    for field modifications after construction but before computed fields are calculated.
    """

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core OptimizerConfig.__post_init__()
        to compute derived fields based on the current field values.
        """
        super().__post_init__()


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

    use_megatron_fsdp: bool = False
    """Use Megatron's Fully Sharded Data Parallel. Cannot be used together with use_torch_fsdp2."""

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

    use_sharp: bool = False
    """Set the use of SHARP for the collective communications of data-parallel process groups.
    When `True`, run barrier within each data-parallel process group,
    which specifies the SHARP application target groups.
    """

    sharp_enabled_group: Optional[Literal["dp", "dp_replica"]] = None
    """IB SHARP can be enabled from only one communication group.
    By default, it is enabled from dp group if not specified and use_sharp=True.
    Available options: [dp, dp_replica]
    """

    high_priority_stream_groups: Optional[list[str]] = None
    """Specify which communicator groups should use high priority streams during creation.
    Assigning high priority to communication streams ensures that communication kernels
    are scheduled with higher priority, minimizing the exposed communication when it is
    overlapped with other computation kernels.
    """

    external_gpu_device_mapping: bool = False
    """If True, indicates that GPU device mapping has been externally managed
    (e.g., via CUDA_VISIBLE_DEVICES environment variable). When True, uses device 0
    instead of local rank for CUDA device selection. This is useful when launching
    with external process managers that handle GPU visibility.
    """

    enable_megatron_core_experimental: bool = False
    """Enable experimental features for Megatron Core."""


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

    check_for_nan_in_loss: bool = True
    """Check for NaN in the loss."""

    check_for_spiky_loss: bool = False
    """Check for spiky loss."""


@dataclass(kw_only=True)
class DataloaderConfig:
    """Base configuration for data loading."""

    dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = None
    """Dataloader type: 'single' for single pass, 'cyclic' for multiple passes with shuffling,
    'batch' for global batch sampling (used in fine-tuning), or 'external' for custom dataloaders."""

    num_workers: int = 8
    """Dataloader number of workers."""

    data_sharding: bool = True
    """Disable data sharding."""

    pin_memory: bool = True
    """Whether to pin memory during data loading for faster GPU training."""

    persistent_workers: bool = False
    """Whether to keep data loading workers persistent across epochs."""


@dataclass(frozen=True)
class DatasetBuildContext:
    """Interface that encapsulates framework internals.

    This context provides metadata needed to build datasets
    while hiding implementation details of the framework.

    Attributes:
        train_samples: Number of samples for training dataset
        valid_samples: Number of samples for validation dataset
        test_samples: Number of samples for test dataset
        tokenizer: Optional tokenizer instance for text processing
    """

    train_samples: int
    valid_samples: int
    test_samples: int
    tokenizer: Optional[MegatronTokenizer] = None


@dataclass
class DatasetProvider(DataloaderConfig, ABC):
    """Abstract base class for custom dataset configurations.

    Provides an interface for users to implement their own dataset builders
    while automatically inheriting all DataloaderConfig functionality.

    Users must:
    1. Inherit from this class
    2. Implement the build_datasets() method

    Example:
        @dataclass
        class S3DatasetConfig(DatasetProvider):
            bucket_name: str
            data_prefix: str
            seq_length: int

            def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
                # Custom implementation to load data from S3
                train_ds = load_s3_dataset(self.bucket_name, f"{self.data_prefix}/train", context.tokenizer)
                valid_ds = load_s3_dataset(self.bucket_name, f"{self.data_prefix}/valid", context.tokenizer)
                test_ds = load_s3_dataset(self.bucket_name, f"{self.data_prefix}/test", context.tokenizer)
                return train_ds, valid_ds, test_ds
    """

    @abstractmethod
    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build train, validation, and test datasets.

        This method is called by the framework during dataset initialization.
        Implementations should use the provided context to create appropriate
        datasets for each split.

        Args:
            context: Build context with sample counts and tokenizer

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset)
            Any element can be None if that split shouldn't be created.

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass


@dataclass
class GPTDatasetConfig(MCoreGPTDatasetConfig, DataloaderConfig):
    """Megatron Core GPTDatasetConfig with deferred post-init.

    This class inherits from MCore's GPTDatasetConfig and DataloaderConfig but defers the
    execution of post_init() until finalize() is explicitly called. This allows
    for field modifications after construction but before computed fields are calculated.
    """

    skip_getting_attention_mask_from_dataset: bool = True
    """If set, the dataset will pass a None attention mask and the attention
    mask is autogenerated from the attn backend"""

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic and Bridge-specific checks.

        This method calls the original Megatron Core GPTDatasetConfig.__post_init__()
        and then performs Bridge-specific validation.
        """
        # Call MCore's post_init
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
    """Configuration specific to finetuning datasets, inheriting from DataloaderConfig.

    Note: For fine-tuning, dataloader_type defaults to 'batch' which ensures sequences
    within each global batch are padded to the same length.
    """

    dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = "batch"
    """Dataloader type for fine-tuning. Defaults to 'batch' for optimal padding behavior."""

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
    """number of iterations to decay learning rate over, If None defaults to `train.train_iters`"""

    lr_decay_samples: Optional[int] = None
    """number of samples to decay learning rate over, If None defaults to `train.train_samples`"""

    lr_wsd_decay_iters: Optional[int] = None
    """number of iterations for the annealing phase in the wsd schedule"""

    lr_wsd_decay_samples: Optional[int] = None
    """number of samples for the annealing phase in the wsd schedule"""

    lr_warmup_fraction: Optional[float] = None
    """fraction of lr-warmup-(iters/samples) to use for warmup (as a float)"""

    lr_warmup_iters: int = 0
    """number of iterations to linearly warmup learning rate over."""

    lr_warmup_samples: int = 0
    """number of samples to linearly warmup learning rate over."""

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

    def finalize(self) -> None:
        """Post-initialization checks for scheduler config."""
        if self.start_weight_decay is not None:
            assert self.start_weight_decay >= 0.0, "start_weight_decay should be positive."
            assert self.end_weight_decay >= self.start_weight_decay

        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, "both override and use-checkpoint are set."

        # Validate mutual exclusivity between iteration-based and sample-based scheduler fields
        has_iter_fields = (
            self.lr_decay_iters is not None or self.lr_warmup_iters != 0 or self.lr_wsd_decay_iters is not None
        )
        has_sample_fields = (
            self.lr_decay_samples is not None or self.lr_warmup_samples != 0 or self.lr_wsd_decay_samples is not None
        )

        assert not (has_iter_fields and has_sample_fields), (
            f"Cannot mix iteration-based and sample-based scheduler fields. "
            f"Found iteration fields: lr_decay_iters={self.lr_decay_iters}, lr_warmup_iters={self.lr_warmup_iters}, lr_wsd_decay_iters={self.lr_wsd_decay_iters}. "
            f"Found sample fields: lr_decay_samples={self.lr_decay_samples}, lr_warmup_samples={self.lr_warmup_samples}, lr_wsd_decay_samples={self.lr_wsd_decay_samples}. "
            f"Use either iteration fields OR sample fields, not both."
        )

        # Validate mutual exclusivity between lr_warmup_fraction and specific warmup fields
        if self.lr_warmup_fraction is not None:
            assert self.lr_warmup_iters == 0 and self.lr_warmup_samples == 0, (
                f"Cannot specify lr_warmup_fraction={self.lr_warmup_fraction} with lr_warmup_iters={self.lr_warmup_iters} or lr_warmup_samples={self.lr_warmup_samples}. "
                f"Use either lr_warmup_fraction OR lr_warmup_iters OR lr_warmup_samples."
            )


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
    Note that either train_iters or train_samples should be provided.
    """

    train_samples: Optional[int] = None
    """Total number of samples to train over all training runs.
    Note that either train_iters or train_samples should be provided."""

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

    iterations_to_skip: list[int] = field(default_factory=list)
    """List of iterations to skip during training, empty by default."""

    # ---------------- Validation config. ----------------

    eval_iters: int = 100
    """Number of iterations to run for evaluation validation/test for."""

    eval_interval: Optional[int] = 1000
    """Interval between running evaluation on validation set."""

    skip_train: bool = False
    """If set, bypass the training loop, optionally do evaluation for validation/test, and exit."""

    def finalize(self) -> None:
        """Validate training mode specification and calculate train_iters from train_samples if needed."""
        has_train_iters = self.train_iters is not None
        has_train_samples = self.train_samples is not None

        assert has_train_iters or has_train_samples, "Either train_iters or train_samples must be provided"
        assert not (has_train_iters and has_train_samples), "Cannot specify both train_iters and train_samples"
        if has_train_samples:
            assert self.train_samples > 0, "train_samples must be positive"
            assert self.rampup_batch_size is None, "Batch size rampup not supported with sample-based training yet"

            # Calculate train_iters from train_samples (rampup_batch_size already validated as None)
            self.train_iters = self.train_samples // self.global_batch_size
            print_rank_0(f"Setting training iterations to {self.train_iters} based on {self.train_samples} samples")


@dataclass(kw_only=True)
class CheckpointConfig:
    """Configuration settings for model checkpointing (saving and loading)."""

    # ---------------- Checkpointing config. ----------------

    save: Optional[str] = None
    """Output directory to save checkpoints to."""

    save_interval: Optional[int] = None
    """Number of iterations between persistent checkpoint saves."""

    most_recent_k: Optional[int] = -1
    """Number of latest checkpoint to be saved."""

    save_optim: bool = True
    """Do not save current optimizer."""

    save_rng: bool = True
    """Do not save current rng state."""

    load: Optional[str] = None
    """Directory containing a model checkpoint."""

    load_optim: bool = True
    """Do not load optimizer when loading checkpoint."""

    load_main_params_from_ckpt: bool = False
    """Load main parameters from checkpoint. When loading a model from a checkpoint without loading
    the optimizer, the model parameters are updated but for fp16 optimizer with main parameters,
    the main parameters need to also be updated.
    """

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

    ckpt_format: Literal["torch_dist", "zarr", "fsdp_dtensor"] = "torch_dist"
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
    """Assume the checkpoint structure is constant across saves to enable optimizations."""

    strict_fsdp_dtensor_load: bool = False
    """Whether to enforce strict loading for FSDP DTensor checkpoints. When False, allows partial loading."""

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

    save_tokenizer_assets: bool = True
    """Save tokenizer files to checkpoint directory. When enabled, saves all tokenizer artifacts
    (vocab files, special tokens, tokenizer config) to make checkpoints self-contained and portable.
    Set to False for performance-sensitive scenarios where tokenizer files are not needed."""

    replication: bool = False
    """If set, replication of local checkpoints is enabled. Needs to be enabled on all ranks."""

    replication_jump: Optional[int] = None
    """Specifies `J`, the spacing between ranks storing replicas of a given rank's data. Replicas
    for rank `n` may be on ranks `n+J`, `n+2J`, ..., or `n-J`, `n-2J`, etc. This flag has an
    effect only if --replication is used. and must be consistent across all ranks."""

    replication_factor: int = 2
    """Number of machines storing the replica of a given rank's data."""

    def finalize(self) -> None:
        """Post-initialization checks for checkpoint config."""
        if self.load_main_params_from_ckpt:
            assert not self.load_optim, "load_main_params_from_ckpt must be used with load_optim=False"

        if self.async_save:
            assert self.save is not None, "async_save is enabled, but save is not set. Set save to a valid path."
            assert self.use_persistent_ckpt_worker, "async_save requires use_persistent_ckpt_worker=True."


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

    log_throughput_to_tensorboard: bool = False
    """Enable throughput logging to tensorboard."""

    throughput_window_size: int = 100
    """Number of batches to use for a rolling average of throughput."""

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

    memory_keys: dict[str, str] | None = None
    """Names of memory statistics to log from `torch.cuda.memory_stats()`"""

    log_l2_norm_grad_to_tensorboard: bool = False
    """Enable gradients logging to tensorboard."""

    log_runtime_to_tensorboard: bool = False
    """Enable runtime metrics logging to tensorboard."""

    runtime_time_unit: str = "hours"
    """ Time unit to use for time logging. """

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

    log_energy: bool = False
    """If set, log energy consumption (in Joules)."""


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

    nvtx_ranges: bool = False
    """Enable NVTX range annotations for profiling. When enabled, inserts NVTX markers
    to categorize execution in profiler output."""

    def finalize(self) -> None:
        """Validate profiling configuration."""
        assert not (self.use_pytorch_profiler and self.use_nsys_profiler), (
            "Exactly one of pytorch or nsys profiler should be enabled, not both, when ProfilingConfig is active."
        )
        assert self.profile_step_start >= 0, f"profile_step_start must be >= 0, got {self.profile_step_start}"
        assert self.profile_step_end >= 0, f"profile_step_end must be >= 0, got {self.profile_step_end}"
        assert self.profile_step_end >= self.profile_step_start, (
            f"profile_step_end ({self.profile_step_end}) must be >= profile_step_start ({self.profile_step_start})"
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

    logger_name: str = "megatron.bridge.NVRxStragglerDetection"
    """Logger name for straggler detection messages."""

    def finalize(self) -> None:
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


@dataclass
class InProcessRestartConfig:
    """Configuration settings for NVIDIA Resiliency Extension in-process restart functionality."""

    enabled: bool = False
    """Enable in-process restart mechanism from nvidia-resiliency-ext."""

    max_iterations: Optional[int] = None
    """Maximum number of in-process restart iterations."""

    monitor_thread_interval: float = 1.0
    """Monitoring interval (in seconds) for the monitoring thread."""

    monitor_process_interval: float = 1.0
    """Monitoring interval (in seconds) for the monitoring process."""

    progress_watchdog_interval: float = 1.0
    """Interval (in seconds) for automatic progress watchdog timestamp updates."""

    heartbeat_interval: float = 30.0
    """Monitoring interval (in seconds) for detecting unresponsive ranks."""

    soft_timeout: float = 60.0
    """Soft progress timeout (in seconds)."""

    hard_timeout: float = 90.0
    """Hard progress timeout (in seconds)."""

    heartbeat_timeout: float = 60.0
    """Timeout (in seconds) for a missing rank detection heartbeat."""

    barrier_timeout: float = 120.0
    """Timeout (in seconds) for internal distributed barrier."""

    completion_timeout: float = 120.0
    """Timeout (in seconds) for barrier on completion on all ranks."""

    last_call_wait: float = 1.0
    """Time interval (in seconds) for other ranks to report concurrent terminal failures."""

    termination_grace_time: float = 1.0
    """Interval (in seconds) between SIGTERM and SIGKILL issued on hard timeout."""

    granularity: Literal["node", "rank"] = "node"
    """Granularity for in-process restart."""

    active_world_size: Optional[int] = None
    """The number of ranks initially executing the workload.
    The remaining ranks from the allocation are set aside as warm reserve.
    If None, defaults to WORLD_SIZE environment variable."""

    empty_cuda_cache: bool = True
    """Empty CUDA cache during restart finalization."""

    max_rank_faults: Optional[int] = None
    """Maximum number of rank faults allowed before terminating the job."""

    monitor_process_logdir: Optional[str] = None
    """Directory for monitor process log files. If None, monitor process logging is disabled."""


# ---------------- Container config (standalone top-level config) ----------------
@dataclass(kw_only=True)
class ConfigContainer(Container):
    """Top-level container holding all configuration objects."""

    rng: RNGConfig = field(default_factory=RNGConfig)
    rerun_state_machine: RerunStateMachineConfig = field(default_factory=RerunStateMachineConfig)
    train: TrainingConfig
    model: GPTModelProvider | T5ModelProvider | MambaModelProvider
    optimizer: OptimizerConfig
    ddp: DistributedDataParallelConfig = field(default_factory=DistributedDataParallelConfig)
    scheduler: SchedulerConfig
    dataset: GPTDatasetConfig | FinetuningDatasetConfig | DatasetProvider
    logger: LoggerConfig
    tokenizer: TokenizerConfig
    checkpoint: CheckpointConfig
    dist: DistributedInitConfig = field(default_factory=DistributedInitConfig)
    ft: Optional[FaultToleranceConfig] = None
    straggler: Optional[StragglerDetectionConfig] = None
    nvrx_straggler: Optional[NVRxStragglerDetectionConfig] = None
    profiling: Optional[ProfilingConfig] = None
    peft: Optional[PEFT] = None
    comm_overlap: Optional[CommOverlapConfig] = None
    mixed_precision: Optional[Union[MixedPrecisionConfig, str]] = None
    inprocess_restart: Optional[InProcessRestartConfig] = None

    def get_data_parallel_size(self, world_size: int) -> int:
        """Calculate the data parallel size based on the model configuration."""
        model_cfg = self.model
        total_model_size = (
            model_cfg.tensor_model_parallel_size
            * model_cfg.pipeline_model_parallel_size
            * model_cfg.context_parallel_size
        )
        assert world_size % total_model_size == 0, f"""
        world size ({world_size}) is not divisible by total_model_size ({model_cfg.tensor_model_parallel_size=} * {model_cfg.pipeline_model_parallel_size=} * {model_cfg.context_parallel_size=})
        """
        return world_size // total_model_size

    def set_data_parallel_size(self) -> None:
        """Calculate and set data_parallel_size for this config and comm_overlap config.

        This method calculates the data parallel size needed by setup methods, without
        triggering full validation or finalization of Megatron Core configs.
        """
        # Calculate data parallel size (needed for comm overlap setup)
        world_size = get_world_size_safe()
        self.data_parallel_size = self.get_data_parallel_size(world_size)

        # Set data_parallel_size on comm_overlap config if present
        if self.comm_overlap is not None:
            self.comm_overlap.data_parallel_size = self.data_parallel_size

    def validate(self) -> None:
        """Performs validation checks on the combined configuration.

        Calculates dependent values like data_parallel_size and scheduler steps.
        Ensures compatibility between different configuration settings.
        """

        if isinstance(self.dataset, GPTDatasetConfig):
            self.dataset.finalize()
        if hasattr(self.ddp, "finalize"):
            self.ddp.finalize()
        if hasattr(self.optimizer, "finalize"):
            self.optimizer.finalize()
        if hasattr(self.model, "finalize"):
            self.model.finalize()

        self.train.finalize()
        self.scheduler.finalize()
        self.checkpoint.finalize()
        if self.profiling is not None:
            self.profiling.finalize()
        if self.nvrx_straggler is not None:
            self.nvrx_straggler.finalize()

        # Re-run post-inits of sub-configs
        for f in fields(self):
            sub_cfg = getattr(self, f.name)
            if hasattr(sub_cfg, "__post_init__") and not hasattr(sub_cfg, "finalize"):
                sub_cfg.__post_init__()

        # Distributed - ensure data_parallel_size is calculated (might already be set by set_data_parallel_size)
        if not hasattr(self, "data_parallel_size") or self.data_parallel_size is None:
            world_size = get_world_size_safe()
            self.data_parallel_size = self.get_data_parallel_size(world_size)
            # Set data_parallel_size on comm_overlap config if present
            if self.comm_overlap is not None:
                self.comm_overlap.data_parallel_size = self.data_parallel_size

        # Run validations
        _validate_and_sync_distributed_optimizer_settings(self)

        if self.dist.use_megatron_fsdp and self.dist.use_torch_fsdp2:
            raise ValueError("Using use_megatron_fsdp and use_torch_fsdp2 at the same time is not supported.")

        # Megatron FSDP Config checks
        if self.dist.use_megatron_fsdp or self.ddp.use_megatron_fsdp:
            # Set Megatron FSDP Configs
            self.dist.use_megatron_fsdp = True
            self.ddp.use_megatron_fsdp = True

            assert not self.dist.use_tp_pp_dp_mapping, "use_tp_pp_dp_mapping is not supported with Megatron FSDP"

            if self.checkpoint.save is not None or self.checkpoint.load is not None:
                # only check if saving or loading
                assert self.checkpoint.ckpt_format == "fsdp_dtensor", (
                    "Megatron FSDP only supports fsdp_dtensor checkpoint format"
                )

            if self.ddp.average_in_collective:
                print_rank_0("average_in_collective is not supported with Megatron FSDP, setting to True")
                self.ddp.average_in_collective = False

            if self.optimizer.use_precision_aware_optimizer:
                self.ddp.preserve_fp32_weights = False

        # Checkpoint
        if self.checkpoint.save is not None or self.checkpoint.load is not None:
            # only check if saving or loading
            if self.checkpoint.ckpt_format == "fsdp_dtensor":
                assert self.ddp.use_megatron_fsdp and not self.dist.use_torch_fsdp2, (
                    "fsdp_dtensor checkpoint format only supports Megatron FSDP"
                )

        # Enforce async_save format restriction
        if self.checkpoint.async_save:
            assert self.checkpoint.ckpt_format == "torch_dist", (
                "async_save is only supported with ckpt_format='torch_dist'"
            )

        self.model.use_cpu_initialization = self.model.use_cpu_initialization or self.dist.lazy_init

        # Make sure all functionality that requires Gloo process groups is disabled.
        if not self.dist.use_gloo_process_groups:
            if self.optimizer.use_distributed_optimizer:
                # If using distributed optimizer, must use distributed checkpointing.
                # Legacy checkpointing uses Gloo process groups to collect full distributed
                # optimizer state in the CPU memory of DP rank 0.
                assert self.checkpoint.ckpt_format == "torch_dist"

        # Cross-validation between training and scheduler configs
        self._validate_training_scheduler_compatibility()

        # Calculate scheduler steps for both iteration-based and sample-based training
        self._calculate_scheduler_steps()

        if self.model.context_parallel_size > 1:
            assert self.model.seq_length % (self.model.context_parallel_size * 2) == 0, (
                "Sequence length must be divisible by 2 * context parallel size if context parallel is used."
            )
            if isinstance(self.dataset, FinetuningDatasetConfig):
                # check calculate_per_token_loss to be True
                # check average_in_collective to be False
                # for context parallel to solve the issue of nan loss on ranks with all tokens masked
                # (only happens in SFT)
                assert self.model.calculate_per_token_loss, (
                    "When finetuning with CP>1, calculate_per_token_loss must be True"
                )
                assert not self.ddp.average_in_collective, (
                    "When finetuning with CP>1, average_in_collective must be False"
                )

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

        if self.peft is not None:
            assert self.checkpoint.pretrained_checkpoint is not None, "PEFT requires a pretrained checkpoint path"

        if self.dataset is not None:
            data_seq_length = (
                self.dataset.seq_length
                if isinstance(self.dataset, FinetuningDatasetConfig)
                else self.dataset.sequence_length
            )

            assert self.model.seq_length == data_seq_length, (
                f"Please ensure sequence length configuration in model config and "
                f"dataset config match.\nSequence length in model config: {self.model.seq_length}, "
                f"Sequence length in dataset config: {data_seq_length}"
            )

        # Validate DeepEP is supported for the current GPU architecture
        validate_deepep(self.model)

        # Sync config. If TE RNG tracker is set in either ways, set them in both places.
        if self.rng.te_rng_tracker or self.model.use_te_rng_tracker:
            self.model.use_te_rng_tracker = self.rng.te_rng_tracker = True

    def _validate_training_scheduler_compatibility(self) -> None:
        """Cross-validation between training and scheduler configs."""
        has_train_samples = self.train.train_samples is not None

        if has_train_samples:
            # Sample-based training validation
            assert self.scheduler.lr_decay_iters is None, (
                "Use lr_decay_samples for sample-based training, not lr_decay_iters"
            )
            assert self.scheduler.lr_warmup_iters == 0, (
                "Use lr_warmup_samples for sample-based training, not lr_warmup_iters"
            )
            assert not (self.scheduler.lr_warmup_fraction is not None and self.scheduler.lr_warmup_samples != 0), (
                "Can only specify one of lr_warmup_fraction or lr_warmup_samples"
            )
        else:
            # Iteration-based training validation
            assert self.scheduler.lr_decay_samples is None, (
                "Use lr_decay_iters for iteration-based training, not lr_decay_samples"
            )
            assert self.scheduler.lr_warmup_samples == 0, (
                "Use lr_warmup_iters for iteration-based training, not lr_warmup_samples"
            )
            assert not (self.scheduler.lr_warmup_fraction is not None and self.scheduler.lr_warmup_iters != 0), (
                "Can only specify one of lr_warmup_fraction or lr_warmup_iters"
            )

    def _calculate_scheduler_steps(self) -> None:
        """Calculate scheduler steps for both iteration-based and sample-based training."""
        is_sample_based = self.train.train_samples is not None

        if is_sample_based:
            if self.scheduler.lr_decay_samples is None:
                self.scheduler.lr_decay_samples = self.train.train_samples
            self.scheduler.lr_decay_steps = self.scheduler.lr_decay_samples
            self.scheduler.wd_incr_steps = self.train.train_samples

            if self.scheduler.lr_wsd_decay_samples is not None:
                self.scheduler.wsd_decay_steps = self.scheduler.lr_wsd_decay_samples

            # Warmup calculation for sample-based training
            if self.scheduler.lr_warmup_fraction is not None:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_fraction * self.scheduler.lr_decay_steps
            else:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_samples
        else:
            # Iteration-based training
            if self.scheduler.lr_decay_iters is None:
                self.scheduler.lr_decay_iters = self.train.train_iters
            self.scheduler.lr_decay_steps = self.scheduler.lr_decay_iters * self.train.global_batch_size
            self.scheduler.wd_incr_steps = self.train.train_iters * self.train.global_batch_size

            if self.scheduler.lr_wsd_decay_iters is not None:
                self.scheduler.wsd_decay_steps = self.scheduler.lr_wsd_decay_iters * self.train.global_batch_size

            if self.scheduler.lr_warmup_fraction is not None:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_fraction * self.scheduler.lr_decay_steps
            else:
                self.scheduler.lr_warmup_steps = self.scheduler.lr_warmup_iters * self.train.global_batch_size


def runtime_config_update(cfg: ConfigContainer) -> None:
    """Apply runtime configuration updates prior to initialization.

    This function handles all configuration modifications that need to happen
    after initial config creation but before final validation and model setup.

    Steps:
    1. Resolve mixed precision configuration from string if needed
    2. Apply mixed precision settings to model, optimizer, and DDP configs
    3. Calculate data parallel size (needed for comm overlap)
    4. Apply communication overlap configuration
    5. Validate configuration after all modifications

    Args:
        cfg: Configuration container to update
    """
    # Apply mixed precision configuration if provided
    if cfg.mixed_precision is not None:
        if isinstance(cfg.mixed_precision, str):
            cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
        cfg.mixed_precision.finalize()
        cfg.mixed_precision.setup(cfg.model, cfg.optimizer, cfg.ddp)

    # Calculate data parallel size (needed for comm overlap methods)
    cfg.set_data_parallel_size()

    # Apply communication overlap configuration if provided
    if cfg.comm_overlap is not None:
        cfg.comm_overlap.finalize()
        cfg.comm_overlap.setup(cfg.model, cfg.optimizer, cfg.ddp)

    # Validate configuration after all modifications
    cfg.validate()


def _validate_and_sync_distributed_optimizer_settings(config: ConfigContainer) -> None:
    """Validate and synchronize distributed optimizer settings between DDP and optimizer configs.

    This function ensures that distributed optimizer settings are consistent across
    DDP and optimizer configurations. If either setting is enabled, both will be
    enabled to maintain consistency.

    Args:
        config: The configuration container to validate and potentially modify.
    """
    ddp_setting = config.ddp.use_distributed_optimizer
    optimizer_setting = config.optimizer.use_distributed_optimizer

    if ddp_setting or optimizer_setting:
        if ddp_setting != optimizer_setting:
            warn_rank_0(
                f"Distributed optimizer settings were not in sync: "
                f"ddp.use_distributed_optimizer={ddp_setting}, "
                f"optimizer.use_distributed_optimizer={optimizer_setting}. "
                f"Automatically enabling distributed optimizer for both settings."
            )
        config.ddp.use_distributed_optimizer = True
        config.optimizer.use_distributed_optimizer = True
