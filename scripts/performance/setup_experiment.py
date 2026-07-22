#!/usr/bin/env python3

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

import glob
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home


try:
    from argument_parser import NUM_GPUS_PER_NODE_MAP, parse_cli_args
    from utils.executors import (
        _kubeflow_numa_binding_enabled,
        _kubeflow_numa_binding_script,
        kubeflow_executor,
        slurm_executor,
    )
    from utils.utils import configure_slurm_gpu_tuning, select_config_variant_interactive
except (ImportError, ModuleNotFoundError):
    from .argument_parser import NUM_GPUS_PER_NODE_MAP, parse_cli_args
    from .utils.executors import (
        _kubeflow_numa_binding_enabled,
        _kubeflow_numa_binding_script,
        kubeflow_executor,
        slurm_executor,
    )
    from .utils.utils import configure_slurm_gpu_tuning, select_config_variant_interactive

try:
    import wandb

    HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
    HAVE_WANDB = False

try:
    from perf_plugins import NsysPlugin, PreemptionPlugin, PyTorchProfilerPlugin
except (ImportError, ModuleNotFoundError):
    from .perf_plugins import NsysPlugin, PreemptionPlugin, PyTorchProfilerPlugin

SCRIPT_DIR = Path(__file__).parent.resolve()
ENTRYPOINT_BOOTSTRAP = "bootstrap.py"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # pin level so nemo_run's WARNING root doesn't suppress INFO


def _filter_run_script_args(argv: List[str]) -> List[str]:
    """Drop launcher-only args before forwarding argv to the rank-local script.

    The launcher (this script) and the rank-local training scripts share one
    parser, but some args are meaningful only to the launcher and must not
    reach the rank-local scripts:

    * ``--additional_slurm_params`` — Slurm orchestration only.
    * ``--enable_vboost`` / ``--lock_gpu_freq`` — applied directly to the
      Slurm executor before submission.
    * ``--csp`` — launcher-only; selects the CSP fabric plugin. The rank-local
      script forwards unrecognized args to Hydra, which rejects ``--csp``.
    * ``--kubeflow_*`` — consumed here to build the Kubeflow TrainJob. Several
      carry JSON values whose ``{}`` / ``[]`` are brace/glob-expanded by the
      shell in the generated launch command, corrupting argv and leaking tokens
      into the training entrypoint's Hydra override parser.

    All of these take a value, passed either as ``--flag value`` (two tokens) or
    ``--flag=value`` (one token).
    """

    def _is_launcher_only(flag: str) -> bool:
        return flag in (
            "-lgc",
            "-vb",
            "--additional_slurm_params",
            "--csp",
            "--enable_vboost",
            "--lock_gpu_freq",
        ) or flag.startswith("--kubeflow_")

    filtered_args = []
    skip_next = False

    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if _is_launcher_only(arg.split("=", 1)[0]):
            skip_next = "=" not in arg
            continue
        filtered_args.append(arg)

    return filtered_args


def _default_experiment_name(
    *,
    use_recipes: bool,
    model_recipe_name: str,
    task: str,
    compute_dtype: str,
    num_gpus: int,
    gpu: str,
    config_variant: str | None,
) -> str:
    """Build a stable experiment name without importing a performance recipe."""
    if use_recipes:
        return f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}"

    fields = [task, model_recipe_name, compute_dtype, f"gpus{num_gpus}", gpu]
    if config_variant and config_variant.lower() not in {"v1", "v2"}:
        fields.append(config_variant.lower())
    return "_".join(fields)


def _build_nemorun_script(
    *,
    script_path: str,
    script_dir: str,
    args: List[str],
    kubeflow_namespace: Optional[str],
    custom_env_vars: Dict[str, str],
) -> run.Script:
    """Build the rank-local task and apply optional Kubeflow NUMA binding."""
    task = run.Script(
        path=script_path,
        entrypoint="python",
        env={"PYTHONPATH": f"{script_dir}:$PYTHONPATH"},
        args=args,
    )
    if kubeflow_namespace and _kubeflow_numa_binding_enabled(custom_env_vars):
        logger.info("Enabling per-rank GPU-local NUMA binding for Kubeflow torchrun workers")
        return _kubeflow_numa_binding_script(task)
    return task


def _build_csp_plugin(csp: str) -> Any:
    """Build a CSP plugin lazily so Slurm/login-node launch does not import Kubeflow helpers."""
    try:
        from utils.csp_plugins import EKSEnvPlugin, GKEEnvPlugin
    except (ImportError, ModuleNotFoundError):
        from .utils.csp_plugins import EKSEnvPlugin, GKEEnvPlugin

    if csp == "aws":
        return EKSEnvPlugin()
    if csp == "gcp":
        return GKEEnvPlugin()
    raise ValueError(f"Unsupported CSP plugin: {csp}")


def _calc_convergence_and_performance(**kwargs: Any) -> Any:
    """Load the evaluator only after training finishes and validation is requested."""
    try:
        from utils.evaluate import calc_convergence_and_performance
    except (ImportError, ModuleNotFoundError):
        from .utils.evaluate import calc_convergence_and_performance

    return calc_convergence_and_performance(**kwargs)


def wait_for_logs_to_settle(glob_pattern: str, timeout_s: int = 180, stable_s: int = 10, poll_s: int = 3) -> List[str]:
    """Re-glob ``glob_pattern`` and wait until the matched log files stop growing.

    On Kubeflow the all-ranks log is aggregated from the per-rank pods and can
    keep growing after ``run.run()`` returns; parsing too early sees only the
    live lm-loss rank and misses the rank-0 memory / GPU-util lines, which makes
    the golden-values check crash on a ``None`` current metric. Polling for
    size-stability lets the metric parse see the fully merged log. Returns the
    globbed paths (whatever exists at timeout).
    """
    deadline = time.time() + timeout_s
    prev_sizes: Dict[str, int] = {}
    stable_since: Optional[float] = None
    while time.time() < deadline:
        paths = sorted(glob.glob(glob_pattern))
        sizes = {p: os.path.getsize(p) for p in paths if os.path.exists(p)}
        if sizes and sizes == prev_sizes:
            stable_since = stable_since if stable_since is not None else time.time()
            if time.time() - stable_since >= stable_s:
                logger.info(f"Logs settled ({len(sizes)} file(s)) after size-stability wait")
                return paths
        else:
            stable_since = None
        prev_sizes = sizes
        time.sleep(poll_s)
    logger.warning(f"Logs did not settle within {timeout_s}s; parsing what exists")
    return sorted(glob.glob(glob_pattern))


def _cumulative_golden_values_dir(save_dir: Optional[str]) -> Optional[str]:
    """Directory holding the cross-slice cumulative golden values in the persistent dir.

    ``save_dir`` is the checkpoints dir (``…/<recipe>/checkpoints``); the cumulative
    cache lives one level up — alongside ``checkpoints/`` and ``wandb/`` — so it
    survives across resume slices on the same persistent volume. Returns ``None`` when
    no ``save_dir`` is configured (the run keeps nothing across slices).
    """
    if not save_dir:
        return None
    return os.path.join(os.path.dirname(os.path.normpath(save_dir)), "golden_values_cumulative")


def read_cumulative_golden_values(
    executor, save_dir: Optional[str], golden_values_path: str, _logger
) -> Optional[Dict[str, Any]]:
    """Read the accumulated per-step golden values carried over from earlier slices.

    Reads directly when the persistent dir is visible to the launcher (e.g. Slurm /
    shared filesystem); otherwise (Kubeflow PVC) pulls the cache through the executor's
    ``copy_from_workspace`` data-mover. Best-effort — returns ``None`` when absent
    (first slice) or unreadable.
    """
    remote_dir = _cumulative_golden_values_dir(save_dir)
    if not remote_dir:
        return None
    name = os.path.basename(golden_values_path)
    direct = os.path.join(remote_dir, name)
    if os.path.isfile(direct):
        try:
            with open(direct) as f:
                return json.load(f)
        except Exception as e:
            _logger.warning(f"Could not read cumulative golden values {direct}: {e}")
            return None
    if isinstance(executor, run.KubeflowExecutor):
        with tempfile.TemporaryDirectory() as td:
            try:
                executor.copy_from_workspace(remote_dir, td, label="gv-cache-read")
            except Exception as e:
                # Absent on the first slice — expected, not an error.
                _logger.info(f"No cumulative golden values pulled from {remote_dir}: {e}")
                return None
            matches = glob.glob(os.path.join(td, "**", name), recursive=True)
            if not matches:
                return None
            try:
                with open(matches[0]) as f:
                    return json.load(f)
            except Exception as e:
                _logger.warning(f"Could not parse cumulative golden values from PVC {remote_dir}: {e}")
    return None


def write_cumulative_golden_values(
    executor, save_dir: Optional[str], golden_values_path: str, values: Dict[str, Any], _logger
) -> None:
    """Persist the merged per-step golden values so the next resume slice extends them.

    Writes directly when the persistent dir is launcher-visible; otherwise (Kubeflow
    PVC) pushes the cache through the executor's ``copy_to_workspace`` data-mover.
    Best-effort — never raises into the run.
    """
    remote_dir = _cumulative_golden_values_dir(save_dir)
    if not remote_dir or not values:
        return
    name = os.path.basename(golden_values_path)
    if isinstance(executor, run.KubeflowExecutor):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, name), "w") as f:
                json.dump(values, f)
            try:
                executor.copy_to_workspace(td, remote_dir, label="gv-cache-write")
            except Exception as e:
                _logger.warning(f"Could not persist cumulative golden values to {remote_dir}: {e}")
        return
    try:
        os.makedirs(remote_dir, exist_ok=True)
        with open(os.path.join(remote_dir, name), "w") as f:
            json.dump(values, f)
        _logger.info(f"Wrote cumulative golden values to {os.path.join(remote_dir, name)}")
    except Exception as e:
        _logger.warning(f"Could not write cumulative golden values to {remote_dir}: {e}")


def check_training_finished(log_file_paths: List[str], is_long_convergence_run: bool = True) -> bool:
    """Check if training is finished.

    For long convergence runs, returns True when a clean-exit marker is found in the logs.
    For normal runs, returns True when the last logged iteration matches the total number
    of iterations (catches jobs that completed all training steps but hung on teardown
    before the job reached SUCCEEDED status).
    """
    found_exit_marker = False
    max_iter_seen = 0
    total_iters = None

    for log_path in log_file_paths:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if (
                    "StopIteration" in line
                    or "after training is done" in line
                    or "exiting program at iteration" in line
                    or "AssertionError: no samples left to consume:" in line
                ):
                    found_exit_marker = True

                m = re.search(r"iteration\s+(\d+)/\s*(\d+)", line)
                if m:
                    current, total = int(m.group(1)), int(m.group(2))
                    max_iter_seen = max(max_iter_seen, current)
                    total_iters = total

    if is_long_convergence_run:
        return found_exit_marker

    return total_iters is not None and max_iter_seen >= total_iters


def check_slurm_timeout(log_file_path: str) -> bool:
    """Check if Slurm job timed out."""
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()
    log = "\n".join(log_lines)
    return "DUE TO TIME LIMIT" in log


def is_flaky_failure(log_file_path: str) -> bool:
    """Check if Slurm job failed due to flaky failure."""
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()
    log = "\n".join(log_lines)

    return (
        "The server socket has failed to listen on any local network address." in log
        or "Some NCCL operations have failed or timed out." in log
        or "uncorrectable ECC error encountered" in log
        or "illegal memory access" in log
        or "illegal instruction" in log
        or "torch.distributed.DistNetworkError" in log
        or "torch.distributed.DistBackendError" in log
        or "ncclRemoteError" in log
        or "Watchdog caught collective operation timeout" in log
        or "Segmentation fault" in log
        or "found NaN in" in log
        or "For debugging consider passing CUDA_LAUNCH_BLOCKING=1" in log
        or "double free or corruption" in log
        or "Call to CUDA function failed." in log
        or "Connection reset by peer" in log
        or "invalid pointer" in log
        or "malloc(): unaligned tcache chunk detected" in log
        or "zmq.error.ZMQError: Address already in use" in log
        or "We couldn't connect to 'https://huggingface.co'" in log
        or "Unpack failed: incomplete input" in log
        or "unspecified launch failure" in log
        or "free(): corrupted unsorted chunks" in log
        or "Segfault encountered" in log
        or "Fatal glibc error" in log
        or "EOFError: No data left in file" in log
    )


def build_performance_config(args) -> Optional[Dict[str, Any]]:
    """Build performance configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with performance configuration or None if performance is disabled
    """
    config = {}

    performance_params = {
        "timing_threshold": args.timing_threshold,
        "skip_first_percent_time": args.skip_first_percent_time,
        "eval_time_start_step": args.eval_time_start_step,
        "eval_time_end_step": args.eval_time_end_step,
    }

    for key, value in performance_params.items():
        if value is not None:
            config[key] = value

    return config if config else None


def ensure_logs_where_written(log_file_paths: List[str]):
    """Ensure logs were written to disk."""
    if len(log_file_paths) == 0:
        raise FileNotFoundError(
            f"Unexpected number of log files found: {log_file_paths}. Expected at least 1, got {len(log_file_paths)}"
        )


def get_job_dir_and_status_from_run(exp_name: str):
    """Get job directory and status from run."""
    result_dict = run.Experiment.from_title(exp_name).status(return_dict=True)
    _, job_dict = list(result_dict.items())[0]
    job_dir = job_dict["local_dir"]
    job_status = str(job_dict["status"])
    return job_dir, job_status


def max_iteration_in_logs(log_file_paths: List[str]) -> int:
    """Return the highest training iteration number found across the logs (0 if none)."""
    max_iter = 0
    for log_path in log_file_paths:
        try:
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    m = re.search(r"iteration\s+(\d+)/\s*\d+", line)
                    if m:
                        max_iter = max(max_iter, int(m.group(1)))
        except OSError:
            continue
    return max_iter


def maybe_increase_n_attempts_on_flaky_failure(
    n_attempts: int,
    max_retries: int,
    is_finished_experiment: bool,
    is_long_convergence_run: bool,
    log_file_paths: List[str],
    made_progress: bool = True,
):
    """Maybe increase number of attempts.

    Long-convergence runs resume across walltime slices, so an attempt that made
    forward progress (advanced the training step count) is a legitimate resume
    and must not consume the retry budget. An attempt that made NO forward
    progress — e.g. a crash during initialization before any step — would
    otherwise resume from the same point and loop forever, so it is bounded
    exactly like a normal run's failure.
    """
    if is_finished_experiment:
        return n_attempts
    if is_long_convergence_run and made_progress:
        return n_attempts
    if any(is_flaky_failure(p) for p in log_file_paths):
        n_attempts += 1  # flaky: retry, bounded by max_retries
    else:
        # non-flaky: give up now. max_retries + 1 (not max_retries) so the outer
        # `while n_attempts <= max_retries` loop actually exits.
        n_attempts = max_retries + 1
    return n_attempts


def main(
    use_recipes: bool,
    model_family_name: str,
    model_recipe_name: str,
    task: str,
    compute_dtype: str,
    gpu: str,
    hf_token: str,
    offline: bool,
    detach: bool,
    dryrun: bool,
    enable_vboost: bool,
    lock_gpu_freq: Optional[int],
    enable_nsys: bool,
    export_nsys_sqlite: bool,
    pytorch_profiler: bool,
    moe_a2a_overlap: bool,
    wandb_key: str,
    wandb_project_name: str,
    wandb_experiment_name: str,
    wandb_entity_name: str,
    profiling_start_step: int,
    profiling_stop_step: int,
    record_memory_history: bool,
    profiling_gpu_metrics: bool,
    profiling_ranks: Optional[List[int]],
    nsys_trace: Optional[List[str]],
    nsys_extra_args: Optional[List[str]],
    nemo_home: str,
    account: str,
    partition: str,
    log_dir: str,
    gpus_per_node: int,
    time_limit: str,
    container_image: str,
    custom_mounts: List[str],
    custom_env_vars: Dict[str, str],
    custom_srun_args: List[str],
    custom_bash_cmds: List[List[str]],
    pretrained_checkpoint: Optional[str],
    save_dir: Optional[str],
    num_gpus: int,
    is_long_convergence_run: bool,
    preempt_time: int,
    additional_slurm_params: Optional[Dict[str, Any]],
    enable_pct_binding: bool,
    golden_values_path: str,
    convergence_params: Dict[str, Any],
    performance_params: Dict[str, Any],
    memory_params: Dict[str, Any],
    max_retries: int,
    retry_on_testing_failure: bool,
    kubeflow_namespace: str,
    csp: Optional[str],
    kubeflow_workdir_pvc: str,
    kubeflow_workdir_pvc_path: str,
    kubeflow_workdir_local_path: Optional[str],
    kubeflow_image_pull_secrets: List[str],
    kubeflow_volumes_json: Optional[str],
    kubeflow_volume_mounts_json: Optional[str],
    kubeflow_tolerations_json: Optional[str],
    kubeflow_affinity_json: Optional[str],
    kubeflow_env_list_json: Optional[str],
    kubeflow_extra_resource_requests_json: Optional[str],
    kubeflow_extra_resource_limits_json: Optional[str],
    kubeflow_pod_spec_overrides_json: Optional[str],
    kubeflow_container_kwargs_json: Optional[str],
    kubeflow_labels_json: Optional[str],
    kubeflow_pod_annotations_json: Optional[str],
    config_variant: str | None = None,
    gres: Optional[str] = None,
    packager: str = "git",
):
    """Sets up the experiment and runs it."""
    if (
        model_family_name in ["qwen3"]
        and model_recipe_name
        in [
            "qwen3_30b_a3b",
            "qwen3_235b_a22b",
        ]
        and task == "pretrain"
    ):
        assert hf_token or offline, (
            "Qwen3 tokenizer requires --hf_token (online) or --offline (with a pre-populated local HF cache). "
            "For --offline, pre-download the tokenizer with `huggingface-cli download` and ensure HF_HOME points "
            "to the cache directory. NullTokenizer to be used soon."
        )

    # Disable PCT binding for certain models on specific hardware/precision combos
    if (
        (
            model_family_name == "nemotronh"
            and model_recipe_name == "nemotron_3_super"
            and compute_dtype == "bf16"
            and gpu == "b300"
        )
        or (
            model_family_name == "deepseek"
            and model_recipe_name == "deepseek_v3"
            and gpu == "b300"
            and config_variant != "large_scale"
        )
        or (model_family_name == "llama" and task == "pretrain" and gpu == "b300")
        or (model_family_name == "kimi" and task == "pretrain" and gpu == "b300")
    ):
        enable_pct_binding = False

    if wandb_key is not None:
        assert wandb_project_name is not None and wandb_experiment_name is not None, (
            "both wandb_project_name and wandb_experiment_name are required for logging with WandB"
        )

    if export_nsys_sqlite and not enable_nsys:
        logger.warning("--export_nsys_sqlite was set without --enable_nsys; no Nsys SQLite export will be generated.")

    script_name = ENTRYPOINT_BOOTSTRAP
    # Keep the historical W&B-name behavior for CI. The lightweight fallback
    # deliberately avoids resolving a recipe: effective parallelism, batches,
    # and process environment are finalized by bootstrap.py in the container.
    exp_name = wandb_experiment_name or _default_experiment_name(
        use_recipes=use_recipes,
        model_recipe_name=model_recipe_name,
        task=task,
        compute_dtype=compute_dtype,
        num_gpus=num_gpus,
        gpu=gpu,
        config_variant=config_variant,
    )

    if pretrained_checkpoint is not None:
        custom_mounts.append(f"{pretrained_checkpoint}:{pretrained_checkpoint}")

    if save_dir:
        save_dir_path = Path(save_dir).resolve()
        # On Kubeflow, save_dir typically lives on a PVC that's only mounted
        # inside the trainer pod, not on the launcher pod where this script
        # runs. Creating the dir from the launcher would either fail (PVC
        # not present) or create a useless dir on the launcher's local FS.
        # Let the trainer container create its own dirs on first write.
        if kubeflow_namespace is None:
            save_dir_path.mkdir(parents=True, exist_ok=True)
            save_dir_mount = f"{save_dir_path}:{save_dir_path}"
            if save_dir_mount not in custom_mounts:
                custom_mounts.append(save_dir_mount)
                logger.info(f"Added checkpoint save directory mount for container: {save_dir_mount}")

    run_script_path = SCRIPT_DIR / script_name
    logger.info(f"Run script path: {run_script_path}")
    if not run_script_path.is_file():
        logger.error(f"Specified run script not found: {run_script_path}")
        sys.exit(1)

    # Script path + PYTHONPATH as seen INSIDE the execution environment. On
    # SLURM the launcher's SCRIPT_DIR is bind-mounted into the container (see
    # custom_mounts below), so the launcher-local path resolves there. On
    # Kubeflow the trainer pod runs the image — which ships Megatron-Bridge at
    # /opt/Megatron-Bridge — and custom_mounts do not apply, so the launcher's
    # /tmp path does not exist in the pod; use the image's script path instead.
    if kubeflow_namespace:
        in_container_script_dir = "/opt/Megatron-Bridge/scripts/performance"
        in_container_script_path = f"{in_container_script_dir}/{script_name}"
    else:
        in_container_script_dir = str(SCRIPT_DIR)
        in_container_script_path = str(run_script_path)

    custom_mounts.extend(
        [
            f"{run_script_path}:{run_script_path}",
            f"{SCRIPT_DIR}:{SCRIPT_DIR}",
        ]
    )

    if kubeflow_namespace:
        if enable_vboost or lock_gpu_freq is not None:
            logger.warning("--enable_vboost and --lock_gpu_freq are Slurm-only and will be ignored on Kubeflow.")
        executor = kubeflow_executor(
            namespace=kubeflow_namespace,
            nodes=-(num_gpus // -gpus_per_node),
            num_gpus_per_node=gpus_per_node,
            container_image=container_image,
            workdir_pvc=kubeflow_workdir_pvc,
            workdir_pvc_path=kubeflow_workdir_pvc_path,
            workdir_local_path=kubeflow_workdir_local_path,
            train_job_basename=f"mb-{model_recipe_name}",
            image_pull_secrets=kubeflow_image_pull_secrets,
            custom_env_vars=custom_env_vars,
            wandb_key=wandb_key,
            hf_token=hf_token,
            volumes=json.loads(kubeflow_volumes_json) if kubeflow_volumes_json else None,
            volume_mounts=json.loads(kubeflow_volume_mounts_json) if kubeflow_volume_mounts_json else None,
            tolerations=json.loads(kubeflow_tolerations_json) if kubeflow_tolerations_json else None,
            affinity=json.loads(kubeflow_affinity_json) if kubeflow_affinity_json else None,
            env_list=json.loads(kubeflow_env_list_json) if kubeflow_env_list_json else None,
            extra_resource_requests=(
                json.loads(kubeflow_extra_resource_requests_json) if kubeflow_extra_resource_requests_json else None
            ),
            extra_resource_limits=(
                json.loads(kubeflow_extra_resource_limits_json) if kubeflow_extra_resource_limits_json else None
            ),
            pod_spec_overrides=(
                json.loads(kubeflow_pod_spec_overrides_json) if kubeflow_pod_spec_overrides_json else None
            ),
            container_kwargs=json.loads(kubeflow_container_kwargs_json) if kubeflow_container_kwargs_json else None,
            labels=json.loads(kubeflow_labels_json) if kubeflow_labels_json else None,
            pod_annotations=(json.loads(kubeflow_pod_annotations_json) if kubeflow_pod_annotations_json else None),
        )
    else:
        executor = slurm_executor(
            gpu=gpu,
            account=account,
            partition=partition,
            log_dir=log_dir,
            nodes=-(num_gpus // -gpus_per_node),
            num_gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            container_image=container_image,
            custom_mounts=custom_mounts,
            custom_env_vars=custom_env_vars,
            custom_srun_args=custom_srun_args,
            custom_bash_cmds=custom_bash_cmds,
            gres=gres,
            hf_token=hf_token,
            offline=offline,
            nemo_home=nemo_home,
            additional_slurm_params=additional_slurm_params,
            wandb_key=wandb_key,
            packager=packager,
            enable_pct_binding=enable_pct_binding,
        )
        configure_slurm_gpu_tuning(
            executor,
            enable_vboost=enable_vboost,
            lock_gpu_freq=lock_gpu_freq,
        )

    plugins = []

    # Long-convergence runs are split across walltime slices and resume from the last
    # checkpoint each slice. Without a preemption signal, Slurm hard-kills the slice at
    # the time limit before a checkpoint is written, so no progress persists and the
    # resume loop stalls. The PreemptionPlugin sends SIGTERM `preempt_time` seconds
    # before the limit and enables the training exit handler, so each slice saves and
    # exits gracefully and the next slice resumes from it (inert off Slurm).
    if is_long_convergence_run:
        plugins.append(PreemptionPlugin(preempt_time=preempt_time, enable_exit_handler=True))

    # CSP fabric plugins (Kubeflow only; inert on Slurm via their isinstance guard):
    # aws -> EKSEnvPlugin (EFA), gcp -> GKEEnvPlugin (gIB). Networking/fabric only;
    # architecture and performance settings stay in the recipe.
    if csp is not None:
        plugins.append(_build_csp_plugin(csp))

    if enable_nsys:
        if nsys_trace is None:
            logger.warning("Using `cuda-sw` trace mode for profiling")
            logger.warning("Profiling results might not be accurate due to software tracing limitations.")
            # TODO: Remove this once the associated functional issues are resolved.
            nsys_trace = ["cuda-sw", "nvtx"]
        plugins.append(
            NsysPlugin(
                profile_step_start=profiling_start_step,
                profile_step_end=profiling_stop_step,
                nsys_gpu_metrics=profiling_gpu_metrics,
                profile_ranks=profiling_ranks,
                nsys_trace=nsys_trace,
                nsys_extra_args=nsys_extra_args,
                export_sqlite=export_nsys_sqlite,
            )
        )
    if pytorch_profiler:
        plugins.append(
            PyTorchProfilerPlugin(
                profile_step_start=profiling_start_step,
                profile_step_end=profiling_stop_step,
                profile_ranks=profiling_ranks,
                record_memory_history=record_memory_history,
            )
        )

    nemorun_script = _build_nemorun_script(
        script_path=in_container_script_path,
        script_dir=in_container_script_dir,
        args=_filter_run_script_args(sys.argv[1:]),
        kubeflow_namespace=kubeflow_namespace,
        custom_env_vars=custom_env_vars,
    )

    logger.info("Will launch the following command with Nemo-Run: %s", " ".join(nemorun_script.to_command()))

    is_finished_experiment = False  # An experiment might consist of multiple training runs, due to restarts.
    is_testing_passed = False  # Whether the testing passed convergence and performance validation.
    error_msg = None
    n_attempts = 0
    # The K8s TrainJob name is generated independently by the executor
    # (mb-<model>-<uuid6>), so the experiment name is no longer length-bound by
    # k8s — keep it full and descriptive for nemo-run bookkeeping + wandb.
    wandb_run_id = None
    max_iter_so_far = 0
    while n_attempts <= max_retries:
        while is_finished_experiment is False:
            if HAVE_WANDB:
                wandb_run_id = (
                    (wandb_run_id or wandb.util.generate_id()) if is_long_convergence_run else wandb.util.generate_id()
                )
                executor.env_vars.update(
                    {
                        "WANDB_RUN_ID": wandb_run_id,
                        "WANDB_RESUME": "allow",
                    }
                )
            if wandb_key is not None:
                executor.env_vars["WANDB_API_KEY"] = wandb_key

            run.run(
                nemorun_script,
                executor=executor,
                plugins=plugins,
                dryrun=dryrun,
                detach=detach,
                name=exp_name,
            )
            if dryrun:
                logger.info("dryrun requested: exiting")
                return

            job_dir, job_status = get_job_dir_and_status_from_run(exp_name)

            terminal_failure = job_status not in ["SUCCEEDED", "SUBMITTED", "PENDING", "RUNNING"]

            if detach:
                is_finished_experiment = True
                is_testing_passed = True
                break

            log_file_paths = list(Path(f"{job_dir}").glob("log*.out"))
            ensure_logs_where_written(log_file_paths)

            is_finished_experiment = (
                check_training_finished(log_file_paths, is_long_convergence_run=True)
                if is_long_convergence_run
                else (
                    job_status == "SUCCEEDED" or check_training_finished(log_file_paths, is_long_convergence_run=False)
                )
            )

            # Raise on terminal failures only if training didn't actually complete —
            # a job can time out due to hanging on teardown after all steps finished.
            # Long-convergence runs are intentionally split across walltime slices
            # (slurm cancels at TIME_LIMIT, we resume from the saved checkpoint on
            # the next outer-loop iteration), so the walltime-cap path is expected
            # and must not raise.
            if terminal_failure and not is_finished_experiment and not is_long_convergence_run:
                raise Exception(f"Experiment failed for {exp_name} with status: {job_status}.")

            # Forward-progress check: a long-convergence attempt that advanced the
            # training step count is a legitimate walltime-slice resume; one that
            # did not (e.g. crashed in init before any step) must be bounded so it
            # cannot resume-loop forever.
            attempt_max_iter = max_iteration_in_logs(log_file_paths)
            made_progress = attempt_max_iter > max_iter_so_far
            max_iter_so_far = max(max_iter_so_far, attempt_max_iter)

            n_attempts = maybe_increase_n_attempts_on_flaky_failure(
                n_attempts=n_attempts,
                max_retries=max_retries,
                is_finished_experiment=is_finished_experiment,
                is_long_convergence_run=is_long_convergence_run,
                log_file_paths=log_file_paths,
                made_progress=made_progress,
            )

            if not is_finished_experiment and n_attempts <= max_retries:
                logger.error(f"Starting attempt {n_attempts + 1} of {max_retries + 1} for {exp_name}")

            if not is_finished_experiment:
                break

        if is_finished_experiment is True and detach is False:
            log_glob = f"{get_nemorun_home()}/experiments/{exp_name}/{exp_name}_*/{exp_name}/log*.out"
            log_paths = sorted(glob.glob(log_glob))

            logger.info(f"Starting convergence check for {model_family_name}_{model_recipe_name}")

            wandb_run = None
            # Bug 2 fix: wandb online mode redirects fd 2 at the OS level via os.dup2(),
            # making all stderr writes invisible. Grab a private copy of fd 2 *before*
            # wandb.init() so our StreamHandler bypasses wandb's capture pipe.
            _dup_file = os.fdopen(os.dup(2), "w", buffering=1)
            _dup_handler = logging.StreamHandler(_dup_file)
            _dup_handler.setLevel(logging.DEBUG)
            _dup_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            logger.addHandler(_dup_handler)

            if HAVE_WANDB and wandb_key:
                wandb_run = wandb.init(
                    project=wandb_project_name,
                    entity=wandb_entity_name,
                    id=wandb_run_id,
                    resume="allow",
                )

                # The K8s all-ranks log is aggregated from the per-rank pods and
                # can keep growing after run.run() returns; parse too early and the
                # rank-0 memory / GPU-util lines are missing (only the live lm-loss
                # rank is present). Wait for the files to stop growing, then re-glob.
                log_paths = wait_for_logs_to_settle(log_glob)

                # Long-convergence runs resume across walltime slices; this slice's
                # logs only cover its own steps. Carry the per-step values recorded by
                # earlier slices (from the persistent dir) into this slice so the
                # written golden values span the whole run rather than just the tail.
                prior_values = (
                    read_cumulative_golden_values(executor, save_dir, golden_values_path, logger)
                    if is_long_convergence_run
                    else None
                )

                is_testing_passed, error_msg, merged_values = _calc_convergence_and_performance(
                    model_family_name=model_family_name,
                    model_recipe_name=model_recipe_name,
                    assets_dir=os.path.join(job_dir, exp_name),
                    log_paths=log_paths,
                    loss_metric="lm loss",
                    timing_metric="elapsed time per iteration (ms)",
                    alloc_metric="alloc",
                    max_alloc_metric="max_alloc",
                    golden_values_path=golden_values_path,
                    convergence_config=convergence_params,
                    performance_config=performance_params,
                    memory_config=memory_params,
                    wandb_run=wandb_run,
                    _logger=logger,
                    prior_values=prior_values,
                )

                # Persist the merged per-step values so the next resume slice extends
                # them instead of overwriting with its own partial curve.
                if is_long_convergence_run:
                    write_cumulative_golden_values(executor, save_dir, golden_values_path, merged_values, logger)

                wandb_run.finish()
                wandb.teardown(exit_code=int(not is_testing_passed))

            logger.removeHandler(_dup_handler)
            _dup_file.close()

            if not retry_on_testing_failure:
                # Train-then-test: each experiment runs the convergence/perf
                # check exactly once. Force-exit the outer loop here — the
                # post-loop block raises AssertionError(error_msg) when testing
                # failed. Without this, a long-convergence run whose testing
                # fails would otherwise spin re-evaluating the same finished
                # training forever until the slurm walltime cap killed it.
                n_attempts = max_retries + 1
            elif not is_testing_passed:
                # Opt-in retry: redo training + testing in the next outer-loop
                # iteration. Bump n_attempts so the outer loop bounds the
                # retries. Reset is_finished_experiment so the inner training
                # loop runs again; clear wandb_run_id so the retry logs to a
                # fresh wandb run instead of resuming the failed one.
                n_attempts += 1
                if n_attempts <= max_retries:
                    is_finished_experiment = False
                    wandb_run_id = None
                    logger.error(
                        f"Testing failed; retrying full train+test "
                        f"({n_attempts + 1} of {max_retries + 1}) for {exp_name}"
                    )

        if is_finished_experiment and is_testing_passed:
            break

    if not is_testing_passed and error_msg is not None:
        raise AssertionError(error_msg)
    if is_testing_passed and error_msg is not None:
        logger.warning(error_msg)

    if not is_finished_experiment:
        raise Exception("Megatron-Bridge CI test job failed")
    elif is_finished_experiment and not detach:
        logger.info("Megatron-Bridge CI test job completed successfully!")


if __name__ == "__main__":
    parser = parse_cli_args()
    args, unknown_args = parser.parse_known_args()

    gpus_per_node = args.gpus_per_node
    if gpus_per_node is None:
        if args.gpu in NUM_GPUS_PER_NODE_MAP:
            gpus_per_node = NUM_GPUS_PER_NODE_MAP[args.gpu]
        else:
            raise ValueError(
                f"Invalid GPU type: {args.gpu}. Please use one of the following: {NUM_GPUS_PER_NODE_MAP.keys()}"
            )

    assert not (args.enable_nsys and args.pytorch_profiler), (
        "Both NSys and PyTorch profiler cannot be enabled at the same time"
    )

    # probably better to use parser.parse_args() and make unknowns an error,
    # but for now we'll just issue a warning.
    if unknown_args:
        logger.warning(f"Ignoring unrecognized arguments: {' '.join(unknown_args)}")

    env = dict(args.env or [])
    custom_env_vars = args.custom_env_vars
    custom_env_vars.update(env)

    # Handle --list_config_variants: show available variants and interactively select
    config_variant = args.config_variant
    if args.list_config_variants:
        config_variant = select_config_variant_interactive(
            model_family_name=args.model_family_name,
            model_recipe_name=args.model_recipe_name,
            gpu=args.gpu,
            compute_dtype=args.compute_dtype,
            task=args.task,
        )

    main(
        use_recipes=args.use_recipes,
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        compute_dtype=args.compute_dtype,
        gpu=args.gpu,
        hf_token=args.hf_token,
        offline=args.offline,
        detach=args.detach,
        dryrun=args.dryrun,
        enable_vboost=args.enable_vboost,
        lock_gpu_freq=args.lock_gpu_freq,
        enable_nsys=args.enable_nsys,
        export_nsys_sqlite=args.export_nsys_sqlite,
        pytorch_profiler=args.pytorch_profiler,
        moe_a2a_overlap=args.moe_a2a_overlap,
        wandb_key=args.wandb_key,
        wandb_project_name=args.wandb_project_name,
        wandb_experiment_name=args.wandb_experiment_name,
        wandb_entity_name=args.wandb_entity_name,
        profiling_start_step=args.profiling_start_step,
        profiling_stop_step=args.profiling_stop_step,
        record_memory_history=args.record_memory_history,
        profiling_gpu_metrics=args.profiling_gpu_metrics,
        profiling_ranks=args.profiling_ranks,
        nsys_trace=args.nsys_trace,
        nsys_extra_args=args.nsys_extra_args,
        nemo_home=args.nemo_home,
        account=args.account,
        partition=args.partition,
        log_dir=args.log_dir,
        gpus_per_node=gpus_per_node,
        time_limit=args.time_limit,
        container_image=args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=custom_env_vars,
        custom_srun_args=args.custom_srun_args,
        custom_bash_cmds=args.custom_bash_cmds,
        pretrained_checkpoint=args.pretrained_checkpoint,
        save_dir=args.save_dir,
        num_gpus=args.num_gpus,
        is_long_convergence_run=args.is_long_convergence_run,
        preempt_time=args.preempt_time,
        additional_slurm_params=args.additional_slurm_params,
        enable_pct_binding=args.enable_pct_binding,
        golden_values_path=args.golden_values_path,
        convergence_params={
            "correlation_threshold": args.correlation_threshold,
            "high_loss_tolerance": args.high_loss_tolerance,
            "medium_loss_tolerance": args.medium_loss_tolerance,
            "low_loss_tolerance": args.low_loss_tolerance,
            "final_loss_tolerance": args.final_loss_tolerance,
            "max_outlier_ratio": args.max_outlier_ratio,
            "outlier_threshold": args.outlier_threshold,
            "skip_first_percent_loss": args.skip_first_percent_loss,
        },
        performance_params={
            "timing_threshold": args.timing_threshold,
            "skip_first_percent_time": args.skip_first_percent_time,
            "eval_time_start_step": args.eval_time_start_step,
            "eval_time_end_step": args.eval_time_end_step,
        },
        memory_params={
            "memory_threshold": args.memory_threshold,
        },
        max_retries=args.max_retries,
        retry_on_testing_failure=args.retry_on_testing_failure,
        kubeflow_namespace=args.kubeflow_namespace,
        csp=args.csp,
        kubeflow_workdir_pvc=args.kubeflow_workdir_pvc,
        kubeflow_workdir_pvc_path=args.kubeflow_workdir_pvc_path,
        kubeflow_workdir_local_path=args.kubeflow_workdir_local_path,
        kubeflow_image_pull_secrets=args.kubeflow_image_pull_secrets,
        kubeflow_volumes_json=args.kubeflow_volumes_json,
        kubeflow_volume_mounts_json=args.kubeflow_volume_mounts_json,
        kubeflow_tolerations_json=args.kubeflow_tolerations_json,
        kubeflow_affinity_json=args.kubeflow_affinity_json,
        kubeflow_env_list_json=args.kubeflow_env_list_json,
        kubeflow_extra_resource_requests_json=args.kubeflow_extra_resource_requests_json,
        kubeflow_extra_resource_limits_json=args.kubeflow_extra_resource_limits_json,
        kubeflow_pod_spec_overrides_json=args.kubeflow_pod_spec_overrides_json,
        kubeflow_container_kwargs_json=args.kubeflow_container_kwargs_json,
        kubeflow_labels_json=args.kubeflow_labels_json,
        kubeflow_pod_annotations_json=args.kubeflow_pod_annotations_json,
        config_variant=config_variant,
        gres=args.gres,
        packager=args.packager,
    )
