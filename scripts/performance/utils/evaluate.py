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
import json
import logging
import math
import os
import pathlib
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


try:
    import numpy as np

    HAVE_NUMPY = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMPY = False

try:
    import wandb

    HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
    HAVE_WANDB = False


# Setup logging
logger = logging.getLogger(__name__)


def get_metrics_from_logfiles(log_paths: List[str], metric: str):
    """
    Parse training log files and extract metrics.

    Args:
        log_paths: Paths to the log files
        metric: Metric name to extract

    Returns:
        For scalar metrics (alloc, max_alloc): float or None
        For per-step metrics: Dict[str, float] keyed by 0-indexed step number
    """
    patterns = {
        "iteration": r"iteration\s+(\d+)/\s+\d+",
        "elapsed time per iteration (ms)": r"elapsed time per iteration \(ms\):\s+([\d.]+)",
        "lm loss": r"lm loss:\s+([\d.E+\-]+)",
        "GPU utilization": r"GPU utilization:\s+([\d.]+)",
        "step time": r"Step Time :\s+([\d.]+)s",
        "grad norm": r"grad norm:\s+([\d.]+|nan|inf)",
        "alloc": r"mem-allocated-gigabytes:\s*([\d\.]+)",
        "max_alloc": r"mem-max-allocated-gigabytes:\s*([\d\.]+)",
    }

    metrics: Dict[str, List] = {k: [] for k in patterns}
    all_lines = []
    handles = []
    for log_path in list(set(log_paths)):
        if "allranks" in log_path:
            continue
        logger.info(f"Reading log file: {log_path}")
        handles.append(open(log_path))

    try:
        for lines in zip(*handles):
            for line in lines:
                all_lines.append(line)
    finally:
        for f in handles:
            f.close()

    for line in all_lines:
        for metric_name, pattern in patterns.items():
            if match := re.search(pattern, line):
                metrics[metric_name].append(float(match.group(1)))

    # Scalar metrics: return first occurrence only
    if metric in ("alloc", "max_alloc"):
        values = metrics[metric]
        return values[0] if values else None

    # Per-step metrics: postprocess into step-keyed dict
    # iteration N announces that step N-1 just completed
    steps = [int(i) - 1 for i in metrics["iteration"]]
    values = metrics[metric]
    if len(values) != len(steps):
        logger.warning(
            f"Metric '{metric}': found {len(values)} values for {len(steps)} iterations; some steps may be missing"
        )
    return {str(step): value for step, value in zip(steps, values)}


def validate_convergence(
    current_values: "np.ndarray",
    golden_values: "np.ndarray",
    steps: List[str],
    logger: logging.Logger,
    wandb_run: Optional["wandb.Run"] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Comprehensive loss curve convergence validation using multiple metrics.

    This function implements a robust multi-metric approach to validate that
    the current training run produces statistically equivalent results to the
    golden reference, accounting for training variability and different loss ranges.

    Args:
        current_values: Current training loss values
        golden_values: Golden reference loss values
        steps: Training step identifiers
        logger: Logger instance for detailed reporting
        config: Optional configuration dict with custom thresholds

    Returns:
        Dict with 'passed' boolean and detailed results
    """

    # Default configuration
    default_config = {
        # Statistical significance threshold
        "correlation_threshold": 0.95,
        # Point-wise tolerances (adaptive based on loss magnitude)
        "high_loss_tolerance": 0.10,  # 10% for loss > 2.0
        "medium_loss_tolerance": 0.05,  # 5% for loss 0.5-2.0
        "low_loss_tolerance": 0.02,  # 2% for loss < 0.5
        # Curve shape metrics
        "final_loss_tolerance": 0.03,  # 3% for final loss
        # Outlier handling
        "max_outlier_ratio": 0.1,  # Max 10% of points can be outliers
        "outlier_threshold": 3.0,  # 3-sigma outlier detection
        # Loss curve analysis
        "skip_first_percent_loss": 0.0,  # Percentage of loss points to skip from beginning
    }

    if config:
        default_config.update(config)
    config = default_config

    results = {"passed": True, "failed_metrics": [], "summary": "", "details": "", "metrics": {}}

    logger.info("Starting comprehensive loss curve validation...")

    # 1. SKIP FIRST PERCENT OF LOSS POINTS (if configured)
    skip_first_n_percent = max(0, int(len(current_values) * config["skip_first_percent_loss"]))
    if skip_first_n_percent > 0:
        current_values = current_values[skip_first_n_percent:]
        golden_values = golden_values[skip_first_n_percent:]
        steps = steps[skip_first_n_percent:]
        logger.info(f"Skipped first {skip_first_n_percent} loss points for analysis")

    # 2. STATISTICAL CORRELATION TEST
    correlation = np.corrcoef(current_values, golden_values)[0, 1]
    results["metrics"]["correlation"] = correlation

    if correlation < config["correlation_threshold"]:
        results["passed"] = False
        results["failed_metrics"].append("correlation")
        logger.warning(f"Correlation {correlation:.4f} < threshold {config['correlation_threshold']}")
    else:
        logger.info(f"✓ Correlation test passed: {correlation:.4f} >= {config['correlation_threshold']:.4f}")

    # 3. ADAPTIVE POINT-WISE TOLERANCE CHECK
    point_wise_failures = []
    for i, (current_val, golden_val) in enumerate(zip(current_values, golden_values)):
        # Determine tolerance based on loss magnitude
        if golden_val > 2.0:
            tolerance = config["high_loss_tolerance"]
        elif golden_val > 0.5:
            tolerance = config["medium_loss_tolerance"]
        else:
            tolerance = config["low_loss_tolerance"]

        # Calculate relative difference
        if golden_val != 0:
            relative_diff = abs(current_val - golden_val) / abs(golden_val)
        else:
            relative_diff = abs(current_val) if current_val != 0 else 0

        if relative_diff > tolerance:
            point_wise_failures.append(
                {
                    "step": steps[i],
                    "current": current_val,
                    "golden": golden_val,
                    "relative_diff": relative_diff,
                    "tolerance": tolerance,
                }
            )

    results["metrics"]["point_wise_failures"] = len(point_wise_failures)
    results["metrics"]["total_points"] = len(current_values)

    if len(point_wise_failures) > 0:
        failure_ratio = len(point_wise_failures) / len(current_values)
        if failure_ratio > config["max_outlier_ratio"]:
            results["passed"] = False
            results["failed_metrics"].append("point_wise_tolerance")
            logger.warning(
                f"Point-wise failures: {len(point_wise_failures)}/{len(current_values)} "
                f"({failure_ratio:.2%}) > max allowed {config['max_outlier_ratio']:.2%}"
            )
        else:
            logger.info(f"✓ Point-wise tolerance: {len(point_wise_failures)} outliers within acceptable range")
    else:
        logger.info("✓ Point-wise tolerance: All points within tolerance")

    # 4. FINAL LOSS VALIDATION
    final_current = current_values[-1]
    final_golden = golden_values[-1]
    final_diff = abs(final_current - final_golden) / final_golden if final_golden != 0 else abs(final_current)

    results["metrics"]["final_loss_current"] = final_current
    results["metrics"]["final_loss_golden"] = final_golden
    results["metrics"]["final_loss_diff"] = final_diff

    if final_diff > config["final_loss_tolerance"]:
        results["passed"] = False
        results["failed_metrics"].append("final_loss")
        logger.warning(f"Final loss difference {final_diff:.4f} > threshold {config['final_loss_tolerance']}")
    else:
        logger.info(f"✓ Final loss validation passed: {final_diff:.4f} <= {config['final_loss_tolerance']:.4f}")

    # 5. OUTLIER DETECTION (3-sigma rule)
    residuals = current_values - golden_values
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    outliers = np.abs(residuals - mean_residual) > config["outlier_threshold"] * std_residual
    outlier_count = np.sum(outliers)

    results["metrics"]["outlier_count"] = outlier_count
    results["metrics"]["outlier_ratio"] = outlier_count / len(current_values)

    if outlier_count / len(current_values) > config["max_outlier_ratio"]:
        results["passed"] = False
        results["failed_metrics"].append("outliers")
        logger.warning(
            f"Too many outliers: {outlier_count}/{len(current_values)} "
            f"({outlier_count / len(current_values):.2%}) > max {config['max_outlier_ratio']:.2%}"
        )
    else:
        logger.info(f"✓ Outlier detection passed: {outlier_count} outliers <= {config['max_outlier_ratio']:.2%}")

    # Generate summary
    if results["passed"]:
        results["summary"] = "All convergence tests passed"
        logger.info("🎉 All convergence validation tests PASSED!")
    else:
        results["summary"] = f"Failed {len(results['failed_metrics'])} out of 5 validation tests"
        logger.error(f"❌ Convergence validation FAILED: {results['summary']}")

        # Add detailed failure information
        details = []
        if point_wise_failures:
            details.append(f"Point-wise failures ({len(point_wise_failures)}):")
            for failure in point_wise_failures[:5]:  # Show first 5 failures
                details.append(
                    f"  Step {failure['step']}: {failure['current']:.6f} vs {failure['golden']:.6f} "
                    f"(diff: {failure['relative_diff']:.4f})"
                )
            if len(point_wise_failures) > 5:
                details.append(f"  ... and {len(point_wise_failures) - 5} more")

        results["details"] = "\n".join(details)

    if wandb_run is not None:
        wandb_run.summary["convergence_passed"] = results["passed"]
        wandb_run.summary["convergence_failed_metrics"] = ",".join(results["failed_metrics"])

    for key, value in results["metrics"].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")

    return results


def validate_performance(
    current_gpu_util_values: "np.ndarray",
    golden_gpu_util_values: "np.ndarray",
    steps: List[str],
    logger: logging.Logger,
    wandb_run: Optional["wandb.Run"] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Validate GPU utilization performance metrics.

    Uses signed difference to detect both regressions (GPU util dropped) and
    unexpected improvements (GPU util jumped suspiciously high).

    Args:
        current_gpu_util_values: Current GPU utilization values per step
        golden_gpu_util_values: Golden reference GPU utilization values per step
        steps: Training step identifiers
        logger: Logger instance for detailed reporting
        wandb_run: Optional wandb run object
        config: Optional configuration dict with custom thresholds

    Returns:
        Dict with 'passed' boolean and detailed results
    """
    RED, GREEN, RESET = "\033[31m", "\033[32m", "\033[0m"

    default_config = {
        "skip_first_percent_time": 0.1,  # Percentage of iterations to skip from beginning
        "timing_threshold": 0.05,  # 5% threshold for GPU util validation
    }

    if config:
        default_config.update(config)
    config = default_config

    # Discard first N% of iterations for stable comparison
    start = config.get("eval_time_start_step")
    if start is None:
        start = max(1, int(len(steps) * config["skip_first_percent_time"]))
    end = config.get("eval_time_end_step")
    current_stable = current_gpu_util_values[start:end]
    golden_stable = golden_gpu_util_values[start:end]

    current_avg = float(np.nanmean(current_stable))
    golden_avg = float(np.nanmean(golden_stable))

    # Signed diff: positive = improvement (higher util), negative = regression (lower util)
    signed_diff = (current_avg - golden_avg) / golden_avg if golden_avg != 0 else 0.0

    is_regression = signed_diff < -config["timing_threshold"]
    is_improvement = signed_diff > config["timing_threshold"]

    logger.info(
        f"GPU utilization comparison (steps [{start}:{end if end is not None else len(steps)}] "
        f"out of {len(steps)} total):"
    )
    logger.info(f"  Current average GPU util: {current_avg:.4f}%")
    logger.info(f"  Golden average GPU util: {golden_avg:.4f}%")
    logger.info(f"  Signed diff: {signed_diff * 100:.2f}%")
    logger.info(f"  Threshold: ±{config['timing_threshold'] * 100:.1f}%")

    results = {"passed": True, "failed_metrics": [], "summary": "", "details": "", "metrics": {}}

    results["metrics"]["current_avg_gpu_util"] = current_avg
    results["metrics"]["golden_avg_gpu_util"] = golden_avg
    results["metrics"]["signed_diff"] = signed_diff
    results["metrics"]["threshold"] = config["timing_threshold"]

    if is_regression:
        logger.warning(
            f"{RED}REGRESSION{RESET}: GPU util dropped {abs(signed_diff) * 100:.2f}% "
            f"(current={current_avg:.4f}%, golden={golden_avg:.4f}%, "
            f"threshold={config['timing_threshold'] * 100:.1f}%)"
        )
        results["passed"] = False
        results["failed_metrics"].append("gpu_util_regression")
        results["metrics"]["direction"] = "regression"
    elif is_improvement:
        logger.warning(
            f"{GREEN}UNEXPECTED IMPROVEMENT{RESET}: GPU util jumped {signed_diff * 100:.2f}% "
            f"(current={current_avg:.4f}%, golden={golden_avg:.4f}%, "
            f"threshold={config['timing_threshold'] * 100:.1f}%)"
        )
        results["passed"] = False
        results["failed_metrics"].append("gpu_util_improvement")
        results["metrics"]["direction"] = "improvement"
    else:
        logger.info(
            f"✓ GPU utilization passed: {signed_diff * 100:.2f}% within "
            f"±{config['timing_threshold'] * 100:.1f}% threshold"
        )
        results["metrics"]["direction"] = "pass"

    if results["passed"]:
        results["summary"] = "All performance tests passed"
        logger.info("🎉 All performance tests PASSED!")
    else:
        direction = results["metrics"]["direction"]
        results["summary"] = f"Failed 1 out of 1 test ({direction})"
        logger.error(f"❌ Performance validation FAILED: {results['summary']}")

    if wandb_run is not None:
        wandb_run.summary["current_avg_gpu_util"] = current_avg
        wandb_run.summary["golden_avg_gpu_util"] = golden_avg
        wandb_run.summary["gpu_util_signed_diff"] = signed_diff
        wandb_run.summary["gpu_util_threshold"] = config["timing_threshold"]
        wandb_run.summary["performance_passed"] = results["passed"]

    for key, value in results["metrics"].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")

    return results


def validate_memory(
    golden_alloc: float,
    current_alloc: float,
    golden_max_alloc: float,
    current_max_alloc: float,
    logger: logging.Logger,
    wandb_run: Optional["wandb.Run"] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Validate memory metrics.
    """

    default_config = {
        "memory_threshold": 0.05,
    }

    if config:
        default_config.update(config)
    config = default_config

    # Calculate memory differences
    max_alloc_diff = (
        abs(current_max_alloc - golden_max_alloc) / golden_max_alloc
        if golden_max_alloc != 0
        else abs(current_max_alloc)
    )
    alloc_diff = abs(current_alloc - golden_alloc) / golden_alloc if golden_alloc != 0 else abs(current_alloc)

    logger.info(f"Max alloc difference: {max_alloc_diff * 100:.2f}%")
    logger.info(f"Memory threshold: {config['memory_threshold'] * 100:.1f}%")
    logger.info(f"Current max alloc: {current_max_alloc}")
    logger.info(f"Golden max alloc: {golden_max_alloc}")
    logger.info(f"Alloc difference: {alloc_diff * 100:.2f}%")
    logger.info(f"Current alloc: {current_alloc}")
    logger.info(f"Golden alloc: {golden_alloc}")

    results = {"passed": True, "failed_metrics": [], "summary": "", "details": "", "metrics": {}}

    results["metrics"]["current_max_alloc"] = current_max_alloc
    results["metrics"]["golden_max_alloc"] = golden_max_alloc
    results["metrics"]["max_alloc_diff"] = max_alloc_diff
    results["metrics"]["current_alloc"] = current_alloc
    results["metrics"]["golden_alloc"] = golden_alloc
    results["metrics"]["alloc_diff"] = alloc_diff
    results["metrics"]["threshold"] = config["memory_threshold"]

    if max_alloc_diff > config["memory_threshold"]:
        logger.warning(
            f"Memory validation FAILED: {max_alloc_diff * 100:.2f}% > {config['memory_threshold'] * 100:.1f}%"
        )
        # Add to memory result
        results["passed"] = False
        results["failed_metrics"].append("max_alloc")
    else:
        logger.info(
            f"✓ Max Memory allocation passed: {max_alloc_diff * 100:.2f}% <= {config['memory_threshold'] * 100:.1f}%"
        )

    if alloc_diff > config["memory_threshold"]:
        logger.warning(f"Alloc validation FAILED: {alloc_diff * 100:.2f}% > {config['memory_threshold'] * 100:.1f}%")
        results["passed"] = False
        results["failed_metrics"].append("alloc")
    else:
        logger.info(f"✓ Alloc validation passed: {alloc_diff * 100:.2f}% <= {config['memory_threshold'] * 100:.1f}%")

    # Generate summary
    if results["passed"]:
        results["summary"] = "All memory validation tests passed"
        logger.info("🎉 All memory validation tests PASSED!")
    else:
        results["summary"] = f"Failed {len(results['failed_metrics'])} out of 2 validation tests"
        logger.error(f"❌ Memory validation FAILED: {results['summary']}")

    if wandb_run is not None:
        wandb_run.summary["memory_passed"] = results["passed"]
        wandb_run.summary["memory_failed_metrics"] = ",".join(results["failed_metrics"])

    for key, value in results["metrics"].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")

    return results


def write_golden_values_to_disk(
    current_values: Dict[str, Any], golden_values_path: str, wandb_run: Optional["wandb.Run"] = None
):
    """
    Write golden values to a file.
    """
    os.makedirs(os.path.dirname(golden_values_path), exist_ok=True)
    with open(golden_values_path, "w") as f:
        json.dump(current_values, f)

    if wandb_run is not None:
        artifact = wandb.Artifact("golden_values", type="dataset")
        with artifact.new_file("golden_values.json", "w") as f:
            json.dump({datetime.now().strftime("%m.%d.%y"): current_values}, f)
        wandb_run.log_artifact(artifact)

    logger.info(f"Golden values were saved for {golden_values_path}.")


def calc_convergence_and_performance(
    model_family_name: str,
    model_recipe_name: str,
    assets_dir: str,
    log_paths: List[str],
    loss_metric: str,
    timing_metric: str,
    alloc_metric: str,
    max_alloc_metric: str,
    golden_values_path: str,
    convergence_config: Dict[str, Any],
    performance_config: Dict[str, Any],
    memory_config: Dict[str, Any],
    wandb_run: Optional["wandb.Run"] = None,
    _logger: logging.Logger = None,
):
    """
    Calculate convergence metrics and validate against golden values.

    Args:
        model_family_name: Type of model (e.g., 'llama3', 'qwen3')
        model_recipe_name: Recipe name of model (e.g., 'llama3_70b_pretrain_config', 'qwen3_30b_a3b_pretrain_config')
        cluster: Cluster name
        assets_dir: Directory containing job results
        loss_metric: Loss metric to extract (default: 'lm loss')
        timing_metric: Timing metric to extract (default: 'iteration-time')
        golden_values_path: Path to golden values directory
        timing_threshold: Threshold for step timing validation
        skip_first_percent_time: Percentage of iterations to skip from the beginning for timing comparison
        convergence_config: Optional configuration dict for loss curve convergence validation.
            Can override: correlation_threshold, high_loss_tolerance, medium_loss_tolerance,
            low_loss_tolerance, final_loss_tolerance, max_outlier_ratio, outlier_threshold,
            skip_first_percent_loss
        wandb_run: An optional wandb run object to log metrics to
        _logger: Logger to use; defaults to this module's logger if not provided
    """
    _logger = _logger or logger

    if not HAVE_WANDB:
        raise ImportError("wandb is required for calculating perf and convergence metrics")

    if not HAVE_NUMPY:
        raise ImportError("numpy is required for calculating perf and convergence metrics")

    current_train_loss = get_metrics_from_logfiles(log_paths, loss_metric)
    current_iter_time = get_metrics_from_logfiles(log_paths, timing_metric)
    current_grad_norm = get_metrics_from_logfiles(log_paths, "grad norm")
    current_alloc = get_metrics_from_logfiles(log_paths, alloc_metric)
    current_max_alloc = get_metrics_from_logfiles(log_paths, max_alloc_metric)
    current_gpu_util = get_metrics_from_logfiles(log_paths, "GPU utilization")

    golden_values_file_name = pathlib.Path(golden_values_path).name
    next_golden_values_path = os.path.join(assets_dir, "golden_values", golden_values_file_name)
    expected_golden_values_path = os.path.join(pathlib.Path(golden_values_path).parent, golden_values_file_name)
    _logger.info(f"Golden values path: {expected_golden_values_path}")

    # Always write actuals into experiment directory
    write_golden_values_to_disk(
        current_values=dict(
            **{
                step: {
                    k: v
                    for k, v in {
                        loss_metric: current_train_loss.get(step),
                        timing_metric: current_iter_time.get(step),
                        "GPU utilization": current_gpu_util.get(step),
                    }.items()
                    if v is not None
                }
                for step in current_train_loss.keys()
            },
            **{
                alloc_metric: current_alloc,
                max_alloc_metric: current_max_alloc,
            },
        ),
        golden_values_path=next_golden_values_path,
        wandb_run=wandb_run,
    )

    error_msg = ""

    # check for grad norm
    has_nan_grad_norm = any(math.isnan(current_grad_norm[step]) for step in current_grad_norm)
    has_inf_grad_norm = any(math.isinf(current_grad_norm[step]) for step in current_grad_norm)
    if has_nan_grad_norm or has_inf_grad_norm:
        error_msg += "Grad norm check failed. Found NaN or Inf in grad norm.\n"
        error_msg += f"Grad norm values: {current_grad_norm}\n"

    # check if golden values are exist for this model
    if not os.path.exists(expected_golden_values_path):
        error_msg += "Convergence check failed due to missing golden values.\n"
        error_msg += "This is expected if it is the first time running this model.\n"
        error_msg += (
            f"You will need to add the golden values ({expected_golden_values_path}) "
            "into the repository before the next run."
        )
        _logger.error(error_msg)
        sys.exit(1)

    _logger.info("Found existing golden values file, performing convergence check")
    with open(expected_golden_values_path, "r") as f:
        expected_golden_values = json.load(f)

    # Normalize: new format stores baseline data under a "baseline" key.
    if "baseline" in expected_golden_values:
        expected_golden_values = expected_golden_values["baseline"]

    steps = []
    golden_train_loss = {}
    golden_iter_time = {}
    golden_gpu_util = {}
    golden_alloc = None
    golden_max_alloc = None
    for key, value in expected_golden_values.items():
        if key == alloc_metric:
            golden_alloc = value
            continue
        if key == max_alloc_metric:
            golden_max_alloc = value
            continue
        steps.append(key)
        golden_train_loss[key] = value[loss_metric]
        golden_iter_time[key] = value[timing_metric]
        golden_gpu_util[key] = value.get("GPU utilization")

    # Extract golden_lm_loss and golden_iter_time lists
    _logger.info(f"Comparing {len(steps)} training steps for convergence")
    steps = sorted(golden_train_loss.keys(), key=int)

    # check for convergence
    golden_train_loss_values = np.array([golden_train_loss[str(step)] for step in steps])
    current_train_loss_values = np.array([current_train_loss.get(s, float("nan")) for s in steps])
    _logger.info(f"Current loss values (last 15): {current_train_loss_values[-15:]}")
    _logger.info(f"Golden loss values (last 15): {golden_train_loss_values[-15:]}")
    convergence_result = validate_convergence(
        current_values=current_train_loss_values,
        golden_values=golden_train_loss_values,
        steps=steps,
        logger=_logger,
        config=convergence_config,
        wandb_run=wandb_run,
    )
    if not convergence_result["passed"]:
        error_msg += f"Convergence check failed. {convergence_result['summary']}\n"
        error_msg += f"Failed metrics: {', '.join(convergence_result['failed_metrics'])}\n"
        if convergence_result.get("details"):
            error_msg += "Details:\n" + convergence_result["details"]

    # check for performance
    golden_iter_time_values = np.array([golden_iter_time[str(step)] for step in steps])
    current_iter_time_values = np.array([current_iter_time.get(s, float("nan")) for s in steps])
    # Use explicit None-check: dict.get(key, default) only applies the default when the key is
    # absent; if the key exists but its value is None (e.g. "GPU utilization" missing from the
    # golden file for that step), .get() returns None — not the default — creating an object
    # array that breaks np.nanmean.
    golden_gpu_util_values = np.array(
        [float(v) if (v := golden_gpu_util.get(s)) is not None else float("nan") for s in steps]
    )
    current_gpu_util_values = np.array(
        [float(v) if (v := current_gpu_util.get(s)) is not None else float("nan") for s in steps]
    )
    _logger.info(f"Current GPU util values (last 15): {current_gpu_util_values[-15:]}")
    _logger.info(f"Golden GPU util values (last 15): {golden_gpu_util_values[-15:]}")
    performance_result = validate_performance(
        current_gpu_util_values=current_gpu_util_values,
        golden_gpu_util_values=golden_gpu_util_values,
        steps=steps,
        logger=_logger,
        config=performance_config,
        wandb_run=wandb_run,
    )
    # Add iter-time averages for debugging (not used for pass/fail)
    start = performance_config.get("eval_time_start_step")
    if start is None:
        start = max(1, int(len(steps) * performance_config.get("skip_first_percent_time", 0.1)))
    end = performance_config.get("eval_time_end_step")
    performance_result["metrics"]["current_avg_iter_time_ms"] = float(np.nanmean(current_iter_time_values[start:end]))
    performance_result["metrics"]["golden_avg_iter_time_ms"] = float(np.nanmean(golden_iter_time_values[start:end]))
    if not performance_result["passed"]:
        direction = performance_result["metrics"]["direction"]
        signed_diff = performance_result["metrics"]["signed_diff"]
        error_msg += f"Performance check failed. {performance_result['summary']}\n"
        error_msg += (
            f"GPU util {direction}: signed diff {signed_diff * 100:.2f}% > "
            f"±{performance_config.get('timing_threshold', 0.05) * 100:.1f}%\n"
        )

    # check for memory
    memory_metrics_missing = golden_alloc is None or golden_max_alloc is None
    if memory_metrics_missing:
        _logger.warning("Memory metrics (alloc, max_alloc) not found in golden values - skipping memory validation")
    else:
        memory_result = validate_memory(
            golden_alloc=golden_alloc,
            current_alloc=current_alloc,
            golden_max_alloc=golden_max_alloc,
            current_max_alloc=current_max_alloc,
            logger=_logger,
            wandb_run=wandb_run,
            config=memory_config,
        )
        if not memory_result["passed"]:
            error_msg += f"Memory check failed. {memory_result['summary']}\n"
            error_msg += f"Max alloc difference: {memory_result['metrics']['max_alloc_diff'] * 100:.2f}%\n"
            error_msg += f"Alloc difference: {memory_result['metrics']['alloc_diff'] * 100:.2f}%\n"
            error_msg += f"Threshold: {memory_config['memory_threshold'] * 100:.1f}%\n"

    if wandb_run is not None:
        wandb_run.define_metric("compare/*", step_metric="compare/step")
        for i in range(len(steps)):
            wandb_run.log(
                {
                    "compare/step": i + 1,
                    "compare/current_lm_loss": current_train_loss_values[i],
                    "compare/current_iter_time": current_iter_time_values[i],
                    "compare/golden_lm_loss": golden_train_loss_values[i],
                    "compare/golden_iter_time": golden_iter_time_values[i],
                    "compare/current_gpu_util": current_gpu_util_values[i],
                    "compare/golden_gpu_util": golden_gpu_util_values[i],
                    "compare/current_grad_norm": current_grad_norm.get(steps[i], float("nan")),
                }
            )

    # Determine if we need to update golden values or if there are actual validation failures
    has_validation_failures = (
        not convergence_result["passed"] or not performance_result["passed"] or has_nan_grad_norm or has_inf_grad_norm
    )

    if not memory_metrics_missing:
        has_validation_failures = has_validation_failures or not memory_result["passed"]

    if memory_metrics_missing:
        if has_validation_failures:
            # There are actual validation failures - warn about them, don't suggest updating golden values
            error_msg += "\n⚠️  WARNING: Convergence or performance validation failed!\n"
            error_msg += "Fix the validation failures above before updating golden values.\n"
            error_msg += "\nNote: Memory metrics (alloc, max_alloc) are also missing from golden values,\n"
            error_msg += "but they should only be added AFTER convergence and performance validations pass.\n"
        else:
            # Only missing metrics, no validation failures - suggest updating golden values
            error_msg += "\n📝 Memory metrics (alloc, max_alloc) are missing from golden values.\n"
            error_msg += "All other validations passed successfully.\n"
            error_msg += f"Please update the golden values file: {expected_golden_values_path}\n"
            error_msg += "Add the following memory metrics to the golden values:\n"
            error_msg += f'  "{alloc_metric}": {current_alloc},\n'
            error_msg += f'  "{max_alloc_metric}": {current_max_alloc}\n'

    _logger.info(f"Convergence check completed successfully for {model_family_name}_{model_recipe_name}")
    return has_validation_failures is False, error_msg
