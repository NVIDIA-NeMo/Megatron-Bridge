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
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import wandb


# Setup logging
logger = logging.getLogger(__name__)


def get_metrics_from_logfiles(log_paths: List[str], metric: str):
    """
    Parse training log file and extract metrics.

    Args:
        log_path: Path to the log file
        metric: Metric name to extract

    Returns:
        Dictionary with format: {step: value}
    """
    metrics = {"elapsed time per iteration (ms)": {}, "lm loss": {}, "GPU utilization": {}, "step time": {}}

    content = ""
    for log_path in log_paths:
        with open(log_path, "r") as f:
            file_content = f.read()
            content += file_content + "\n"

    patterns = {
        "iteration": r"iteration\s+(\d+)/",
        "elapsed time per iteration (ms)": r"elapsed time per iteration \(ms\):\s+([\d.]+)",
        "lm loss": r"lm loss:\s+([\d.E+\-]+)",
        "GPU utilization": r"GPU utilization:\s+([\d.]+)",
        "step time": r"Step Time :\s+([\d.]+)s",
    }

    pending_step_time = None
    pending_gpu_util = None

    for line in content.split("\n"):
        # Check for step time and GPU utilization
        if match := re.search(patterns["step time"], line):
            pending_step_time = float(match.group(1))

        if match := re.search(patterns["GPU utilization"], line):
            pending_gpu_util = float(match.group(1))

        # Check for iteration line
        if match := re.search(patterns["iteration"], line):
            current_iteration = int(match.group(1))

            # Assign pending metrics to the iteration that just completed
            # (current_iteration - 1, but use 0-indexed so current_iteration - 1)
            completed_step = str(current_iteration - 1)

            if pending_step_time is not None:
                metrics["step time"][completed_step] = pending_step_time
                pending_step_time = None

            if pending_gpu_util is not None:
                metrics["GPU utilization"][completed_step] = pending_gpu_util
                pending_gpu_util = None

            # Extract metrics from the iteration line itself
            for metric_name in ["elapsed time per iteration (ms)", "lm loss"]:
                if match := re.search(patterns[metric_name], line):
                    metrics[metric_name][completed_step] = float(match.group(1))

    return metrics[metric]


def validate_loss_curve_convergence(
    current_values: np.ndarray,
    golden_values: np.ndarray,
    steps: List[str],
    logger: logging.Logger,
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

    return results


def calc_convergence(
    model_type: str,
    model_size: str,
    num_nodes: int,
    max_steps: int,
    cluster: str,
    assets_dir: str,
    log_paths: List[str],
    loss_metric: str,
    timing_metric: str,
    golden_values_path: str,
    timing_threshold: float,
    skip_first_percent_time: float,
    convergence_config: Dict[str, Any] = None,
    wandb_run: Optional[wandb.Run] = None,
):
    """
    Calculate convergence metrics and validate against golden values.

    Args:
        model_type: Type of model (e.g., 'gpt', 'bert')
        model_size: Size of model (e.g., 'small', 'medium', 'large')
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
    """
    logger.info(f"Starting convergence check for {model_type}_{model_size} on cluster {cluster}")

    current_train_loss = get_metrics_from_logfiles(log_paths, loss_metric)
    current_step_timing = get_metrics_from_logfiles(log_paths, timing_metric)

    golden_values_file_name = f"{model_type}_{model_size}_{num_nodes}node_{max_steps}steps_{cluster}.json"
    expected_golden_values_path = os.path.join(golden_values_path, golden_values_file_name)
    current_golden_values_path = os.path.join(assets_dir, "golden_values", golden_values_file_name)
    today_date = datetime.now().strftime("%m.%d.%y")
    logger.info(f"Golden values path: {expected_golden_values_path}")

    # Always write actuals into experiment directory
    os.makedirs(os.path.dirname(current_golden_values_path), exist_ok=True)
    with open(current_golden_values_path, "w") as f:
        current_golden_values = {
            str(step): {loss_metric: current_train_loss[str(step)], timing_metric: current_step_timing[str(step)]}
            for step in range(len(current_train_loss))
        }
        json.dump(current_golden_values, f)
    logger.info(f"Golden values were saved for {model_type}_{model_size}: {current_golden_values}")

    # check if golden values are exist for this model
    error_msg = None
    if os.path.exists(expected_golden_values_path):
        logger.info("Found existing golden values file, performing convergence check")
        # read train loss from current test
        with open(expected_golden_values_path, "r") as f:
            expected_golden_values = json.load(f)

        steps = []
        golden_train_loss = {}
        golden_iter_time = {}
        for key, value in expected_golden_values.items():
            steps.append(key)
            golden_train_loss[key] = value[loss_metric]
            golden_iter_time[key] = value[timing_metric]

        # Extract golden_lm_loss and golden_iter_time lists
        steps = sorted(golden_train_loss.keys(), key=int)
        golden_lm_loss = [golden_train_loss[str(step)] for step in steps]
        golden_iter_time = [golden_iter_time[str(step)] for step in steps]

        # check for convergence
        current_train_loss_values = np.array([current_train_loss[s] for s in steps])
        golden_train_loss_values = np.array(golden_lm_loss)
        golden_step_timing_values = np.array(golden_iter_time)

        logger.info(f"Comparing {len(steps)} training steps for convergence")
        logger.info(f"Current loss values: {current_train_loss_values}")
        logger.info(f"Golden loss values: {golden_train_loss_values}")
        logger.info(f"Extracted golden_lm_loss: {golden_lm_loss}")
        logger.info(f"Extracted golden_iter_time: {golden_iter_time}")

        # Multi-metric convergence validation strategy
        convergence_result = validate_loss_curve_convergence(
            current_train_loss_values, golden_train_loss_values, steps, logger, convergence_config
        )
        # Step Timing Validation
        # Get current step timing values for the same steps
        current_step_timing_values = np.array([current_step_timing[s] for s in steps])

        # Discard first N% of iterations for stable timing comparison
        skip_first_n_percent = max(1, int(len(steps) * skip_first_percent_time))
        current_timing_stable = current_step_timing_values[skip_first_n_percent:]
        golden_timing_stable = golden_step_timing_values[skip_first_n_percent:]

        # Calculate average step timing
        current_avg_timing = np.mean(current_timing_stable)
        golden_avg_timing = np.mean(golden_timing_stable)

        # Calculate timing difference
        timing_diff = abs(current_avg_timing - golden_avg_timing) / golden_avg_timing

        logger.info(f"Step timing comparison (excluding first {skip_first_percent_time * 100:.1f}% of iterations):")
        logger.info(f"  Current average timing: {current_avg_timing:.4f}s")
        logger.info(f"  Golden average timing: {golden_avg_timing:.4f}s")
        logger.info(f"  Timing difference: {timing_diff:.4f} ({timing_diff * 100:.2f}%)")
        logger.info(f"  Threshold: {timing_threshold * 100:.1f}%")

        if wandb_run:
            wandb_run.summary["current_avg_timing"] = current_avg_timing
            wandb_run.summary["golden_avg_timing"] = golden_avg_timing
            wandb_run.summary["timing_diff"] = timing_diff
            wandb_run.summary["timing_threshold"] = timing_threshold
            wandb_run.define_metric("compare/*", step_metric="compare/step")
            for i in range(len(steps)):
                wandb_run.log(
                    {
                        "compare/step": i + 1,
                        "compare/current_lm_loss": current_train_loss_values[i],
                        "compare/current_iter_time": current_step_timing_values[i],
                        "compare/golden_lm_loss": golden_lm_loss[i],
                        "compare/golden_iter_time": golden_iter_time[i],
                    }
                )

            artifact = wandb.Artifact("golden_values", type="dataset")
            with artifact.new_file("golden_values.json", "w") as f:
                json.dump({today_date: current_golden_values}, f)

            wandb_run.log_artifact(artifact)

        # Check if timing is within threshold
        if timing_diff > timing_threshold:
            logger.warning(f"Step timing validation FAILED: {timing_diff * 100:.2f}% > {timing_threshold * 100:.1f}%")
            # Add timing failure to convergence result
            convergence_result["passed"] = False
            convergence_result["failed_metrics"].append("step_timing")
            convergence_result["summary"] = f"Failed {len(convergence_result['failed_metrics'])} out of 6 tests"
        else:
            logger.info(f"✓ Step timing validation passed: {timing_diff * 100:.2f}% <= {timing_threshold * 100:.1f}%")

        if not convergence_result["passed"]:
            error_msg = f"Convergence check failed. {convergence_result['summary']}\n"
            error_msg += f"Failed metrics: {', '.join(convergence_result['failed_metrics'])}\n"
            if convergence_result.get("details"):
                error_msg += "Details:\n" + convergence_result["details"]

        for key, value in convergence_result["metrics"].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info(f"Convergence check is passed. Train loss history file was updated for {model_type}_{model_size}.")
    else:
        error_msg = "Convergence check failed due to missing golden values.\n"
        error_msg += "This is expected if it is the first time running this model.\n"
        error_msg += (
            f"You will need to add the golden values ({expected_golden_values_path}) "
            "into the repository before the next run."
        )

    logger.info(f"Convergence check completed successfully for {model_type}_{model_size}")
    return error_msg is None, error_msg
