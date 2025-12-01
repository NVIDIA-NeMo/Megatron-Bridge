from pathlib import Path
from typing import Any, Optional

from megatron.bridge.utils.common_utils import print_rank_last


def on_save_checkpoint_success(
    checkpoint_path: str,
    save_dir: str,
    iteration: int,
    mlflow_logger: Optional[Any],
) -> None:
    """Callback executed after a checkpoint is successfully saved.

    If an MLFlow logger is provided, logs the checkpoint directory as an MLFlow
    artifact under a structured artifact path that includes the iteration number.

    Args:
        checkpoint_path: The path to the specific checkpoint file/directory saved.
        save_dir: The base directory where checkpoints are being saved.
        iteration: The training iteration at which the checkpoint was saved.
        mlflow_logger: The MLFlow module (e.g., ``mlflow``) with an active run.
                       If None, this function is a no-op.
    """
    if mlflow_logger is None:
        return

    try:
        checkpoint_path = str(Path(checkpoint_path).resolve())
        base_name = Path(save_dir).name or "checkpoints"
        artifact_subdir = f"{base_name}/iter_{iteration:07d}"
        mlflow_logger.log_artifacts(checkpoint_path, artifact_path=artifact_subdir)
    except Exception as exc:
        # continue training
        print_rank_last(f"Failed to log checkpoint artifacts to MLFlow: {exc}")


def on_load_checkpoint_success(
    checkpoint_path: str,
    load_dir: str,
    mlflow_logger: Optional[Any],
) -> None:
    """Callback executed after a checkpoint is successfully loaded.

    For MLFlow, this emits a simple metric and tag to document which checkpoint
    was loaded during the run. It does not perform artifact lookups.

    Args:
        checkpoint_path: The path to the specific checkpoint file/directory loaded.
        load_dir: The base directory from which the checkpoint was loaded.
        mlflow_logger: The MLFlow module (e.g., ``mlflow``) with an active run.
                       If None, this function is a no-op.
    """
    if mlflow_logger is None:
        return

    try:
        resolved_ckpt = str(Path(checkpoint_path).resolve())
        resolved_load_dir = str(Path(load_dir).resolve())
        mlflow_logger.set_tags(
            {
                "last_loaded_checkpoint": resolved_ckpt,
                "checkpoint_base_dir": resolved_load_dir,
            }
        )
    except Exception as exc:
        print_rank_last(f"Failed to record loaded checkpoint information to MLFlow: {exc}")
