try:
    from modelopt.torch.opt.plugins import restore_sharded_modelopt_state
except ImportError as e:
    raise ImportError('Required `"nvidia-modelopt[torch]"` is not installed!') from e

import os.path

from megatron.core import dist_checkpointing

from megatron.bridge.utils.common_utils import unwrap_model


def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint path.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        True if modelopt_state folder exists, False otherwise
    """
    modelopt_state_path = os.path.join(checkpoint_path, "modelopt_state")
    return os.path.isdir(modelopt_state_path)


def load_modelopt_state(model, checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.

    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    unwrapped_model = unwrap_model(model)
    restore_sharded_modelopt_state(unwrapped_model, checkpoint_path)


def load_modelopt_checkpoint(model, checkpoint_path: str) -> None:
    """Load modelopt_state from a checkpoint.

    Args:
        model: The model to load the modelopt_state into
        checkpoint_path: Path to the checkpoint directory
    """
    unwrapped_model = unwrap_model(model)
    sharded_state_dict = unwrapped_model[0].sharded_state_dict()
    model_state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_path, strict="assume_ok_unexpected")
    unwrapped_model[0].load_state_dict(model_state_dict, strict=False)
