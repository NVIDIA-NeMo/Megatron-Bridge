import time
from unittest.mock import patch

import yaml

from megatron.bridge.training.utils.checkpoint_utils import read_run_config


def test_performance_large_config_file(tmp_path):
    """Test performance with large configuration files."""
    # Create a large config file
    large_config = {
        "model": {
            "layers": list(range(1000)),  # Large list
            "weights": {f"layer_{i}": [0.1] * 100 for i in range(100)},  # Nested large data
        },
        "training": {"hyperparameters": {f"param_{i}": i * 0.001 for i in range(1000)}},
    }

    config_file = tmp_path / "large_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(large_config, f)

    with (
        patch("megatron.bridge.training.utils.checkpoint_utils.get_rank_safe", return_value=0),
        patch("megatron.bridge.training.utils.checkpoint_utils.torch.distributed.is_initialized", return_value=False),
    ):
        start_time = time.time()
        result = read_run_config(str(config_file))
        end_time = time.time()

        # Verify correctness
        assert result == large_config

        # Performance should be reasonable (less than 2 seconds for large config)
        assert end_time - start_time < 2.0, f"Reading large config took {end_time - start_time:.2f} seconds"
