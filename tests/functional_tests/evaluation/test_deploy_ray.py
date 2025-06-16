from megatron.hub.evaluation.api import deploy
import pytest
import os
import logging

logger = logging.getLogger(__name__)

@pytest.mark.run_only_on("GPU")
def test_deploy_ray():
    """
    Test deploying a model using Ray backend.
    """
    nemo_checkpoint = '/home/TestData/nemo2_ckpt/llama3-1b-lingua'
    max_input_len = 4096
    max_batch_size = 4
    num_gpus = 1
    legacy_ckpt = True
    serving_backend = "ray"
    num_replicas = 1
    num_cpus_per_replica = 8
    cuda_visible_devices = "0"

    # Test deployment
    logger.info("Testing ray deployment...")
    deploy(
        nemo_checkpoint=nemo_checkpoint,
        max_input_len=max_input_len,
        max_batch_size=max_batch_size,
        num_gpus=num_gpus,
        legacy_ckpt=legacy_ckpt,
        serving_backend=serving_backend,
        num_replicas=num_replicas,
        num_cpus_per_replica=num_cpus_per_replica,
        cuda_visible_devices=cuda_visible_devices
    )
    logger.info("Deployment test completed successfully.")


