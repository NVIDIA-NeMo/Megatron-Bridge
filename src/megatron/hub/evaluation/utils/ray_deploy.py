import logging
import multiprocessing
import signal
import sys
from typing import Optional

from nemo_deploy.deploy_ray import DeployRay
from nemo_deploy.nlp.megatronllm_deployable_ray import MegatronRayDeployable


logger = logging.getLogger(__name__)


def get_available_cpus() -> int:
    """Get the number of available CPUs."""
    return multiprocessing.cpu_count()


def signal_handler(signum, frame, ray_deployer: DeployRay):
    """Handle termination signals."""
    logger.info("Received termination signal. Shutting down...")
    ray_deployer.stop()
    sys.exit(0)


def deploy_with_ray(
    nemo_checkpoint: str,
    num_gpus: int,
    num_nodes: int,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    num_replicas: int,
    num_cpus_per_replica: Optional[int] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_id: str = "megatron_model",
    enable_cuda_graphs: bool = False,
    enable_flash_decode: bool = True,
    legacy_ckpt: bool = False,
    include_dashboard: bool = True,
    cuda_visible_devices: str = "",
) -> None:
    """
    Deploy the model using Ray Serve.

    Args:
        nemo_checkpoint: Path to the NeMo checkpoint
        num_gpus: Number of GPUs per node
        num_nodes: Number of nodes
        tensor_model_parallel_size: Tensor parallelism size
        pipeline_model_parallel_size: Pipeline parallelism size
        context_parallel_size: Context parallelism size
        expert_model_parallel_size: Expert parallelism size
        num_replicas: Number of model replicas to deploy
        num_cpus_per_replica: Number of CPUs per replica
        host: Host address to serve on
        port: Port to serve on
        model_id: Model identifier
        enable_cuda_graphs: Whether to enable CUDA graphs
        enable_flash_decode: Whether to enable flash decode
        legacy_ckpt: Whether using legacy checkpoint format
        include_dashboard: Whether to include Ray dashboard
        cuda_visible_devices: CUDA visible devices string
    """
    # Calculate total GPUs
    total_gpus = num_gpus * num_nodes
    logger.info(f"Total GPUs: {total_gpus}")

    # Calculate GPUs per replica
    gpus_per_replica = total_gpus // num_replicas

    # Validate parallelism configuration
    parallelism_per_replica = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size

    if parallelism_per_replica != gpus_per_replica:
        logger.error(
            f"Parallelism per replica ({parallelism_per_replica}) must equal GPUs per replica ({gpus_per_replica})"
        )
        logger.error(f"Total GPUs: {total_gpus}, Num replicas: {num_replicas}, GPUs per replica: {gpus_per_replica}")
        logger.error(
            f"Each replica needs: tensor_parallel({tensor_model_parallel_size}) * "
            f"pipeline_parallel({pipeline_model_parallel_size}) * "
            f"context_parallel({context_parallel_size}) = {parallelism_per_replica} GPUs"
        )
        sys.exit(1)

    logger.info(f"Configuration: {num_replicas} replicas, {gpus_per_replica} GPUs per replica")

    # Initialize Ray deployment
    ray_deployer = DeployRay(
        num_cpus=get_available_cpus(),
        num_gpus=total_gpus,
        include_dashboard=include_dashboard,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            }
        },
    )

    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, ray_deployer))
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: signal_handler(signum, frame, ray_deployer),
    )

    try:
        # Start Ray Serve
        ray_deployer.start(host=host, port=port)

        # Create the Multi-Rank Megatron model deployment
        app = MegatronRayDeployable.options(
            num_replicas=num_replicas,
            ray_actor_options={"num_cpus": num_cpus_per_replica},
        ).bind(
            nemo_checkpoint_filepath=nemo_checkpoint,
            num_gpus=gpus_per_replica,
            num_nodes=num_nodes,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            context_parallel_size=context_parallel_size,
            model_id=model_id,
            enable_cuda_graphs=enable_cuda_graphs,
            enable_flash_decode=enable_flash_decode,
            legacy_ckpt=legacy_ckpt,
        )

        # Deploy the model
        ray_deployer.run(app, model_id)

        logger.info(f"Megatron model deployed successfully at {host}:{port}")
        logger.info("Press Ctrl+C to stop the deployment")

        # Keep the script running
        while True:
            signal.pause()
    except Exception as e:
        logger.error(f"Error during deployment: {str(e)}")
        ray_deployer.stop()
        sys.exit(1)
