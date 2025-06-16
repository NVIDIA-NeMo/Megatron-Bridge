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

import importlib
import logging
from pathlib import Path
from typing import Optional, Union

import torch

from megatron.hub.evaluation.utils.api import EvaluationConfig, EvaluationTarget, MisconfigurationError


AnyPath = Union[Path, str]

logger = logging.getLogger(__name__)


def deploy(
    nemo_checkpoint: Optional[AnyPath] = None,
    serving_backend: str = "pytriton",
    model_name: str = "megatron_model",
    server_port: int = 8000,
    server_address: str = "0.0.0.0",
    fastapi_http_address: str = "0.0.0.0",
    fastapi_port: int = 8080,
    num_gpus: int = 1,
    num_nodes: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    max_input_len: int = 4096,
    max_batch_size: int = 8,
    enable_flash_decode: bool = True,
    enable_cuda_graphs: bool = True,
    # Ray deployment specific args
    num_replicas: Optional[int] = None,
    num_cpus_per_replica: Optional[int] = None,
    include_dashboard: bool = True,
    cuda_visible_devices: str = "",
    legacy_ckpt: bool = False,
):
    """
    Deploys nemo model on either PyTriton server or Ray Serve.

    Args:
        nemo_checkpoint (Path): Path for nemo checkpoint.
        serving_backend (str): Backend to use for serving ("pytriton" or "ray"). Default: "pytriton".
        model_name (str): Name for the model that gets deployed on PyTriton or Ray.
        server_port (int): HTTP port for the PyTriton or Ray server. Default: 8000.
        server_address (str): HTTP address for the PyTriton or Ray server. Default: "0.0.0.0".
        fastapi_http_address (str): HTTP address for FastAPI interface/server. Default: "0.0.0.0".
        fastapi_port (int): Port for FastAPI interface/server. Default: 8080.
        num_gpus (int): Number of GPUs per node. Default: 1.
        num_nodes (int): Number of nodes. Default: 1.
        tensor_parallelism_size (int): Tensor parallelism size. Default: 1.
        pipeline_parallelism_size (int): Pipeline parallelism size. Default: 1.
        context_parallel_size (int): Context parallelism size. Default: 1.
        expert_model_parallel_size (int): Expert parallelism size. Default: 1.
        max_input_len (int): Max input length of the model. Default: 4096.
        max_batch_size (int): Max batch size of the model. Default: 8.
        enable_flash_decode (bool): If True runs inferencewith flash decode enabled. Default: True.
        enable_cuda_graphs (bool): Whether to enable CUDA graphs for inference. Default: True.
        legacy_ckpt (bool): Indicates whether the checkpoint is in the legacy format. Default: False.
        ##### Ray deployment specific args #####
        num_replicas (int): Number of model replicas for Ray deployment. Default: 1. Only applicable for Ray backend.
        num_cpus_per_replica (int): Number of CPUs per replica for Ray deployment. Default: 8
        include_dashboard (bool): Whether to include Ray dashboard. Default: True.
        cuda_visible_devices (list): Comma-separated list of CUDA visible devices. Default: [0,1].
        legacy_ckpt (bool): Indicates whether the checkpoint is in legacy format. Default: False.
    """
    if serving_backend == "ray":
        if num_replicas is None:
            raise ValueError("num_replicas must be specified when using Ray backend")

        from megatron.hub.evaluation.utils.ray_deploy import deploy_with_ray

        deploy_with_ray(
            nemo_checkpoint=nemo_checkpoint,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            tensor_model_parallel_size=tensor_parallelism_size,
            pipeline_model_parallel_size=pipeline_parallelism_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            num_replicas=num_replicas,
            num_cpus_per_replica=num_cpus_per_replica,
            host=server_address,
            port=server_port,
            model_id=model_name,
            enable_cuda_graphs=enable_cuda_graphs,
            enable_flash_decode=enable_flash_decode,
            legacy_ckpt=legacy_ckpt,
            include_dashboard=include_dashboard,
            cuda_visible_devices=cuda_visible_devices,
        )
    else:  # pytriton backend
        import os
        import uvicorn
        from nemo_deploy import DeployPyTriton

        if server_port == fastapi_port:
            raise ValueError("FastAPI port and Triton server port cannot use the same port. Please change them")

        # Store triton ip, port relevant for FastAPI as env vars
        os.environ["TRITON_HTTP_ADDRESS"] = server_address
        os.environ["TRITON_PORT"] = str(server_port)

        try:
            from nemo_deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2
        except Exception as e:
            raise ValueError(
                f"Unable to import MegatronLLMDeployable, due to: {type(e).__name__}: {e} cannot run "
                f"evaluation with in-framework deployment"
            )

        triton_deployable = MegatronLLMDeployableNemo2(
            nemo_checkpoint_filepath=nemo_checkpoint,
            num_devices=num_gpus,
            num_nodes=num_nodes,
            tensor_model_parallel_size=tensor_parallelism_size,
            pipeline_model_parallel_size=pipeline_parallelism_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            inference_max_seq_length=max_input_len,
            enable_flash_decode=enable_flash_decode,
            enable_cuda_graphs=enable_cuda_graphs,
            max_batch_size=max_batch_size,
            legacy_ckpt=legacy_ckpt,
        )

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                try:
                    nm = DeployPyTriton(
                        model=triton_deployable,
                        triton_model_name=model_name,
                        max_batch_size=max_batch_size,
                        http_port=server_port,
                        address=server_address,
                    )

                    logger.info("Triton deploy function will be called.")
                    nm.deploy()
                    nm.run()
                except Exception as error:
                    logger.error("Error message has occurred during deploy function. Error message: " + str(error))
                    return

                try:
                    # start fastapi server which acts as a proxy to Pytriton server. Applies to PyTriton backend only.
                    try:
                        logger.info("REST service will be started.")
                        uvicorn.run(
                            "nemo_deploy.service.fastapi_interface_to_pytriton:app",
                            host=fastapi_http_address,
                            port=fastapi_port,
                            reload=True,
                        )
                    except Exception as error:
                        logger.error(
                            "Error message has occurred during REST service start. Error message: " + str(error)
                        )
                    logger.info("Model serving on Triton will be started.")
                    nm.serve()
                except Exception as error:
                    logger.error("Error message has occurred during deploy function. Error message: " + str(error))
                    return

                logger.info("Model serving will be stopped.")
                nm.stop()
            elif torch.distributed.get_rank() > 0:
                triton_deployable.generate_other_ranks()


def evaluate(
    target_cfg: EvaluationTarget,
    eval_cfg: EvaluationConfig = EvaluationConfig(type="gsm8k"),
) -> dict:
    """
    Evaluates nemo model deployed on PyTriton server using nvidia-lm-eval

    Args:
        target_cfg (EvaluationTarget): target of the evaluation. Providing model_id and
            url in EvaluationTarget.api_endpoint is required to run evaluations.
        eval_cfg (EvaluationConfig): configuration for evaluations. Default type (task): gsm8k.
    """
    import yaml

    from megatron.hub.evaluation.utils.base import find_framework, wait_for_fastapi_server

    eval_type_components = eval_cfg.type.split(".")
    if len(eval_type_components) == 2:
        framework_name, task_name = eval_type_components
        # evaluation package expect framework name to be hyphenated
        framework_name = framework_name.replace("_", "-")
        eval_cfg.type = f"{framework_name}.{task_name}"
    elif len(eval_type_components) == 1:
        framework_name, task_name = None, eval_type_components[0]
    else:
        raise MisconfigurationError(
            "eval_type must follow 'framework_name.task_name'. No additional dots are allowed."
        )

    if framework_name is None:
        framework_module_name = find_framework(task_name)
    else:
        framework_module_name = f"core_evals.{framework_name.replace('-', '_')}"
    try:
        evaluate = importlib.import_module(".evaluate", package=framework_module_name)
    except ImportError:
        raise ImportError(
            f"Please ensure that {framework_module_name} is installed in your env "
            f"as it is required to run {eval_cfg.type} evaluation"
        )

    base_url, _ = target_cfg.api_endpoint.url.split("/v1")
    server_ready = wait_for_fastapi_server(base_url=base_url, model_name=target_cfg.api_endpoint.model_id)
    if not server_ready:
        raise RuntimeError("Server not ready for evaluation")

    results = evaluate.evaluate_accuracy(
        target_cfg=target_cfg,
        eval_cfg=eval_cfg,
    )
    results_dict = results.model_dump()

    logger.info("========== RESULTS ==========")
    logger.info(yaml.dump(results_dict))

    return results_dict
