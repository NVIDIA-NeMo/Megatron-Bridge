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

from typing import TYPE_CHECKING, Optional, Union
from pathlib import Path
import logging
import importlib

import torch

from nemo_lm.evaluation.utils.api import EvaluationConfig, EvaluationTarget, MisconfigurationError

AnyPath = Union[Path, str]

logger = logging.getLogger(__name__)

def deploy(
    nemo_checkpoint: Optional[AnyPath] = None,
    triton_model_name: str = "triton_model",
    triton_model_version: Optional[int] = 1,
    triton_http_port: int = 8000,
    triton_grpc_port: int = 8001,
    triton_http_address: str = "0.0.0.0",
    start_fastapi_server: bool = True,
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
):
    """
    Deploys nemo model on a PyTriton server "in-framework" to be used as OAI API compatible server for evaluations with
    NVIDIA Evals Factory (https://pypi.org/project/nvidia-lm-eval/).

    Args:
        nemo_checkpoint (Path): Path for nemo checkpoint.
        triton_model_name (str): Name for the model that gets deployed on PyTriton. Please ensure that the same model
            name is passed to the evalute method for the model to be accessible while sending evalution requests.
            Default: 'triton_model'.
        triton_model_version (Optional[int]): Version for the triton model. Default: 1.
        triton_http_port (int): HTTP port for the PyTriton server. Default: 8000.
        triton_grpc_port (int): gRPC Port for the PyTriton server. Default: 8001.
        triton_http_address (str): HTTP address for the PyTriton server. Default:  "0.0.0.0".
        triton_model_repository (Path): Folder for the trt-llm conversion, trt-llm engine gets saved in this specified
            path. If None, saves it in /tmp dir. Default: None.
        start_fastapi_server (bool): Starts FastAPI server which acts as a proxy in between to expose the
            v1/completions and v1/chat/completions OpenAI (OAI) compatible endpoints as PyTriton does not expose a
            standard HTTP/REST API. Only supported for "in-framework" deployment and not with "trtllm" backend.
            Default: True.
        fastapi_http_address (str): HTTP address for FastAPI interface/server.  Default: "0.0.0.0". OAI endpoints via
            FastAPI interface are only supported for "in-framework" backend.
        fastapi_port (int): Port for FastAPI interface/server. Applicable only for "in-framework" backend.
            Default: 8080.
        num_gpus (int): Number of GPUs per node for export to trtllm and deploy. Default: 1.
        tensor_parallelism_size (int): Tensor parallelism size. Default: 1.
        pipeline_parallelism_size (int): Pipeline parallelism size. Default: 1.
        context_parallel_size (int): Context parallelism size. Default: 1.
        expert_model_parallel_size (int): Expert parallelism size. Default: 1.
        dtype (str): dtype of the TensorRT-LLM model. Autodetected from the model weights dtype by default.
        max_input_len (int): Max input length of the model. Default: 4096.
        max_batch_size (int): Max batch size of the model. Default: 8.
        openai_format_response (bool): Return the response from PyTriton server in OpenAI compatible format.
            Needs to be True while running evaluation. Default: True.
        enable_flash_decode (bool): If True runs in-framework deployment with flash decode enabled (not supported for
            the trtllm backend).
    """
    import os
    import uvicorn
    from nemo_deploy import DeployPyTriton

    assert (
        start_fastapi_server is True
    ), 'in-framework deployment exposes OAI API endpoints v1/completions and \
    v1/chat/completions hence needs fastAPI interface to expose these endpoints to PyTriton. Please set \
    start_fastapi_server to True'
    if triton_http_port == fastapi_port:
        raise ValueError("FastAPI port and Triton server port cannot use the same port. Please change them")
    # Store triton ip, port relevant for FastAPI as env vars to be accessible by fastapi_interface_to_pytriton.py
    os.environ["TRITON_HTTP_ADDRESS"] = triton_http_address
    os.environ["TRITON_PORT"] = str(triton_http_port)

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
        max_batch_size=max_batch_size,
    )

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            try:
                nm = DeployPyTriton(
                    model=triton_deployable,
                    triton_model_name=triton_model_name,
                    triton_model_version=triton_model_version,
                    max_batch_size=max_batch_size,
                    http_port=triton_http_port,
                    grpc_port=triton_grpc_port,
                    address=triton_http_address,
                )

                logger.info("Triton deploy function will be called.")
                nm.deploy()
                nm.run()
            except Exception as error:
                logger.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            try:
                if start_fastapi_server:
                    try:
                        logger.info("REST service will be started.")
                        uvicorn.run(
                            'nemo.deploy.service.fastapi_interface_to_pytriton:app',
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
    from nemo_lm.evaluation.utils.base import find_framework, wait_for_fastapi_server
    import yaml

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

    base_url, _ = target_cfg.api_endpoint.url.split('/v1')
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