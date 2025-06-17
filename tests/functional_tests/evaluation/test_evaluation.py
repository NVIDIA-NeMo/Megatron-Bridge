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
import logging
import os
import signal
import subprocess

import pytest

from megatron.hub.evaluation.api import evaluate
from megatron.hub.evaluation.utils.api import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
from megatron.hub.evaluation.utils.base import wait_for_fastapi_server


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

nemo2_ckpt_path = "/home/TestData/nemo2_ckpt/llama3-1b-lingua"
max_batch_size = 4
legacy_ckpt = True


@pytest.fixture()
def triton_server():
    # Set environment variables
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/home/TestData/HF_HOME"
    os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"

    # Run deployment

    deploy_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "tests/functional_tests/evaluation/deploy_in_fw_script.py",
            "--nemo2_ckpt_path",
            nemo2_ckpt_path,
            "--max_batch_size",
            str(max_batch_size),
        ]
        + (["--legacy_ckpt"] if legacy_ckpt else []),
        env=None,
    )
    # Wait for server readiness
    logger.info("Waiting for server readiness...")
    server_ready = wait_for_fastapi_server(base_url="http://0.0.0.0:8886", max_retries=900)
    assert server_ready, "Server is not ready. Please look at the deploy process log for the error"

    yield

    deploy_proc.send_signal(signal.SIGTERM)
    deploy_proc.wait(timeout=30)
    try:
        os.killpg(deploy_proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class TestEvaluation:
    """
    Test evaluation with NVIDIA Evals Factory on nemo2 model deployed on PyTriton.
    """

    @pytest.mark.run_only_on("GPU")
    def test_gsm8k_evaluation(self, triton_server):
        """
        Test GSM8K evaluation benchmark.
        """

        # Run evaluation
        logger.info("Starting evaluation...")
        api_endpoint = ApiEndpoint(url="http://0.0.0.0:8886/v1/completions/")
        eval_target = EvaluationTarget(api_endpoint=api_endpoint)
        eval_params = {
            "limit_samples": 1,
        }
        eval_config = EvaluationConfig(type="gsm8k", params=ConfigParams(**eval_params))
        evaluate(target_cfg=eval_target, eval_cfg=eval_config)
        logger.info("Evaluation completed.")

    @pytest.mark.pleasefixme
    @pytest.mark.run_only_on("GPU")
    def test_arc_challenge_evaluation(self, triton_server):
        """
        Test ARC Challenge evaluation benchmark.
        """
        # Run evaluation
        logger.info("Starting evaluation...")
        api_endpoint = ApiEndpoint(url="http://0.0.0.0:8886/v1/completions/")
        eval_target = EvaluationTarget(api_endpoint=api_endpoint)
        eval_params = {
            "limit_samples": 1,
            "extra": {
                "tokenizer_backend": "huggingface",
                "tokenizer": "/home/TestData/nemo2_ckpt/llama3-1b-lingua/context/lingua",
            },
        }
        eval_config = EvaluationConfig(type="arc_challenge", params=ConfigParams(**eval_params))
        evaluate(target_cfg=eval_target, eval_cfg=eval_config)
        logger.info("Evaluation completed.")
