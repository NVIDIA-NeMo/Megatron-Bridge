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
import time
from megatron.hub.evaluation.api import evaluate
from megatron.hub.evaluation.utils.api import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
from megatron.hub.evaluation.utils.base import wait_for_fastapi_server


logger = logging.getLogger(__name__)


class TestEvaluation:
    """
    Test evaluation with NVIDIA Evals Factory on nemo2 model deployed on PyTriton.
    """

    @pytest.mark.run_only_on("GPU")
    def test_gsm8k_evaluation(self):
        """
        Test GSM8K evaluation benchmark.
        """
        nemo2_ckpt_path = "/home/TestData/nemo2_ckpt/llama3-1b-lingua"
        max_batch_size = 4
        eval_type = "gsm8k"
        limit = 1
        legacy_ckpt = True

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = "/home/TestData/HF_HOME"
        os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"

        # Run deployment
        deploy_proc = subprocess.Popen(
            [
                "python",
                "tests/functional_tests/evaluation/deploy_in_fw_script.py",
                "--nemo2_ckpt_path",
                nemo2_ckpt_path,
                "--max_batch_size",
                str(max_batch_size),
            ]
            + (["--legacy_ckpt"] if legacy_ckpt else []),
        )

        try:
            # Wait for server readiness
            logger.info("Waiting for server readiness...")
            server_ready = wait_for_fastapi_server(base_url="http://0.0.0.0:8886", max_retries=600)
            assert server_ready, "Server is not ready. Please look at the deploy process log for the error"

            # Run evaluation
            logger.info("Starting evaluation...")
            api_endpoint = ApiEndpoint(url="http://0.0.0.0:8886/v1/completions/")
            eval_target = EvaluationTarget(api_endpoint=api_endpoint)
            eval_params = {
                "limit_samples": limit,
            }
            eval_config = EvaluationConfig(type=eval_type, params=ConfigParams(**eval_params))
            evaluate(target_cfg=eval_target, eval_cfg=eval_config)
            logger.info("Evaluation completed.")

        finally:
            deploy_proc.send_signal(signal.SIGINT)
            time.sleep(60)  # Wait for the server to shutdown

    @pytest.mark.run_only_on("GPU")
    def test_arc_challenge_evaluation(self):
        """
        Test ARC Challenge evaluation benchmark.
        """
        nemo2_ckpt_path = "/home/TestData/nemo2_ckpt/llama3-1b-lingua"
        tokenizer_path = "/home/TestData/nemo2_ckpt/llama3-1b-lingua/context/lingua"
        max_batch_size = 4
        eval_type = "arc_challenge"
        limit = 1
        legacy_ckpt = True

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = "/home/TestData/HF_HOME"
        os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"

        # Run deployment
        deploy_proc = subprocess.Popen(
            [
                "python",
                "tests/functional_tests/evaluation/deploy_in_fw_script.py",
                "--nemo2_ckpt_path",
                nemo2_ckpt_path,
                "--max_batch_size",
                str(max_batch_size),
            ]
            + (["--legacy_ckpt"] if legacy_ckpt else []),
        )

        try:
            # Wait for server readiness
            logger.info("Waiting for server readiness...")
            server_ready = wait_for_fastapi_server(base_url="http://0.0.0.0:8886", max_retries=600)
            assert server_ready, "Server is not ready. Please look at the deploy process log for the error"

            # Run evaluation
            logger.info("Starting evaluation...")
            api_endpoint = ApiEndpoint(url="http://0.0.0.0:8886/v1/completions/")
            eval_target = EvaluationTarget(api_endpoint=api_endpoint)
            eval_params = {
                "limit_samples": limit,
                "extra": {
                    "tokenizer_backend": "huggingface",
                    "tokenizer": tokenizer_path,
                },
            }
            eval_config = EvaluationConfig(type=eval_type, params=ConfigParams(**eval_params))
            evaluate(target_cfg=eval_target, eval_cfg=eval_config)
            logger.info("Evaluation completed.")

        finally:
            deploy_proc.send_signal(signal.SIGINT)
            time.sleep(60)  # Wait for the server to shutdown
