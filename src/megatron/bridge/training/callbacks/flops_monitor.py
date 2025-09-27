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

import torch
import numpy as np

from typing import Tuple, List

from megatron.bridge.training.callbacks.utils import flops_formulas
from megatron.bridge.training.callbacks.abstract_callback import AbstractCallback

_model_flops_map = {
    "gpt3": flops_formulas.gpt3,
    "llama2": flops_formulas.llama2,
    "llama3": flops_formulas.llama3,
    "llama4": flops_formulas.llama3,
    "deepseekv3": flops_formulas.deepseekv3,
    "qwen3": flops_formulas.qwen3,
    "nemotronh": flops_formulas.nemotronh,
}


class FLOPsMonitor(AbstractCallback):
    """
    Calculate and log FLOPs per second after every train step.

    Args:
        model_config (GPTModelProvider): Model parameters.
        data_config (GPTDatasetConfig): Data module being used in the experiment.
        global_batch_size (int): global batch size.
        vocab_size (int): tokenizer vocab size.
        model_name (str): Name of the model being run. The following models are supported:
            gpt3, llama2, llama3, llama4, nemotronh, deepseek, qwen3.
    """

    higher_is_better = True

    def __init__(
        self,
        model_config,
        data_config,
        global_batch_size: int,
        vocab_size: int,
        model_name: str,
    ):
        self.model_cfg = model_config
        self.data_cfg = data_config
        # use config params only when NOT provided explicitly
        self.model = model_name

        gbs = global_batch_size
        enc_seq_len = self.data_cfg.sequence_length
        hs = self.model_cfg.hidden_size
        layers = self.model_cfg.num_layers
        ffn_hs = self.model_cfg.ffn_hidden_size
        attention_heads = self.model_cfg.num_attention_heads
        moe_router_topk = self.model_cfg.moe_router_topk
        model_pattern = getattr(self.model_cfg, "hybrid_override_pattern", None)

        # this handles both- 1. key is present, value is None; 2. key is absent
        query_groups = self.model_cfg.num_query_groups
        if query_groups is None:
            query_groups = attention_heads

        config_kwargs = {
            "gbs": gbs,
            "enc_seq_len": enc_seq_len,
            "hs": hs,
            "layers": layers,
            "ffn_hs": ffn_hs,
            "attention_heads": attention_heads,
            "moe_router_topk": moe_router_topk,
            "query_groups": query_groups,
            "vocab_size": vocab_size,
            "model_pattern": model_pattern,
        }

        from megatron.core.transformer.transformer_config import MLATransformerConfig

        if isinstance(self.model_cfg, MLATransformerConfig):
            config_kwargs["qk_head_dim"] = self.model_cfg.qk_head_dim
            config_kwargs["qk_pos_emb_head_dim"] = self.model_cfg.qk_pos_emb_head_dim
            config_kwargs["v_head_dim"] = self.model_cfg.v_head_dim
            config_kwargs["q_lora_rank"] = self.model_cfg.q_lora_rank
            config_kwargs["kv_lora_rank"] = self.model_cfg.kv_lora_rank
        config_kwargs["moe_layer_freq"] = self.model_cfg.moe_layer_freq
        config_kwargs["moe_shared_expert_intermediate_size"] = self.model_cfg.moe_shared_expert_intermediate_size
        config_kwargs["moe_ffn_hidden_size"] = self.model_cfg.moe_ffn_hidden_size
        config_kwargs["mtp_num_layers"] = self.model_cfg.mtp_num_layers

        if self.model_cfg.is_hybrid_model:
            config_kwargs['is_hybrid_model'] = True
            config_kwargs['hybrid_override_pattern'] = self.model_cfg.hybrid_override_pattern
            config_kwargs['mamba_state_dim'] = self.model_cfg.mamba_state_dim
            config_kwargs['mamba_head_dim'] = self.model_cfg.mamba_head_dim
            config_kwargs['mamba_num_groups'] = self.model_cfg.mamba_num_groups
            config_kwargs['mamba_num_heads'] = self.model_cfg.mamba_num_heads

        if self.model_cfg.window_size is not None:
            config_kwargs["window_size"] = self.model_cfg.window_size
        if getattr(self.model_cfg, "window_attn_skip_freq", None) is not None:
            config_kwargs["window_attn_skip_freq"] = self.model_cfg.window_attn_skip_freq
        if self.model_cfg.kv_channels is not None:
            config_kwargs["kv_channels"] = self.model_cfg.kv_channels

        self.flops_config = flops_formulas.FLOPSConfig(**config_kwargs)

        self.model = self.model.lower() if self.model is not None else self.model

    def track(
        self,
        iteration: int,
        writer,
        wandb_writer,
        time_per_iteration: int,
        **kwargs,
    ) -> None:
        """
        Callback hook to calculate TFLOPs per sec per GPU after training
        """
        tflops_per_gpu, flops = self.eval_tflops_per_sec_per_gpu(time_per_iteration)
        writer.add_scalar("tflops/TFLOPS_per_GPU", tflops_per_gpu, iteration)
        if wandb_writer:
            wandb_writer.log({"tflops/TFLOPS_per_GPU":  tflops_per_gpu}, iteration)

        tflops = flops / (1e12 * time_per_iteration)
        writer.add_scalar("tflops/TFLOPS", tflops, iteration)
        if wandb_writer:
            wandb_writer.log({"tflops/TFLOPS": tflops}, iteration)

    def eval_tflops_per_sec_per_gpu(self, time_per_iteration: List | float | int) -> float:
        """
        Args:
            time_per_iteration (Any[List, float, int]): Train step time (in seconds).
        Returns:
            (float): Model TFLOPs per sec per gpu
        """
        total_flops, flops_per_gpu = self.eval_model_flops()

        if not isinstance(time_per_iteration, list):
            time_per_iteration = [time_per_iteration]
        # efficient mean computation if num train steps is very large
        step_time_arr = np.array(time_per_iteration)
        time_per_iteration = np.mean(step_time_arr[len(step_time_arr) // 2 :])

        flops_per_sec_per_gpu = flops_per_gpu / (1e12 * time_per_iteration)

        return flops_per_sec_per_gpu, total_flops

    def eval_model_flops(self) -> Tuple[float, float]:
        """
        Calculate model FLOPs for a given model
        """
        if self.model is not None:
            model_matches = [model for model in _model_flops_map if model in self.model]
            self.model = model_matches[0] if len(model_matches) > 0 else self.model
        if self.model not in _model_flops_map:
            logging.info(f"FLOPs measurement supported for {list(_model_flops_map.keys())}")
            raise KeyError(f"Failed to extract valid model name from or missing FLOPs calculations for {self.model}")

        total_flops = _model_flops_map[self.model](self.flops_config)
        num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        flops_per_gpu = total_flops / num_devices

        return total_flops, flops_per_gpu
