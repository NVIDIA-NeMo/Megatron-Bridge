# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron Core model provider for BAGEL-7B-MoT."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from megatron.bridge.models.bagel.modeling import (
    BagelDiffusionSubmodule,
    BagelVisionSubmodule,
    OfficialBagelVisionEncoder,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)


def gelu_pytorch_tanh(value: torch.Tensor) -> torch.Tensor:
    """Apply the connector activation used by official BAGEL."""
    return torch.nn.functional.gelu(value, approximate="tanh")


@dataclass
class BagelModelProvider(GPTModelProvider):
    """Build the Qwen2-MoT, packed SigLIP, and diffusion BAGEL model."""

    num_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_query_groups: int = 4
    ffn_hidden_size: int = 18944
    seq_length: int = 36864
    max_position_embeddings: int = 32768
    vocab_size: int = 152064
    activation_func: Any = torch.nn.functional.silu
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"
    rms_norm_eps: float = 1e-6
    layernorm_epsilon: float = 1e-6
    position_embedding_type: str = "rope"
    rotary_base: float = 1_000_000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    qk_layernorm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    persist_layer_norm: bool = True
    share_embeddings_and_output_weights: bool = False
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    pipeline_dtype: torch.dtype = torch.bfloat16
    use_flex_attention: bool = True
    bagel_repo: str | None = None
    model_path: str | None = None
    vision_model_path: str | None = None
    vae_path: str | None = None
    latent_patch_size: int = 2
    max_latent_size: int = 64
    max_num_patch_per_side: int = 70
    timestep_shift: float = 1.0
    ce_weight: float = 1.0
    mse_weight: float = 1.0
    ce_loss_reweighting: bool = False
    recompute_vit: bool = False
    freeze_vit: bool = False
    freeze_llm: bool = False
    native_model_checkpoint: str | None = None
    native_model_seed: int | None = None
    native_world_size: int | None = None
    validate_native_checkpoint_metadata: bool = True
    reset_reference_training_rng: bool = False

    def finalize(self) -> None:
        """Validate the first supported BAGEL training topology."""
        topology = (
            self.tensor_model_parallel_size,
            self.pipeline_model_parallel_size,
            self.context_parallel_size,
        )
        if topology != (1, 1, 1):
            raise ValueError(f"BAGEL currently requires TP=PP=CP=1, got {topology}")
        if self.native_model_checkpoint is None:
            if self.native_model_seed is not None or self.native_world_size is not None:
                raise ValueError("BAGEL native seed/world size require native_model_checkpoint")
            if not self.validate_native_checkpoint_metadata:
                raise ValueError("BAGEL metadata validation override requires native_model_checkpoint")
            if self.reset_reference_training_rng:
                raise ValueError("BAGEL reference RNG reset requires native_model_checkpoint")
        elif self.native_model_seed is None or self.native_world_size is None:
            raise ValueError("BAGEL native checkpoint requires native_model_seed and native_world_size")
        elif self.native_model_seed <= 0 or self.native_world_size <= 0:
            raise ValueError("BAGEL native_model_seed and native_world_size must be positive")
        self.sequence_parallel = False
        self.variable_seq_lengths = True
        super().finalize()

    def _official_config(self) -> Any:
        """Load the official local BAGEL configuration without model weights."""
        if self.bagel_repo is None or self.model_path is None:
            raise ValueError("BAGEL model requires model.bagel_repo and model.model_path")
        repo = str(Path(self.bagel_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from modeling.bagel import BagelConfig, Qwen2Config, SiglipVisionConfig

        config_path = Path(self.model_path) / "config.json"
        if not config_path.is_file():
            raise ValueError(f"BAGEL config is missing: {config_path}")
        values = json.loads(config_path.read_text(encoding="utf-8"))
        llm_config = Qwen2Config(**values["llm_config"])
        official_architecture = (
            llm_config.num_hidden_layers,
            llm_config.hidden_size,
            llm_config.num_attention_heads,
            llm_config.num_key_value_heads,
            llm_config.intermediate_size,
            llm_config.vocab_size,
        )
        provider_architecture = (
            self.num_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_query_groups,
            self.ffn_hidden_size,
            self.vocab_size,
        )
        if official_architecture != provider_architecture:
            raise ValueError("BAGEL provider architecture does not match model_path/config.json")
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.freeze_und = False
        vit_config = SiglipVisionConfig(**values["vit_config"])
        vit_config.num_hidden_layers -= 1
        vit_config.rope = False
        return BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=SimpleNamespace(**values["vae_config"]),
            latent_patch_size=self.latent_patch_size,
            max_latent_size=self.max_latent_size,
            vit_max_num_patch_per_side=self.max_num_patch_per_side,
            interpolate_pos=values.get("interpolate_pos", False),
            timestep_shift=self.timestep_shift,
        )

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> torch.nn.Module:
        """Instantiate PR #3635's MCore BAGEL model."""
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
        from megatron.core.models.bagel.bagel_mimo import BagelMimoModel
        from megatron.core.models.bagel.mcore_bagel_llm import BagelMCoreModel
        from megatron.core.models.bagel.transformer_mot_block import get_mot_layer_spec
        from megatron.core.models.mimo import MimoModelConfig
        from megatron.core.models.vision.multimodal_projector import MultimodalProjector
        from megatron.core.transformer import ModuleSpec
        from megatron.core.transformer.mlp import MLP, MLPSubmodules
        from megatron.core.transformer.transformer_config import TransformerConfig

        if self._pg_collection is None:
            raise RuntimeError("BAGEL provider requires initialized process groups")
        bagel_config = self._official_config()
        from modeling.bagel.modeling_utils import PositionEmbedding, TimestepEmbedder

        pre_process = True if pre_process is None else pre_process
        post_process = True if post_process is None else post_process

        projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=1,
            ffn_hidden_size=self.hidden_size,
            activation_func=gelu_pytorch_tanh,
            add_bias_linear=True,
            bias_activation_fusion=False,
            bf16=self.bf16,
            fp16=self.fp16,
            params_dtype=self.params_dtype,
        )
        projection_spec = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        )
        language_spec = ModuleSpec(
            module=BagelMCoreModel,
            params={
                "config": self,
                "transformer_layer_spec": get_mot_layer_spec(
                    num_experts=self.num_moe_experts,
                    moe_grouped_gemm=self.moe_grouped_gemm,
                    use_flex_attention=self.use_flex_attention,
                    qk_layernorm=True,
                    use_te=self.transformer_impl == "transformer_engine",
                ),
                "vocab_size": self.vocab_size,
                "max_sequence_length": self.seq_length,
                "pre_process": pre_process,
                "post_process": post_process,
                "share_embeddings_and_output_weights": False,
                "position_embedding_type": "rope",
                "rotary_base": self.rotary_base,
                "llm_config": bagel_config.llm_config,
                "use_flex_attention": self.use_flex_attention,
                "pg_collection": self._pg_collection,
                "vp_stage": vp_stage,
            },
        )
        vision_spec = ModuleSpec(
            module=BagelVisionSubmodule,
            params={"pg_collection": self._pg_collection},
            submodules={
                "encoders": {
                    "vision_encoder": ModuleSpec(
                        module=OfficialBagelVisionEncoder,
                        params={
                            "bagel_config": bagel_config,
                            "vision_model_path": self.vision_model_path,
                            "dtype": self.params_dtype,
                            "recompute": self.recompute_vit,
                        },
                    )
                },
                "input_projections": [
                    ModuleSpec(
                        module=MultimodalProjector,
                        params={
                            "config": projection_config,
                            "submodules": projection_spec.submodules,
                            "projector_type": "mlp",
                            "input_size": bagel_config.vit_config.hidden_size,
                        },
                    )
                ],
            },
        )
        latent_channels = getattr(bagel_config.vae_config, "z_channels", 16)
        diffusion_spec = ModuleSpec(
            module=BagelDiffusionSubmodule,
            params={"dtype": self.params_dtype, "pg_collection": self._pg_collection},
            submodules={
                "encoders": {
                    "timestep": ModuleSpec(
                        module=TimestepEmbedder,
                        params={"hidden_size": self.hidden_size},
                    ),
                    "latent_position_ids": ModuleSpec(
                        module=PositionEmbedding,
                        params={
                            "max_num_patch_per_side": self.max_latent_size,
                            "hidden_size": self.hidden_size,
                        },
                    ),
                },
                "input_projections": [
                    ModuleSpec(
                        module=torch.nn.Linear,
                        params={
                            "in_features": self.latent_patch_size**2 * latent_channels,
                            "out_features": self.hidden_size,
                            "dtype": self.params_dtype,
                        },
                    )
                ],
                "output_projections": [
                    ModuleSpec(
                        module=torch.nn.Linear,
                        params={
                            "in_features": self.hidden_size,
                            "out_features": self.latent_patch_size**2 * latent_channels,
                            "dtype": self.params_dtype,
                        },
                    )
                ],
            },
        )
        model = BagelMimoModel(
            MimoModelConfig(
                language_model_spec=language_spec,
                modality_submodules_spec={"images": vision_spec, "diffusion": diffusion_spec},
                special_token_ids={"images": 0},
            ),
            pg_collection=self._pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        output_projection = model.modality_submodules["diffusion"].output_projections[0]
        torch.nn.init.zeros_(output_projection.weight)
        if output_projection.bias is not None:
            torch.nn.init.zeros_(output_projection.bias)
        if self.native_model_checkpoint is not None:
            from examples.mimo_bagel.utils.native_checkpoint import initialize_bagel_from_native_checkpoint

            report = initialize_bagel_from_native_checkpoint(
                model,
                self.native_model_checkpoint,
                expected_model_seed=self.native_model_seed,
                expected_world_size=self.native_world_size,
                validate_metadata=self.validate_native_checkpoint_metadata,
                llm_config=bagel_config.llm_config,
            )
            logger.info(
                "Loaded native BAGEL initialization: source=%d target=%d fp32_main=%d",
                report.source_tensors_consumed,
                report.target_tensors_verified,
                report.fp32_main_tensors_preserved,
            )
        if self.freeze_vit:
            model.modality_submodules["images"].encoders["vision_encoder"].requires_grad_(False)
        if self.freeze_llm:
            model.language_model.requires_grad_(False)
        return model
