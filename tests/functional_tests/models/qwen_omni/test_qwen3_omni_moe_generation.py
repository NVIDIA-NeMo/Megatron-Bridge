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

"""
Functional tests for Qwen3 Omni Moe HF to Megatron generation.

Example run commands:
    # Run the generation test
    pytest tests/functional_tests/models/qwen_omni/test_qwen3_omni_moe_generation.py

Note: This test use small proxy/toy models for fast generation testing.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import (
    AutoTokenizer,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
)
from transformers.models.qwen3_omni_moe import Qwen3OmniMoeConfig


HF_QWEN3_OMNI_MOE_TOY_MODEL_CONFIG = {
    "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
    "assistant_token_id": 77091,
    "code2wav_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "codebook_dim": 512,
        "codebook_size": 2048,
        "decoder_dim": 1536,
        "hidden_act": "silu",
        "hidden_size": 256,
        "intermediate_size": 1024,
        "layer_scale_initial_scale": 0.01,
        "max_position_embeddings": 2000,
        "model_type": "",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_quantizers": 4,
        "num_semantic_quantizers": 1,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000,
        "semantic_codebook_size": 4096,
        "sliding_window": 18,
        "upsample_rates": [8, 5, 4, 3],
        "upsampling_ratios": [2, 2],
        "vector_quantization_hidden_dimension": 128,
    },
    "dtype": "bfloat16",
    "enable_audio_output": True,
    "im_end_token_id": 151645,
    "im_start_token_id": 151644,
    "model_type": "qwen3_omni_moe",
    "system_token_id": 8948,
    "talker_config": {
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0,
            "decoder_sparse_step": 1,
            "head_dim": 16,
            "hidden_act": "silu",
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 512,
            "max_position_embeddings": 16384,
            "mlp_only_layers": [],
            "moe_intermediate_size": 128,
            "norm_topk_prob": True,
            "num_attention_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "interleaved": True,
                "mrope_section": [16, 24, 24],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000,
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": 256,
            "sliding_window": None,
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 3072,
        },
        "accept_hidden_layer": 6,
        "audio_end_token_id": 151670,
        "audio_start_token_id": 151669,
        "audio_token_id": 151675,
        "code_predictor_config": {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_bias": False,
            "attention_dropout": 0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": True,
            "dtype": None,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "head_dim": 16,
            "hidden_act": "silu",
            "hidden_size": 256,
            "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {"LABEL_0": 0, "LABEL_1": 1},
            "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 2048,
            "max_window_layers": 28,
            "min_length": 0,
            "model_type": "qwen3_omni_moe_talker_code_predictor",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 4,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_code_groups": 16,
            "num_hidden_layers": 5,
            "num_key_value_heads": 2,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": None,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000,
            "sep_token_id": None,
            "sliding_window": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": False,
            "tokenizer_class": None,
            "top_k": 10,
            "top_p": 1.0,
            "torchscript": False,
            "typical_p": 1.0,
            "use_bfloat16": False,
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 2048,
        },
        "codec_bos_id": 2149,
        "codec_eos_token_id": 2150,
        "codec_nothink_id": 2155,
        "codec_pad_id": 2148,
        "codec_think_bos_id": 2156,
        "codec_think_eos_id": 2157,
        "image_token_id": 151655,
        "model_type": "qwen3_omni_moe_talker",
        "num_code_groups": 16,
        "output_router_logits": False,
        "position_id_per_seconds": 13,
        "seconds_per_chunk": 2,
        "spatial_merge_size": 2,
        "speaker_id": {"chelsie": 2301, "ethan": 2302, "aiden": 2303},
        "thinker_hidden_size": 512,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
    },
    "thinker_config": {
        "audio_config": {
            "_name_or_path": "",
            "activation_dropout": 0,
            "activation_function": "gelu",
            "add_cross_attention": False,
            "architectures": None,
            "attention_dropout": 0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "conv_chunksize": 500,
            "cross_attention_hidden_size": None,
            "d_model": 1280,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": True,
            "downsample_hidden_size": 480,
            "dropout": 0,
            "dtype": None,
            "early_stopping": False,
            "encoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "encoder_layers": 32,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
            "initializer_range": 0.02,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {"LABEL_0": 0, "LABEL_1": 1},
            "length_penalty": 1.0,
            "max_length": 20,
            "max_source_positions": 1500,
            "min_length": 0,
            "model_type": "qwen3_omni_moe_audio_encoder",
            "n_window": 50,
            "n_window_infer": 800,
            "no_repeat_ngram_size": 0,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_hidden_layers": 32,
            "num_mel_bins": 128,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_dim": 2048,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": None,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "scale_embedding": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 10,
            "top_p": 1.0,
            "torchscript": False,
            "typical_p": 1.0,
            "use_bfloat16": False,
        },
        "audio_end_token_id": 151670,
        "audio_start_token_id": 151669,
        "audio_token_id": 151675,
        "dtype": "bfloat16",
        "image_token_id": 151655,
        "initializer_range": 0.02,
        "model_type": "qwen3_omni_moe_thinker",
        "position_id_per_seconds": 13,
        "seconds_per_chunk": 2,
        "text_config": {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": None,
            "decoder_sparse_step": 1,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": True,
            "dtype": None,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 512,
            "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
            "initializer_range": 0.02,
            "intermediate_size": 384,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {"LABEL_0": 0, "LABEL_1": 1},
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 16384,
            "min_length": 0,
            "mlp_only_layers": [],
            "model_type": "qwen3_omni_moe_text",
            "moe_intermediate_size": 384,
            "no_repeat_ngram_size": 0,
            "norm_topk_prob": True,
            "num_attention_heads": 8,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 8,
            "num_key_value_heads": 2,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_router_logits": False,
            "output_scores": False,
            "pad_token_id": None,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "interleaved": True,
                "mrope_interleaved": True,
                "mrope_section": [16, 24, 24],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000,
            "router_aux_loss_coef": 0.001,
            "sep_token_id": None,
            "shared_expert_intermediate_size": 0,
            "sliding_window": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": False,
            "tokenizer_class": None,
            "top_k": 10,
            "top_p": 1.0,
            "torchscript": False,
            "typical_p": 1.0,
            "use_bfloat16": False,
            "use_cache": True,
            "use_qk_norm": True,
            "use_sliding_window": False,
            "vocab_size": 152064,
        },
        "user_token_id": 872,
        "video_token_id": 151656,
        "vision_config": {
            "_name_or_path": "",
            "add_cross_attention": False,
            "apply_vit_abs_pos_embed": True,
            "architectures": None,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "deepstack_visual_indexes": [1, 2, 3],
            "depth": 27,
            "diversity_penalty": 0.0,
            "do_sample": True,
            "dtype": None,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 288,
            "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
            "image_size": 384,
            "in_channels": 3,
            "in_chans": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {"LABEL_0": 0, "LABEL_1": 1},
            "length_penalty": 1.0,
            "max_length": 20,
            "min_length": 0,
            "model_type": "qwen3_omni_moe_vision_encoder",
            "no_repeat_ngram_size": 0,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_heads": 2,
            "num_return_sequences": 1,
            "out_hidden_size": 2048,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": None,
            "patch_size": 16,
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "spatial_merge_size": 2,
            "spatial_patch_size": 16,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "temporal_patch_size": 2,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "tokens_per_second": 2,
            "top_k": 10,
            "top_p": 1.0,
            "torchscript": False,
            "typical_p": 1.0,
            "use_bfloat16": False,
        },
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652,
    },
    "transformers_version": "4.57.0.dev0",
    "tts_bos_token_id": 151672,
    "tts_eos_token_id": 151673,
    "tts_pad_token_id": 151671,
    "user_token_id": 872,
}


class TestQwen3OmniMoeGeneration:
    """
    Test Qwen3 Omni Moe model generation using HF to Megatron conversion with vision inputs.
    Uses small proxy/toy models for fast generation testing.
    """

    @pytest.fixture(scope="class")
    def qwen3_omni_moe_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Qwen3 Omni MoE toy model to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace MoE model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("qwen3_omni_moe_generation_toy_model")
        model_dir = temp_dir / "qwen3_omni_moe_toy"

        # Create Qwen3 VL MoE config from the toy model config
        config = Qwen3OmniMoeConfig(**HF_QWEN3_OMNI_MOE_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16

        # Set rope_scaling on text_config
        if hasattr(config.thinker_config, "text_config") and config.thinker_config.text_config is not None:
            config.thinker_config.text_config.rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}

        # Create model with random weights and convert to bfloat16
        model = Qwen3OmniMoeForConditionalGeneration(config)
        model = model.to(dtype=torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
        tokenizer.save_pretrained(model_dir)

        # Also save the image processor
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
        processor.save_pretrained(model_dir)

        model.save_pretrained(model_dir, safe_serialization=True)
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_QWEN3_OMNI_MOE_TOY_MODEL_CONFIG, f, indent=2)

        print(f"Created MoE toy model at: {model_dir}")
        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_omni_moe_image_generation(self, qwen3_omni_moe_toy_model_path):
        """
        Test Qwen3 Omni MoE toy model with image generation and EP=2.
        Uses a small proxy MoE model instead of the full 30B model for fast testing.
        Uses real image to test vision-language pipeline with corrected vision config.

        Args:
            qwen3_omni_moe_toy_model_path: Path to the toy Qwen3 Omni MoE model (from fixture)
        """
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "examples/conversion/hf_to_megatron_generate_vlm.py",
            f"--hf_model_path={qwen3_omni_moe_toy_model_path}",
            "--prompt=Describe this image.",
            "--ep=2",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent,
            )

            # Print output for debugging
            print("\n" + "=" * 80)
            print("STDOUT:")
            print(result.stdout)
            print("\n" + "=" * 80)
            print("STDERR:")
            print(result.stderr)
            print("=" * 80 + "\n")

            if result.returncode != 0:
                assert False, f"Qwen3 Omni MoE toy model generation failed with return code {result.returncode}"

            print("SUCCESS: Qwen3 Omni MoE toy model generation test completed successfully")

        except subprocess.TimeoutExpired:
            assert False, "Qwen3 Omni MoE toy model generation test timed out after 5 minutes"
        except Exception as e:
            print(f"Error during Qwen3 Omni MoE toy model generation test: {e}")
            raise
