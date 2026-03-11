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

"""Tests for scripts/translate_mlm_to_bridge.py."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
#  Load the standalone script as a module (it is not an installed package)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "translate_mlm_to_bridge.py"
_spec = importlib.util.spec_from_file_location("translate_mlm_to_bridge", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_try_numeric = _mod._try_numeric
_try_parse_value = _mod._try_parse_value
parse_raw_args = _mod.parse_raw_args
translate = _mod.translate
TranslationResult = _mod.TranslationResult
emit_overrides = _mod.emit_overrides
parse_bridge_overrides = _mod.parse_bridge_overrides
translate_bridge_to_mlm = _mod.translate_bridge_to_mlm
ReverseTranslationResult = _mod.ReverseTranslationResult
emit_mlm_args = _mod.emit_mlm_args
_flatten_dict = _mod._flatten_dict
_format_value_for_override = _mod._format_value_for_override


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_result(**overrides: Any) -> TranslationResult:
    """Build a TranslationResult with given override key/value pairs."""
    r = TranslationResult()
    for k, v in overrides.items():
        r.add_override(k, v)
    return r


def _make_reverse_result(**args: Any) -> ReverseTranslationResult:
    """Build a ReverseTranslationResult with given mlm_args key/value pairs."""
    r = ReverseTranslationResult()
    for k, v in args.items():
        r.add_arg(k, v)
    return r


# ===========================================================================
#  Group 1 — _try_numeric
# ===========================================================================


class TestTryNumeric:
    """Tests for _try_numeric(val: str) -> int | float | str."""

    def test_integer(self):
        """Integer string returns int."""
        assert _try_numeric("42") == 42
        assert isinstance(_try_numeric("42"), int)

    def test_negative_integer(self):
        """Negative integer string returns int."""
        assert _try_numeric("-10") == -10

    def test_float(self):
        """Float string returns float."""
        assert _try_numeric("3.14") == pytest.approx(3.14)
        assert isinstance(_try_numeric("3.14"), float)

    def test_negative_float(self):
        """Negative float string returns float."""
        assert _try_numeric("-2.5") == pytest.approx(-2.5)

    def test_scientific_notation(self):
        """Scientific notation string returns float."""
        assert _try_numeric("1e-4") == pytest.approx(1e-4)

    def test_string_passthrough(self):
        """Non-numeric string is returned unchanged."""
        assert _try_numeric("cosine") == "cosine"

    def test_string_with_underscore(self):
        """Underscore string is returned as-is."""
        assert _try_numeric("bf16_mixed") == "bf16_mixed"

    def test_zero(self):
        """Zero is returned as int."""
        assert _try_numeric("0") == 0
        assert isinstance(_try_numeric("0"), int)


# ===========================================================================
#  Group 2 — _try_parse_value
# ===========================================================================


class TestTryParseValue:
    """Tests for _try_parse_value(val: str) -> Any."""

    def test_true_lowercase(self):
        """'true' parses to bool True."""
        assert _try_parse_value("true") is True

    def test_true_mixed_case(self):
        """'True' parses to bool True."""
        assert _try_parse_value("True") is True

    def test_false_lowercase(self):
        """'false' parses to bool False."""
        assert _try_parse_value("false") is False

    def test_none_lowercase(self):
        """'none' parses to None."""
        assert _try_parse_value("none") is None

    def test_null(self):
        """'null' parses to None."""
        assert _try_parse_value("null") is None

    def test_integer(self):
        """Integer string parses to int."""
        assert _try_parse_value("32") == 32

    def test_float(self):
        """Float string parses to float."""
        assert _try_parse_value("0.9") == pytest.approx(0.9)

    def test_list_literal(self):
        """List literal string parses to list."""
        assert _try_parse_value("[1,2,3]") == [1, 2, 3]

    def test_tuple_literal(self):
        """Tuple literal string parses to tuple."""
        assert _try_parse_value("(1,2,3)") == (1, 2, 3)

    def test_plain_string(self):
        """Plain word string returned as-is."""
        assert _try_parse_value("cosine") == "cosine"

    def test_moe_layer_freq_list(self):
        """MoE layer frequency list parses correctly."""
        assert _try_parse_value("[1,0,1,0]") == [1, 0, 1, 0]

    def test_empty_string(self):
        """Empty string returned as string."""
        result = _try_parse_value("")
        assert isinstance(result, str)


# ===========================================================================
#  Group 3 — parse_raw_args
# ===========================================================================


class TestParseRawArgs:
    """Tests for parse_raw_args(args_str: str) -> tuple[dict, dict]."""

    def test_simple_value_arg(self):
        """Single value arg parsed correctly."""
        args, env = parse_raw_args("--num-layers 32")
        assert args["num-layers"] == 32

    def test_multiple_args(self):
        """Multiple args parsed into dict."""
        args, _ = parse_raw_args("--num-layers 32 --hidden-size 4096")
        assert args["num-layers"] == 32
        assert args["hidden-size"] == 4096

    def test_flag_swiglu(self):
        """--swiglu (known flag) parsed as True."""
        args, _ = parse_raw_args("--swiglu")
        assert args["swiglu"] is True

    def test_flag_bf16(self):
        """--bf16 (known flag) parsed as True."""
        args, _ = parse_raw_args("--bf16")
        assert args["bf16"] is True

    def test_flag_disable_bias_linear(self):
        """--disable-bias-linear (known flag) parsed as True."""
        args, _ = parse_raw_args("--disable-bias-linear")
        assert args["disable-bias-linear"] is True

    def test_mixed_flags_and_values(self):
        """Mix of flags and value args parsed correctly."""
        args, _ = parse_raw_args("--bf16 --num-layers 32 --swiglu")
        assert args["bf16"] is True
        assert args["num-layers"] == 32
        assert args["swiglu"] is True

    def test_float_value(self):
        """Float value parsed correctly."""
        args, _ = parse_raw_args("--lr 3e-4")
        assert args["lr"] == pytest.approx(3e-4)

    def test_string_value(self):
        """String value parsed as string."""
        args, _ = parse_raw_args("--normalization RMSNorm")
        assert args["normalization"] == "RMSNorm"

    def test_returns_empty_env_vars(self):
        """Second return value is always empty dict."""
        _, env = parse_raw_args("--num-layers 32 --bf16")
        assert env == {}

    def test_sequential_flag_then_value(self):
        """Flag immediately before value arg both parsed."""
        args, _ = parse_raw_args("--swiglu --num-layers 32")
        assert args["swiglu"] is True
        assert args["num-layers"] == 32

    def test_empty_string(self):
        """Empty string produces empty dict."""
        args, env = parse_raw_args("")
        assert args == {}
        assert env == {}


# ===========================================================================
#  Group 4 — translate (MLM → Bridge)
# ===========================================================================


class TestTranslateBasicMappings:
    """Basic single-arg translate() tests."""

    def test_num_layers(self):
        """--num-layers maps to model.num_layers."""
        r = translate({"num-layers": 32})
        assert r.overrides["model.num_layers"] == 32

    def test_hidden_size(self):
        """--hidden-size maps to model.hidden_size."""
        r = translate({"hidden-size": 4096})
        assert r.overrides["model.hidden_size"] == 4096

    def test_num_attention_heads(self):
        """--num-attention-heads maps correctly."""
        r = translate({"num-attention-heads": 32})
        assert r.overrides["model.num_attention_heads"] == 32

    def test_global_batch_size(self):
        """--global-batch-size maps to train.global_batch_size."""
        r = translate({"global-batch-size": 256})
        assert r.overrides["train.global_batch_size"] == 256

    def test_lr(self):
        """--lr maps to optimizer.lr."""
        r = translate({"lr": 3e-4})
        assert r.overrides["optimizer.lr"] == pytest.approx(3e-4)

    def test_seed(self):
        """--seed maps to rng.seed."""
        r = translate({"seed": 42})
        assert r.overrides["rng.seed"] == 42

    def test_tensor_model_parallel_size(self):
        """--tensor-model-parallel-size maps correctly."""
        r = translate({"tensor-model-parallel-size": 4})
        assert r.overrides["model.tensor_model_parallel_size"] == 4

    def test_train_iters(self):
        """--train-iters maps to train.train_iters."""
        r = translate({"train-iters": 1000})
        assert r.overrides["train.train_iters"] == 1000

    def test_weight_decay(self):
        """--weight-decay maps to optimizer.weight_decay."""
        r = translate({"weight-decay": 0.1})
        assert r.overrides["optimizer.weight_decay"] == pytest.approx(0.1)


class TestTranslatePrecision:
    """Tests for precision flag translations."""

    def test_bf16(self):
        """--bf16 produces mixed_precision=bf16_mixed."""
        r = translate({"bf16": True})
        assert r.overrides["mixed_precision"] == "bf16_mixed"
        assert "mixed_precision._bf16" not in r.overrides

    def test_fp16(self):
        """--fp16 produces mixed_precision=16-mixed."""
        r = translate({"fp16": True})
        assert r.overrides["mixed_precision"] == "16-mixed"
        assert "mixed_precision._fp16" not in r.overrides

    def test_no_precision_flag(self):
        """Without bf16/fp16, no mixed_precision key emitted."""
        r = translate({"num-layers": 32})
        assert "mixed_precision" not in r.overrides

    def test_no_internal_keys_remain(self):
        """Internal ._* keys are cleaned up after processing."""
        r = translate({"bf16": True, "fp16": True})
        for key in r.overrides:
            assert "._" not in key


class TestTranslateActivations:
    """Tests for activation function translations."""

    def test_swiglu_sets_gated_and_silu(self):
        """--swiglu sets model.gated_linear_unit=True and model.activation_func=silu."""
        r = translate({"swiglu": True})
        assert r.overrides["model.gated_linear_unit"] is True
        assert r.overrides["model.activation_func"] == "silu"

    def test_swiglu_adds_note(self):
        """--swiglu adds a note about the translation."""
        r = translate({"swiglu": True})
        assert any("swiglu" in note.lower() for note in r.notes)

    def test_squared_relu(self):
        """--squared-relu sets model.activation_func=squared_relu."""
        r = translate({"squared-relu": True})
        assert r.overrides["model.activation_func"] == "squared_relu"
        assert "model.gated_linear_unit" not in r.overrides

    def test_squared_relu_adds_note(self):
        """--squared-relu adds a note."""
        r = translate({"squared-relu": True})
        assert any("squared_relu" in note for note in r.notes)


class TestTranslateFlagTransforms:
    """Tests for flag and flag_invert transforms."""

    def test_disable_bias_linear_inverts(self):
        """--disable-bias-linear produces model.add_bias_linear=False."""
        r = translate({"disable-bias-linear": True})
        assert r.overrides["model.add_bias_linear"] is False

    def test_untie_embeddings_inverts(self):
        """--untie-embeddings-and-output-weights produces share=False."""
        r = translate({"untie-embeddings-and-output-weights": True})
        assert r.overrides["model.share_embeddings_and_output_weights"] is False

    def test_sequence_parallel_flag(self):
        """--sequence-parallel produces model.sequence_parallel=True."""
        r = translate({"sequence-parallel": True})
        assert r.overrides["model.sequence_parallel"] is True

    def test_no_save_optim_inverts(self):
        """--no-save-optim produces checkpoint.save_optim=False."""
        r = translate({"no-save-optim": True})
        assert r.overrides["checkpoint.save_optim"] is False

    def test_qk_layernorm_flag(self):
        """--qk-layernorm produces model.qk_layernorm=True."""
        r = translate({"qk-layernorm": True})
        assert r.overrides["model.qk_layernorm"] is True


class TestTranslateSeqLength:
    """Tests for seq-length dual mapping."""

    def test_seq_length_maps_to_both(self):
        """--seq-length maps to both dataset.sequence_length and model.seq_length."""
        r = translate({"seq-length": 4096})
        assert r.overrides["dataset.sequence_length"] == 4096
        assert r.overrides["model.seq_length"] == 4096


class TestTranslateSkippedAndUnknown:
    """Tests for skipped and unknown arg handling."""

    def test_skip_use_mcore_models(self):
        """--use-mcore-models is skipped (not needed in Bridge)."""
        r = translate({"use-mcore-models": True})
        names = [name for name, _ in r.skipped]
        assert "use-mcore-models" in names
        assert "use-mcore-models" not in r.overrides

    def test_skip_use_flash_attn(self):
        """--use-flash-attn is skipped (Bridge default)."""
        r = translate({"use-flash-attn": True})
        names = [name for name, _ in r.skipped]
        assert "use-flash-attn" in names

    def test_skip_mock_data(self):
        """--mock-data is skipped."""
        r = translate({"mock-data": True})
        names = [name for name, _ in r.skipped]
        assert "mock-data" in names

    def test_unknown_arg(self):
        """Unrecognised arg goes to result.unknown."""
        r = translate({"totally-unknown-flag": 99})
        names = [name for name, _ in r.unknown]
        assert "totally-unknown-flag" in names

    def test_unknown_arg_not_in_overrides(self):
        """Unknown arg does not appear in overrides."""
        r = translate({"totally-unknown-flag": 99})
        assert "totally-unknown-flag" not in r.overrides


class TestTranslateEnvVars:
    """Tests for env_vars handling."""

    def test_env_vars_stored(self):
        """Passed env_vars dict is stored on result."""
        r = translate({}, env_vars={"SOME_VAR": "value"})
        assert r.env_vars["SOME_VAR"] == "value"

    def test_no_env_vars_default(self):
        """Default env_vars is empty dict."""
        r = translate({})
        assert r.env_vars == {}


class TestTranslateMlaMoe:
    """Tests for MLA/MoE detection."""

    def test_mla_detection(self):
        """--multi-latent-attention sets uses_mla=True and adds a note."""
        r = translate({"multi-latent-attention": True})
        assert r.uses_mla is True
        assert any("MLA" in note for note in r.notes)

    def test_moe_detection(self):
        """--num-experts sets uses_moe=True and adds a note."""
        r = translate({"num-experts": 8})
        assert r.uses_moe is True
        assert any("MoE" in note for note in r.notes)

    def test_no_mla_moe_by_default(self):
        """By default, uses_mla and uses_moe are False."""
        r = translate({})
        assert r.uses_mla is False
        assert r.uses_moe is False


class TestTranslateDataArgs:
    """Tests for data_path and split transforms."""

    def test_data_path_string(self):
        """String data-path stored as-is."""
        r = translate({"data-path": "/some/path"})
        assert r.overrides["dataset.data_path"] == "/some/path"

    def test_data_path_list(self):
        """List data-path joined with spaces."""
        r = translate({"data-path": ["/a", "/b"]})
        assert r.overrides["dataset.data_path"] == "/a /b"

    def test_split_string(self):
        """String split stored as-is."""
        r = translate({"split": "900,50,50"})
        assert r.overrides["dataset.split"] == "900,50,50"

    def test_split_tuple(self):
        """Tuple split joined with commas."""
        r = translate({"split": (900, 50, 50)})
        assert r.overrides["dataset.split"] == "900,50,50"


class TestTranslateSequenceParallelNote:
    """Tests for the sequence_parallel + TP warning."""

    def test_sp_with_tp1_warns(self):
        """sequence-parallel without TP > 1 adds a note."""
        r = translate({"sequence-parallel": True})
        assert any("sequence_parallel" in note or "tensor_model_parallel" in note for note in r.notes)

    def test_sp_with_tp2_no_warn(self):
        """sequence-parallel with TP=2 does not add the warning note."""
        r = translate({"sequence-parallel": True, "tensor-model-parallel-size": 2})
        # No SP note about TP requirement
        sp_notes = [n for n in r.notes if "requires tensor_model_parallel" in n]
        assert len(sp_notes) == 0


# ===========================================================================
#  Group 5 — _format_value_for_override
# ===========================================================================


class TestFormatValueForOverride:
    """Tests for _format_value_for_override(val, key='')."""

    def test_bool_true(self):
        """True formats as 'true'."""
        assert _format_value_for_override(True) == "true"

    def test_bool_false(self):
        """False formats as 'false'."""
        assert _format_value_for_override(False) == "false"

    def test_none(self):
        """None formats as 'null'."""
        assert _format_value_for_override(None) == "null"

    def test_integer(self):
        """Integer formats as its string representation."""
        assert _format_value_for_override(32) == "32"

    def test_float(self):
        """Float formats without extra quoting."""
        assert _format_value_for_override(3.14) == "3.14"

    def test_plain_string_no_special_chars(self):
        """Plain string without spaces or commas returned as-is."""
        assert _format_value_for_override("cosine") == "cosine"

    def test_string_with_space(self):
        """String with space is single-quoted."""
        result = _format_value_for_override("some path")
        assert result.startswith("'") and result.endswith("'")

    def test_string_with_comma(self):
        """String with comma is single-quoted."""
        result = _format_value_for_override("900,50,50")
        assert result.startswith("'") and result.endswith("'")

    def test_split_key_double_quoted(self):
        """dataset.split key gets double+single quoting."""
        result = _format_value_for_override("900,50,50", key="dataset.split")
        assert '"' in result

    def test_list_val(self):
        """List formatted as repr."""
        assert _format_value_for_override([1, 2, 3]) == "[1, 2, 3]"


# ===========================================================================
#  Group 6 — emit_overrides
# ===========================================================================


class TestEmitOverrides:
    """Tests for emit_overrides(result) -> str."""

    def test_contains_header(self):
        """Output always contains the header comment."""
        r = _make_result()
        out = emit_overrides(r)
        assert "Megatron Bridge Overrides" in out

    def test_single_override(self):
        """Override key=value appears in output."""
        r = _make_result(**{"model.num_layers": 32})
        out = emit_overrides(r)
        assert "model.num_layers=32" in out

    def test_bool_formatted_lowercase(self):
        """Bool overrides use lowercase true/false."""
        r = _make_result(**{"model.add_bias_linear": False})
        out = emit_overrides(r)
        assert "model.add_bias_linear=false" in out

    def test_unknown_args_in_comment(self):
        """Unknown args appear as comments."""
        r = TranslationResult()
        r.unknown.append(("my-flag", 99))
        out = emit_overrides(r)
        assert "Unknown" in out
        assert "my-flag" in out

    def test_skipped_args_in_comment(self):
        """Skipped args appear as comments."""
        r = TranslationResult()
        r.skipped.append(("use-flash-attn", True))
        out = emit_overrides(r)
        assert "Skipped" in out

    def test_notes_appear_as_comments(self):
        """Notes appear prefixed with '# NOTE:'."""
        r = TranslationResult()
        r.add_note("some important note")
        out = emit_overrides(r)
        assert "# NOTE: some important note" in out

    def test_section_label_for_model(self):
        """model.* overrides include 'Model Architecture' section label."""
        r = _make_result(**{"model.hidden_size": 4096})
        out = emit_overrides(r)
        assert "Model Architecture" in out

    def test_mixed_precision_section_label(self):
        """mixed_precision override includes 'Mixed Precision' label."""
        r = _make_result(**{"mixed_precision": "bf16_mixed"})
        out = emit_overrides(r)
        assert "Mixed Precision" in out


# ===========================================================================
#  Group 8 — parse_bridge_overrides
# ===========================================================================


class TestParseBridgeOverrides:
    """Tests for parse_bridge_overrides(overrides_str) -> dict."""

    def test_simple_pair(self):
        """Single key=value pair parsed."""
        result = parse_bridge_overrides("model.num_layers=32")
        assert result["model.num_layers"] == 32

    def test_multiple_pairs(self):
        """Multiple key=value pairs parsed."""
        result = parse_bridge_overrides("model.num_layers=32 mixed_precision=bf16_mixed")
        assert result["model.num_layers"] == 32
        assert result["mixed_precision"] == "bf16_mixed"

    def test_bool_true(self):
        """Boolean 'true' parsed as True."""
        result = parse_bridge_overrides("model.add_bias_linear=true")
        assert result["model.add_bias_linear"] is True

    def test_bool_false(self):
        """Boolean 'false' parsed as False."""
        result = parse_bridge_overrides("model.add_bias_linear=false")
        assert result["model.add_bias_linear"] is False

    def test_float_value(self):
        """Float value parsed correctly."""
        result = parse_bridge_overrides("optimizer.lr=3e-4")
        assert result["optimizer.lr"] == pytest.approx(3e-4)

    def test_string_value(self):
        """String value remains a string."""
        result = parse_bridge_overrides("model.normalization=RMSNorm")
        assert result["model.normalization"] == "RMSNorm"

    def test_skips_tokens_without_equals(self):
        """Tokens without '=' are ignored."""
        result = parse_bridge_overrides("model.num_layers=32 invalid_token model.hidden_size=4096")
        assert "invalid_token" not in result
        assert result["model.num_layers"] == 32
        assert result["model.hidden_size"] == 4096

    def test_empty_string(self):
        """Empty input produces empty dict."""
        assert parse_bridge_overrides("") == {}

    def test_none_value(self):
        """'none' value parsed as None."""
        result = parse_bridge_overrides("model.something=none")
        assert result["model.something"] is None


# ===========================================================================
#  Group 9 — translate_bridge_to_mlm (Bridge → MLM)
# ===========================================================================


class TestTranslateBridgeToMlmPrecision:
    """Tests for precision reverse translation."""

    def test_bf16_mixed_to_bf16_flag(self):
        """mixed_precision=bf16_mixed → --bf16 flag."""
        r = translate_bridge_to_mlm({"mixed_precision": "bf16_mixed"})
        assert "bf16" in r.mlm_args

    def test_16_mixed_to_fp16_flag(self):
        """mixed_precision=16-mixed → --fp16 flag."""
        r = translate_bridge_to_mlm({"mixed_precision": "16-mixed"})
        assert "fp16" in r.mlm_args

    def test_fp16_mixed_alias(self):
        """mixed_precision=fp16_mixed → --fp16 flag."""
        r = translate_bridge_to_mlm({"mixed_precision": "fp16_mixed"})
        assert "fp16" in r.mlm_args

    def test_unknown_precision_adds_note(self):
        """Unknown mixed_precision value generates a note."""
        r = translate_bridge_to_mlm({"mixed_precision": "8-mixed"})
        assert any("8-mixed" in note for note in r.notes)


class TestTranslateBridgeToMlmActivation:
    """Tests for activation function reverse translation."""

    def test_silu_plus_gated_to_swiglu(self):
        """silu + gated_linear_unit=True → --swiglu."""
        r = translate_bridge_to_mlm({"model.activation_func": "silu", "model.gated_linear_unit": True})
        assert "swiglu" in r.mlm_args

    def test_squared_relu_to_flag(self):
        """squared_relu activation → --squared-relu."""
        r = translate_bridge_to_mlm({"model.activation_func": "squared_relu"})
        assert "squared-relu" in r.mlm_args

    def test_gelu_no_swiglu(self):
        """gelu activation does not produce --swiglu or --squared-relu."""
        r = translate_bridge_to_mlm({"model.activation_func": "gelu"})
        assert "swiglu" not in r.mlm_args
        assert "squared-relu" not in r.mlm_args

    def test_silu_without_gated_not_swiglu(self):
        """silu without gated_linear_unit does not produce --swiglu."""
        r = translate_bridge_to_mlm({"model.activation_func": "silu", "model.gated_linear_unit": False})
        assert "swiglu" not in r.mlm_args


class TestTranslateBridgeToMlmSeqLength:
    """Tests for seq-length reverse translation."""

    def test_model_seq_length(self):
        """model.seq_length → --seq-length."""
        r = translate_bridge_to_mlm({"model.seq_length": 4096})
        assert r.mlm_args.get("seq-length") == 4096

    def test_dataset_sequence_length(self):
        """dataset.sequence_length → --seq-length."""
        r = translate_bridge_to_mlm({"dataset.sequence_length": 4096})
        assert r.mlm_args.get("seq-length") == 4096

    def test_both_seq_length_not_duplicated(self):
        """Both seq_length fields produce only one --seq-length entry."""
        r = translate_bridge_to_mlm({"model.seq_length": 4096, "dataset.sequence_length": 4096})
        count = list(r.mlm_args.keys()).count("seq-length")
        assert count == 1


class TestTranslateBridgeToMlmFlagInvert:
    """Tests for flag_invert reverse translations."""

    def test_add_bias_linear_false_to_disable_flag(self):
        """model.add_bias_linear=False → --disable-bias-linear."""
        r = translate_bridge_to_mlm({"model.add_bias_linear": False})
        assert "disable-bias-linear" in r.mlm_args

    def test_add_bias_linear_true_no_flag(self):
        """model.add_bias_linear=True → no --disable-bias-linear."""
        r = translate_bridge_to_mlm({"model.add_bias_linear": True})
        assert "disable-bias-linear" not in r.mlm_args


class TestTranslateBridgeToMlmDataArgs:
    """Tests for data_path and split reverse translation."""

    def test_data_path_string(self):
        """String dataset.data_path → --data-path."""
        r = translate_bridge_to_mlm({"dataset.data_path": "/path/to/data"})
        assert r.mlm_args.get("data-path") == "/path/to/data"

    def test_data_path_list(self):
        """List dataset.data_path joined with spaces."""
        r = translate_bridge_to_mlm({"dataset.data_path": ["/a", "/b"]})
        assert r.mlm_args.get("data-path") == "/a /b"

    def test_split_string(self):
        """dataset.split → --split."""
        r = translate_bridge_to_mlm({"dataset.split": "900,50,50"})
        assert r.mlm_args.get("split") == "900,50,50"


class TestTranslateBridgeToMlmBridgeOnlyKeys:
    """Tests for Bridge-only keys that should be skipped."""

    def test_ignore_rng_key(self):
        """Top-level 'rng' key is skipped."""
        r = translate_bridge_to_mlm({"rng": "some_val"})
        skipped_keys = [k for k, _ in r.skipped]
        assert "rng" in skipped_keys

    def test_ignore_model_timers(self):
        """model.timers is skipped."""
        r = translate_bridge_to_mlm({"model.timers": True})
        skipped_keys = [k for k, _ in r.skipped]
        assert "model.timers" in skipped_keys

    def test_ignore_target_underscore_key(self):
        """Keys ending with ._target_ are skipped."""
        r = translate_bridge_to_mlm({"model._target_": "some.class"})
        skipped_keys = [k for k, _ in r.skipped]
        assert "model._target_" in skipped_keys


# ===========================================================================
#  Group 10 — emit_mlm_args
# ===========================================================================


class TestEmitMlmArgs:
    """Tests for emit_mlm_args(result) -> str."""

    def test_contains_header(self):
        """Output contains the header block."""
        r = _make_reverse_result()
        out = emit_mlm_args(r)
        assert "Megatron-LM" in out

    def test_flag_arg_without_value(self):
        """Flag arg (value=None) emitted as '--flag' with no value."""
        r = _make_reverse_result(bf16=None)
        out = emit_mlm_args(r)
        assert "--bf16" in out

    def test_value_arg_with_value(self):
        """Value arg emitted as '--name value'."""
        r = _make_reverse_result(**{"num-layers": 32})
        out = emit_mlm_args(r)
        assert "--num-layers 32" in out

    def test_notes_emitted_as_comments(self):
        """Notes appear as '# NOTE:' comments."""
        r = ReverseTranslationResult()
        r.add_note("some note")
        out = emit_mlm_args(r)
        assert "# NOTE: some note" in out

    def test_unknown_keys_in_comment(self):
        """Unknown keys appear in a comment block."""
        r = ReverseTranslationResult()
        r.unknown.append(("foo.bar", 99))
        out = emit_mlm_args(r)
        assert "Unknown" in out
        assert "foo.bar" in out

    def test_skipped_count_in_comment(self):
        """Skipped entries noted with count."""
        r = ReverseTranslationResult()
        r.skipped.append(("rng", None))
        out = emit_mlm_args(r)
        assert "Skipped" in out


# ===========================================================================
#  Group 12 — _flatten_dict
# ===========================================================================


class TestFlattenDict:
    """Tests for _flatten_dict(d, prefix='') -> dict."""

    def test_flat_dict_unchanged(self):
        """Flat dict returned as-is."""
        d = {"a": 1, "b": 2}
        assert _flatten_dict(d) == d

    def test_single_level_nesting(self):
        """Single nested level uses dot separator."""
        assert _flatten_dict({"model": {"num_layers": 32}}) == {"model.num_layers": 32}

    def test_deep_nesting(self):
        """Deep nesting produces multi-level dotted key."""
        assert _flatten_dict({"a": {"b": {"c": 99}}}) == {"a.b.c": 99}

    def test_mixed_levels(self):
        """Mix of flat and nested keys handled correctly."""
        result = _flatten_dict({"top": 1, "nested": {"val": 2}})
        assert result == {"top": 1, "nested.val": 2}

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert _flatten_dict({}) == {}

    def test_list_value_not_flattened(self):
        """List value treated as leaf, not further flattened."""
        result = _flatten_dict({"a": [1, 2, 3]})
        assert result == {"a": [1, 2, 3]}

    def test_none_value_preserved(self):
        """None values are preserved as leaves."""
        result = _flatten_dict({"key": None})
        assert result == {"key": None}


# ===========================================================================
#  Group 13 — Round-trip: MLM → Bridge (integration)
# ===========================================================================


class TestRoundTripMlmToBridge:
    """Integration: parse_raw_args → translate → emit_overrides → verify."""

    def test_round_trip_basic_arch(self):
        """Core architecture args survive round-trip."""
        args, env = parse_raw_args("--num-layers 32 --hidden-size 4096 --num-attention-heads 32")
        r = translate(args, env)
        assert r.overrides["model.num_layers"] == 32
        assert r.overrides["model.hidden_size"] == 4096
        assert r.overrides["model.num_attention_heads"] == 32

    def test_round_trip_bf16(self):
        """--bf16 survives round-trip with no internal keys leaked."""
        args, env = parse_raw_args("--bf16 --num-layers 32")
        r = translate(args, env)
        assert r.overrides.get("mixed_precision") == "bf16_mixed"
        assert all("._" not in k for k in r.overrides)

    def test_round_trip_swiglu(self):
        """--swiglu correctly expands to two Bridge overrides."""
        args, env = parse_raw_args("--swiglu --num-layers 32")
        r = translate(args, env)
        assert r.overrides.get("model.gated_linear_unit") is True
        assert r.overrides.get("model.activation_func") == "silu"

    def test_round_trip_disable_bias_linear(self):
        """--disable-bias-linear produces model.add_bias_linear=false."""
        args, env = parse_raw_args("--disable-bias-linear")
        r = translate(args, env)
        assert r.overrides.get("model.add_bias_linear") is False

    def test_round_trip_seq_length(self):
        """--seq-length maps to both dataset and model fields."""
        args, env = parse_raw_args("--seq-length 4096")
        r = translate(args, env)
        assert r.overrides.get("dataset.sequence_length") == 4096
        assert r.overrides.get("model.seq_length") == 4096


# ===========================================================================
#  Group 14 — Round-trip: Bridge → MLM (integration)
# ===========================================================================


class TestRoundTripBridgeToMlm:
    """Integration: parse_bridge_overrides → translate_bridge_to_mlm → verify."""

    def test_bridge_to_mlm_bf16(self):
        """bf16_mixed → --bf16."""
        overrides = parse_bridge_overrides("mixed_precision=bf16_mixed")
        r = translate_bridge_to_mlm(overrides)
        assert "bf16" in r.mlm_args

    def test_bridge_to_mlm_swiglu(self):
        """silu + gated → --swiglu."""
        overrides = parse_bridge_overrides("model.activation_func=silu model.gated_linear_unit=true")
        r = translate_bridge_to_mlm(overrides)
        assert "swiglu" in r.mlm_args

    def test_bridge_to_mlm_num_layers(self):
        """model.num_layers → --num-layers."""
        overrides = parse_bridge_overrides("model.num_layers=32")
        r = translate_bridge_to_mlm(overrides)
        assert r.mlm_args.get("num-layers") == 32

    def test_bridge_to_mlm_seq_length(self):
        """dataset.sequence_length → --seq-length."""
        overrides = parse_bridge_overrides("dataset.sequence_length=4096")
        r = translate_bridge_to_mlm(overrides)
        assert r.mlm_args.get("seq-length") == 4096
