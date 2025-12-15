import os
import sys
import re
from omegaconf import OmegaConf

from megatron.bridge.recipes import *
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
)
from megatron.bridge.models.transformer_config import TransformerConfig

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout


def get_env_vars(key):
    val = os.environ.get(key, None)
    assert val is not None, f"'{key}' environment variable is not set."
    return val

def get_mbridge_base_recipe_config_builder(model_name: str):
    recipes = {
        "Qwen3-235B-A22B": qwen3_235b_a22b_pretrain_config,
        "DeepSeek-V3": deepseek_v3_pretrain_config,
    }
    model_recipe = recipes.get(model_name, None)
    if model_recipe is None:
        raise ValueError(f"Model {model_name} not supported")
    return model_recipe

def get_model_recipe():
    model_name = get_env_vars("MODEL")
    cur_dir_path = get_env_vars("MBRIDGE_PATH")
    conf_path = os.path.join(cur_dir_path, "scripts", "moe_model_zoo", "model_configs", f"{model_name}.yaml.tmp")
    assert os.path.exists(conf_path), f"Model config file not found: {conf_path}"

    base_recipe_config_builder = get_mbridge_base_recipe_config_builder(model_name)
    base_recipe = base_recipe_config_builder(
        data_paths=[get_env_vars("DATA_PATH")],
        train_iters=None,
        lr_warmup_iters=None,
        lr_decay_iters=None,
    )
    
    # Merge yaml config into base recipe
    # 1. create omegaconf dict config from base recipe
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(base_recipe)
    
    # 2. load yaml config and parse extra configs from command line overrides
    overrides = sys.argv[1:]
    yaml_config = OmegaConf.load(conf_path)
    override_conf = OmegaConf.from_dotlist(overrides)
    merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_config, override_conf)
    
    # 3. apply the final merged OmegaConf configuration back to the original ConfigContainer
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)

    # 4. apply overrides while preserving excluded fields
    apply_overrides(base_recipe, final_overrides_as_dict, excluded_fields)
    map_str_to_values(base_recipe.model)

    return base_recipe


def map_str_to_values(config: TransformerConfig):
    assert isinstance(config, TransformerConfig), "config must be a TransformerConfig"
    if isinstance(config.attention_backend, str):
        config.attention_backend = getattr(AttnBackend, config.attention_backend)
    if isinstance(config.recompute_modules, str):
        config.recompute_modules = config.recompute_modules.split(",")
    if isinstance(config.moe_layer_freq, str):
        config.moe_layer_freq = moe_freq_type(config.moe_layer_freq)
    if isinstance(config.pipeline_model_parallel_layout, str):
        config.pipeline_model_parallel_layout = PipelineParallelLayerLayout.parse_str_to_list(config.pipeline_model_parallel_layout)
    return config

def moe_freq_type(x):
    """Frequency between MoE layers and Dense layers.

    Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
      where 1 indicates an expert layer and 0 indicates a dense layer.
      This allows defining arbitrary patterns of expert and dense layers.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([0]+[1]*23)": 1 dense layer followed by 23 expert layers
          "([1]*3+[0]*2)*2": Three expert layers followed by two dense layers, repeated twice.
    """
    if isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)

def _eval_pattern(pattern):
    """ Validate and evaluate a string containing a Python list expression """
    assert isinstance(pattern, str)

    # validate input, only allow comma, digits, [, ], (, ), +, and *
    if bool(re.compile(r'[^,\d\[\]\(\)\+\*]').search(pattern)):
        raise ValueError(f"Invalid pattern: {pattern}")

    return eval(pattern)