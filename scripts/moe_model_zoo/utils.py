import os
import sys


from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_235b_a22b_pretrain_config
from megatron.bridge.training.config import runtime_config_update
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
)

from omegaconf import OmegaConf

def get_env_vars(key):
    val = os.environ.get(key, None)
    assert val is not None, f"'{key}' environment variable is not set."
    return val
    
def get_model_recipe():
    model_name = get_env_vars("MODEL")
    cur_dir_path = get_env_vars("MBRIDGE_PATH")
    conf_path = os.path.join(cur_dir_path, "scripts", "moe_model_zoo", "model_configs", f"{model_name}.yaml.tmp")
    assert os.path.exists(conf_path), f"Model config file not found: {conf_path}"

    #TODO: @pingtianl support other models
    base_recipe = qwen3_235b_a22b_pretrain_config(
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

    return base_recipe
