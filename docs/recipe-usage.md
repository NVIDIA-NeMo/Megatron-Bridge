# Using Recipes

Megatron Bridge provides production-ready training recipes for several popular models. You can find an overview of supported recipes and 🤗 HuggingFace bridges [here](index.md#supported-models).
This guide will cover the next steps to make use of a training recipe, including how to [override configuration](#overriding-configuration) and how to [launch a job](#launch-methods).

## Overriding configuration

Recipes are provided through a {py:class}`~bridge.training.config.ConfigContainer` object. This is a dataclass that holds all configuration objects needed for training. You can find a more detailed overview of the `ConfigContainer` [here](training/config-container-overview.md).
The benefit of providing the full recipe through a pythonic structure is that it is agnostic to any configuration approach that a user may prefer, whether that's YAML, `argparse` or something else.
Here are a few different ways to override the configuration recipe.


### Pythonic

If you prefer to manage configuration in Python, you can directly modify attributes of the `ConfigContainer`:

```python
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config

# Get the base ConfigContainer from the recipe
cfg: ConfigContainer = pretrain_config()

# Apply overrides. Note the hierarchical structure
cfg.train.train_iters = 20
cfg.train.global_batch_size = 8
cfg.train.micro_batch_size = 1
cfg.logger.log_interval = 1
```

You can also replace entire sub-configs of the `ConfigContainer`:

```python
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.models.llama import Llama3ModelProvider

cfg: ConfigContainer = pretrain_config()

small_llama = Llama3ModelProvider(
                    num_layers=2,
                    hidden_size=768,
                    ffn_hidden_size=2688,
                    num_attention_heads=16
              )
cfg.model = small_llama
```

### YAML
Overriding a configuration recipe with a YAML file can be done using OmegaConf utilities:

```python
from omegaconf import OmegaConf
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
)

cfg: ConfigContainer = pretrain_config()
yaml_filepath = "conf/llama3-8b-benchmark-cfg.yaml"

# Convert the initial Python dataclass to an OmegaConf DictConfig for merging
# excluded_fields holds some configuration that cannot be serialized into a DictConfig
merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

# Load and merge YAML overrides
yaml_overrides_omega = OmegaConf.load(yaml_filepath)
merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

# Apply overrides while preserving excluded fields
final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
apply_overrides(cfg, final_overrides_as_dict, excluded_fields)
```

The above snippet will update `cfg` with all overrides from `llama3-8b-benchmark-cfg.yaml`.

### Hydra-style

Megatron Bridge provides some utilities to update the ConfigContainer using Hydra-style CLI overrides:

```python
import sys
from omegaconf import OmegaConf
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)

cfg: ConfigContainer = pretrain_config()
cli_overrides = sys.argv[1:]

# Convert the initial Python dataclass to an OmegaConf DictConfig for merging
# excluded_fields holds some configuration that cannot be serialized into a DictConfig
merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

# Parse and merge CLI overrides
merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

# Apply overrides while preserving excluded fields
final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
apply_overrides(cfg, final_overrides_as_dict, excluded_fields)
```

After the above snippet, `cfg` will be updated with all CLI-provided overrides. 
A script containing the above code could be called like so:

```sh
torchrun <torchrun arguments> pretrain_cli_overrides.py model.tensor_model_parallel_size=4 train.train_iters=100000 ...
```
