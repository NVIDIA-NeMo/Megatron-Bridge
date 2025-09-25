# Using Recipes

Megatron Bridge provides production-ready training recipes for several popular models. You can find an overview of supported recipes and 🤗 HuggingFace bridges [here](index.md#supported-models).
This guide will cover the next steps to make use of a training recipe, including how to [override configuration](#overriding-configuration) and how to [launch a job](#launch-methods).

## Overriding configuration

Recipes are provided through a {py:class}`~bridge.training.config.ConfigContainer` object. This is a dataclass that holds all configuration objects needed for training. You can find a more detailed overview of the `ConfigContainer` [here](training/config-container-overview.md).
The benefit of providing the full recipe through a pythonic structure is that it is agnostic to any configuration approach that a user may prefer, whether that's YAML, `argparse` or something else. In other words, the user may override the recipe however they see fit.

The following sections detail a few different ways to override the configuration recipe. For a complete training script, please see [this example](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/recipes/llama/pretrain_llama3_8b.py).


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

## Launch methods

Megatron Bridge supports launching scripts with both `torchrun` and [NeMo-Run](https://github.com/NVIDIA-NeMo/Run).
Once your script is ready to be launched, refer to one of the following sections.

### Torchrun
Megatron Bridge training scripts can be launched with the `torchrun` command that most PyTorch users are familiar with.
Simply specify the number of GPUs to use with `--nproc-per-node` and the number of nodes with `--nnodes`. For example, on a single node:

```sh
torchrun --nnodes 1 --nproc-per-node 8 /path/to/pretrain_script.py <args to pretrain script>
```

For multi-node training, it is recommended to use a cluster orchestration system like SLURM.
The `torchrun` command should be wrapped as specified by your cluster orchestration system.
For example, with Slurm, wrap the `torchrun` command inside of `srun`:

```sh
# launch.sub

srun --nodes 2 --gpus-per-node 8 \
    --container-image <image tag> --container-mounts <mounts> \
    bash -c "
        torchrun --nnodes $SLURM_NNODES --nproc-per-node $SLURM_GPUS_PER_NODE /path/to/pretrain_script.py <args to pretrain script>
    "
```

Along with any other required flags. It is also recommended to use a NeMo Framework container with Slurm. You can find a list of container tags on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags).


### Avoiding Hangs

When working with any scripts in Megatron Bridge, please make sure you wrap your code in an `if __name__ == "__main__":`
block. Otherwise, your code may hang unexpectedly.

The reason for this is that Megatron Bridge uses Python's `multiprocessing` module in the backend when running a
multi-GPU job. The multiprocessing module will create new Python processes that will import the current module (your
script). If you did not add `__name__== "__main__"`,  then your module will spawn new processes which import the
module and then each spawn new processes. This results in an infinite loop of process spawning.

## Resources

- [OmegaConf documentation](https://omegaconf.readthedocs.io/en/2.3_branch/)
- [torchrun Documentation](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [PyTorch Multinode Training documentation](https://docs.pytorch.org/tutorials/intermediate/ddp_series_multinode.html)
