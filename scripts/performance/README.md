# Performance Recipes

## NOTE: This directory will change a lot over the coming weeks.


- Scripts defined in `scripts/performance` are recipes optimized for performance. These scripts can launch pre-training experiments on Slurm based clusters.

## YAML configuration files

There are YAML configuration files for supported models in `scripts/performance/configs`.
- You can override the defaul configs using these files. 
- Follow key-value conventions as present in `megatron.bridge.training.config.ConfigContainer`
- You can override any config as present in base classes in Megatron-LM.

## Example

The following line shows an example of how you can launch a pre-training experiment-

`python -m scripts.performance.setup_experiment --account <your_slurm_account> --partition <your_slurm_partition> --gpu gb200 --model_name <model name> --model_size <model_size>`

## Configuration Options

- Slurm account, partition, gpu, model name and model size are mandatory arguments for launching the experiment.
- You can use the following optional arguments as needed-

- You can use the following optional arguments as needed-
  - -l/--log_dir: Location to store your experiment artifacts and logs.
    - Make sure the environemnt variable `NEMORUN_HOME=<log_dir>` is accessible and set correctly in your virtual environment.
    - You can run `export NEMORUN_HOME=<log_dir>` in your terminal. You can add it your bashrc file (or equivalent for your OS/Linux distro) for setting it permanently.
  - -t/--time_limit: Maximum time limit for your experiment. Your slurm job will be cancelled after this. Default is 30 minutes.
  - -i/--container_image: The NeMo container you want to use. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'.
  - -c/--compute_dtype: Specifies whether you want to use bf16 or fp8 precision for training. Defaults to 'bf16'. You can choose to use 'fp8'.
  - --fp8_recipe: FP8 recipe. Options: 'ds' (per-tensor delayed scaling), 'cs '(per-tensor current scaling), 'mxfp8' (block-level scaling -- 32 values). Defaults to 'ds'.
  - -en/--enable_nsys: Enable nsys profiling. It is disabled by default. When enabled, profiling will be enabled for 1 step from step 5 to step 6. You can change the step in the respective recipe script.
  - -d/--dryrun: Using this argument will not launch the experiment. It will simply print the sbatch script to stdout. This can be helpful to verify you have set your experiment correctly as needed.
  - -g/--gpu: Target gpu type. Defaults to 'h100'. Options- 'h100', 'b200', 'gb200'.
  - -ng/--num_gpus: Number of gpus.
  - -gn/--gpus_per_node: Number of gpus per node. Defaults to 8.
  - -cm/--custom_mounts: Comma separated string of mounts.
  - -m/--model_name: model name you want to run the job for. e.g. "llama3"
  - -s/--model_size: model size you want to use for a model (family)- e.g. "8b"
  - --domain: Domain to use for the experiment- llm, vlm, diffusion. Default: llm
  - -vb/--enable_vboost: Enable VBoost which steers more power towards tensor cores. Disabled by default
  - --task: Task to run. Defaults to 'pretrain'. choices=["pretrain", "sft", "lora"]

## Virtual Environment

- For creating a virtual env on login node on a Slurm cluster, comment the following lines in `pyproject.toml` present in parent directory of this repo-
  - These lines need to be commented as they cannot be installed on login-node.

```
"megatron-core[dev,mlm]>=0.14.0a0,<0.16.0",
```

```
no-build-isolation-package = [
    "transformer-engine",
    "transformer-engine-torch",
    "mamba-ssm",
    "causal-conv1d",
    "nv-grouped-gemm",
    "flash_mla",
]
```

- From the parent directory of this repo, run the following 3 commands-

```
python -m venv bridge_venv
source bridge_venv/bin/activate
pip install .
pip install git+https://github.com/NVIDIA-NeMo/Run.git
pip install transformers
pip install git+https://github.com/NVIDIA/Megatron-LM.git@main
pip install einops
```

- The YAML config files are resolved on compute node inside the container.
