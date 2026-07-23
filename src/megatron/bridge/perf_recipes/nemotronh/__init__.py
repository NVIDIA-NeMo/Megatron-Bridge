from megatron.bridge.perf_recipes.nemotronh.b200.nemotronh import (
    nemotron_3_nano_pretrain_8gpu_b200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_b200_bf16_config,
    nemotron_3_super_pretrain_64gpu_b200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_b200_nvfp4_config,
    nemotronh_56b_pretrain_64gpu_b200_fp8cs_config,
    nemotronh_56b_pretrain_256gpu_b200_bf16_config,
    nemotronh_56b_pretrain_256gpu_b200_fp8cs_config,
)
from megatron.bridge.perf_recipes.nemotronh.b300.nemotronh import (
    nemotron_3_nano_pretrain_8gpu_b300_bf16_config,
    nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_b300_bf16_config,
    nemotron_3_super_pretrain_64gpu_b300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_b300_nvfp4_config,
    nemotronh_56b_pretrain_64gpu_b300_fp8cs_config,
)
from megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh import (
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_gb200_bf16_config,
    nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config,
    nemotron_3_ultra_pretrain_96gpu_gb200_bf16_config,
    nemotronh_56b_pretrain_64gpu_gb200_fp8cs_config,
)
from megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh import (
    nemotron_3_nano_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_gb300_bf16_config,
    nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config,
    nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config,
    nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config,
    nemotronh_56b_pretrain_256gpu_gb300_bf16_config,
    nemotronh_56b_pretrain_256gpu_gb300_fp8cs_config,
)
from megatron.bridge.perf_recipes.nemotronh.h100.nemotronh import (
    nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
    nemotronh_56b_pretrain_64gpu_h100_fp8cs_config,
)
from megatron.bridge.perf_recipes.nemotronh.vr200.nemotronh import (
    nemotron_3_nano_pretrain_8gpu_vr200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_vr200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_vr200_nvfp4_config,
    nemotron_3_super_pretrain_64gpu_vr200_bf16_config,
    nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config,
    nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config,
)

# The DP-autotuned Ultra GB300 MXFP8 recipes are generated in the gb300 submodule
# (one config function per valid GPU count, GBS = num_gpus). Re-export them into this
# package namespace so find_perf_recipe() (getattr on the family package) resolves them.
from megatron.bridge.perf_recipes.nemotronh.gb300 import nemotronh as _ultra_gb300_mod


for _name in dir(_ultra_gb300_mod):
    if _name.startswith("nemotron_3_ultra_pretrain_") and _name.endswith("gpu_gb300_fp8mx_config"):
        globals()[_name] = getattr(_ultra_gb300_mod, _name)
del _ultra_gb300_mod, _name
