from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping, ReplicatedMapping


class AmaxMapping(ReplicatedMapping):
    """Amax mapping for quantization."""
    def __init__(self, megatron_param: str, hf_param: str):
        """Initialize the Amax mapping."""
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True


def convert_to_amax_map(mappings: list[MegatronParamMapping], mapped_name='.weight_quantizer._amax') -> list[MegatronParamMapping]:
    """Convert weight mappings to amax mappings for quantization.
    
    This function converts parameter mappings for weights to their corresponding 
    amax (absolute maximum) parameter mappings used in quantization. For example:
    - "layer.weight" -> "layer.weight_quantizer._amax"
    
    Args:
        mappings: List of MegatronParamMapping objects for weight parameters
        
    Returns:
        List of new MegatronParamMapping objects for amax parameters
        
    Note:
        Only mappings with parameter names ending in '.weight' are converted.
        Other mappings are ignored.
    """
    extended_mapping = []
    
    for mapping in mappings:
        # Check if megatron_param ends with .weight
        if not mapping.megatron_param.endswith('.weight'):
            continue
            
        # Convert megatron_param: replace '.weight' with '.weight_quantizer._amax'
        new_megatron_param = mapping.megatron_param.replace('.weight', mapped_name)
        
        # Convert hf_param based on its type
        if isinstance(mapping.hf_param, dict):
            # For dict-based hf_param (e.g., QKVMapping, GatedMLPMapping)
            # Convert each value in the dictionary
            new_hf_param = {
                key: value.replace('.weight', mapped_name) if value.endswith('.weight') else value
                for key, value in mapping.hf_param.items()
            }
        elif isinstance(mapping.hf_param, str):
            # For string-based hf_param
            if mapping.hf_param.endswith('.weight'):
                new_hf_param = mapping.hf_param.replace('.weight', mapped_name)
            else:
                # If hf_param doesn't end with .weight, skip this mapping
                continue
        else:
            # Unknown hf_param type, skip
            print(f"Unknown hf_param type: {type(mapping.hf_param)}")
            continue
        
        # Amax tensors are small scalars and should not be TP-sharded. Always map
        # them as replicated parameters to avoid any TP chunking logic.
        # If hf_param is a dict (e.g., QKV/Gate-Up), pick the first entry; these
        # amax values are expected to be identical, so any entry works.
        if isinstance(new_hf_param, dict):
            if not new_hf_param:
                continue
            _, picked_hf_param = next(iter(new_hf_param.items()))
        else:
            picked_hf_param = new_hf_param

        new_mapping = AmaxMapping(
            megatron_param=new_megatron_param,
            hf_param=picked_hf_param,
        )
        extended_mapping.append(new_mapping)
    
    return extended_mapping