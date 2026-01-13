from functools import partial

import torch
import torch.nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import ShardingStrategy


def shard_model(
    model,
    dtype=torch.bfloat16,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    min_num_params=1e3,  # Minimum parameters to benefit from sharding
):
    # Bottom-up approach: shard individual modules that are in target dtype
    for name, module in model.named_modules():
        # Skip the root model itself
        if module is model:
            continue
        
        # Only consider leaf modules (modules with parameters but no submodules with parameters)
        has_params = any(True for _ in module.parameters(recurse=False))
        if not has_params:
            continue
        
        # Check if all parameters in this module are in the target dtype
        params_list = list(module.parameters(recurse=False))
        if not params_list:
            continue
            
        all_target_dtype = all(p.dtype == dtype for p in params_list)
        if not all_target_dtype:
            continue
        
        # Check if module is large enough to benefit from sharding
        num_params = sum(p.numel() for p in params_list)
        if num_params < min_num_params:
            continue
        
        # Apply FSDP2 to this module
        fully_shard(
            module,
            mesh=process_group,
            reshard_after_forward=sharding_strategy == ShardingStrategy.FULL_SHARD,
        )
    
    return model
