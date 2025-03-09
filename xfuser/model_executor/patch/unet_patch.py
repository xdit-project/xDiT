import torch
import torch.distributed as dist
from typing import Union, Optional, Dict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

def unet_cfg_parallel_monkey_patch_forward(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    return_dict: bool = True,
    *args,
    **kwargs
):
    assert dist.is_initialized(), "Distributed training is not initialized"

    # Initialize output_buffer and buffer_list as instance attributes if they don't exist
    if not hasattr(self, 'output_buffer'):
        self.output_buffer = None
    if not hasattr(self, 'buffer_list'):
        self.buffer_list = None
    
    b, c, h, w = sample.shape
    original_forward = type(self).forward

    rank = dist.get_rank()
    sample = sample[rank:rank+1]
    timestep = timestep[rank:rank+1] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep
    encoder_hidden_states = encoder_hidden_states[rank:rank+1]
    if added_cond_kwargs is not None:
        new_added_cond_kwargs = {}
        for k in added_cond_kwargs:
            new_added_cond_kwargs[k] = added_cond_kwargs[k][rank : rank + 1]
        added_cond_kwargs = new_added_cond_kwargs

    output = original_forward(
        self,
        sample=sample,
        timestep=timestep, 
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
        *args,
        **kwargs
    )[0]

    world_size = dist.get_world_size()
    assert world_size == 2, f"world_size is {world_size}, expected 2 in unet_cfg_parallel_monkey_patch_forward"

    if self.output_buffer is None:
        self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)
    if self.buffer_list is None:
        self.buffer_list = [torch.empty_like(output) for _ in range(world_size)]

    dist.all_gather(self.buffer_list, output.contiguous(), async_op=False)
    torch.cat(self.buffer_list[: 1], dim=2, out=self.output_buffer[0:1])
    torch.cat(self.buffer_list[1 :], dim=2, out=self.output_buffer[1:2])
    output = self.output_buffer

    if return_dict:
        output = UNet2DConditionOutput(sample=output)
    else:
        output = (output,)
    return output

def apply_unet_cfg_parallel_monkey_patch(pipe):
    """Apply the monkey patch to the pipeline's UNet if world size is 2."""
    import types
    world_size = dist.get_world_size()
    if world_size == 2:
        pipe.unet.forward = types.MethodType(unet_cfg_parallel_monkey_patch_forward, pipe.unet)
    return pipe 