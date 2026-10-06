import types
import weakref

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from xfuser.core.distributed import get_cfg_group


def _cfg_shard(value, batch_size: int, rank: int):
    """Return this CFG rank's half of every tensor that carries the UNet batch."""
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            half = batch_size // 2
            return value[rank * half : (rank + 1) * half]
        return value
    if isinstance(value, dict):
        return {key: _cfg_shard(item, batch_size, rank) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cfg_shard(item, batch_size, rank) for item in value)
    return value


def unet_cfg_parallel_monkey_patch_forward(
    self,
    sample: torch.Tensor,
    timestep,
    encoder_hidden_states: torch.Tensor,
    *args,
    return_dict: bool = True,
    **kwargs,
):
    """Run the UNet on one CFG half of the batch and gather both halves.

    Diffusers stacks the batch as ``[unconditional, conditional]`` when
    classifier-free guidance is on, so each of the two CFG ranks runs one half
    and the halves are gathered back in that order. Without CFG the batch holds
    only conditional samples and every rank runs the whole batch.
    """
    original_forward = type(self).forward
    batch_size = sample.shape[0]
    pipeline = self._xfuser_cfg_pipeline()
    split = (
        pipeline is not None
        and getattr(pipeline, "_guidance_scale", None) is not None
        and pipeline.do_classifier_free_guidance
    )
    if not split:
        return original_forward(self, sample, timestep, encoder_hidden_states, *args, return_dict=return_dict, **kwargs)

    cfg_group = get_cfg_group()
    rank = cfg_group.rank_in_group
    output = original_forward(
        self,
        _cfg_shard(sample, batch_size, rank),
        _cfg_shard(timestep, batch_size, rank),
        _cfg_shard(encoder_hidden_states, batch_size, rank),
        *_cfg_shard(args, batch_size, rank),
        return_dict=False,
        **_cfg_shard(kwargs, batch_size, rank),
    )[0]
    output = cfg_group.all_gather(output.contiguous(), dim=0)

    if return_dict:
        return UNet2DConditionOutput(sample=output)
    return (output,)


def apply_unet_cfg_parallel_monkey_patch(pipe):
    """Split the UNet batch across the two ranks of the CFG parallel group."""
    pipe.unet._xfuser_cfg_pipeline = weakref.ref(pipe)
    pipe.unet.forward = types.MethodType(unet_cfg_parallel_monkey_patch_forward, pipe.unet)
    return pipe
