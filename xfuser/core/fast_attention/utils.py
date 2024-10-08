import torch
from xfuser.core.distributed import (
    get_dp_group,
    get_data_parallel_rank,
)
from diffusers import DiffusionPipeline
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from xfuser.model_executor.layers.attention_processor import xFuserAttentionBaseWrapper
from collections import Counter
import os
import json
import numpy as np


from .fast_attn_state import (
    get_fast_attn_step,
    get_fast_attn_calib,
    get_fast_attn_threshold,
    get_fast_attn_coco_path,
    get_fast_attn_use_cache,
    get_fast_attn_config_file,
    get_fast_attn_layer_name,
)

from .attn_layer import (
    xFuserFastAttention,
    FastAttnMethod,
)

from xfuser.logger import init_logger

logger = init_logger(__name__)


def save_config_file(step_methods, file_path):
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    format_data = {
        f"block{blocki}": {f"step{stepi}": method.name for stepi, method in enumerate(methods)}
        for blocki, methods in enumerate(step_methods)
    }
    with open(file_path, "w") as file:
        json.dump(format_data, file, indent=2)


def load_config_file(file_path):
    with open(file_path, "r") as file:
        format_data = json.load(file)
    steps_methods = [[FastAttnMethod[method] for method in format_method.values()] for format_method in format_data.values()]
    return steps_methods


def compression_loss(a, b):
    ls = []
    if a.__class__.__name__ == "Transformer2DModelOutput":
        a = [a.sample]
        b = [b.sample]
    weight = torch.tensor(0.0)
    for ai, bi in zip(a, b):
        if isinstance(ai, torch.Tensor):
            weight += ai.numel()
            diff = (ai - bi) / (torch.max(ai, bi) + 1e-6)
            loss = diff.abs().clip(0, 10).mean()
            ls.append(loss)
    weight_sum = get_dp_group().all_reduce(weight.clone().to(ai.device))
    local_loss = (weight / weight_sum) * (sum(ls) / len(ls))
    global_loss = get_dp_group().all_reduce(local_loss.clone().to(ai.device)).item()
    return global_loss


def transformer_forward_pre_hook(m: Transformer2DModel, args, kwargs):
    attn_name = get_fast_attn_layer_name()
    now_stepi = getattr(m.transformer_blocks[0], attn_name).stepi
    # batch_size = get_fast_attn_calib()
    # dp_degree =

    for blocki, block in enumerate(m.transformer_blocks):
        # Set `need_compute_residual` to False to avoid the process of trying different
        # compression strategies to override the saved residual.
        fast_attn = getattr(block, attn_name).processor.fast_attn
        fast_attn.need_compute_residual[now_stepi] = False
        fast_attn.need_cache_output = False
    raw_outs = m.forward(*args, **kwargs)
    for blocki, block in enumerate(m.transformer_blocks):
        if now_stepi == 0:
            continue
        fast_attn = getattr(block, attn_name).processor.fast_attn
        method_candidates = [
            FastAttnMethod.OUTPUT_SHARE,
            FastAttnMethod.RESIDUAL_WINDOW_ATTN_CFG_SHARE,
            FastAttnMethod.RESIDUAL_WINDOW_ATTN,
            FastAttnMethod.FULL_ATTN_CFG_SHARE,
        ]
        selected_method = FastAttnMethod.FULL_ATTN
        for method in method_candidates:
            # Try compress this attention using `method`
            fast_attn.steps_method[now_stepi] = method

            # Set the timestep index of every layer back to now_stepi
            # (which are increased by one in every forward)
            for _block in m.transformer_blocks:
                for layer in _block.children():
                    if isinstance(layer, xFuserAttentionBaseWrapper):
                        layer.stepi = now_stepi

            # Compute the overall transformer output
            outs = m.forward(*args, **kwargs)

            loss = compression_loss(raw_outs, outs)
            threshold = m.loss_thresholds[now_stepi][blocki]

            if loss < threshold:
                selected_method = method
                break

        fast_attn.steps_method[now_stepi] = selected_method
        del loss, outs
    del raw_outs

    # Set the timestep index of every layer back to now_stepi
    # (which are increased by one in every forward)
    for _block in m.transformer_blocks:
        for layer in _block.children():
            if isinstance(layer, xFuserAttentionBaseWrapper):
                layer.stepi = now_stepi

    for blocki, block in enumerate(m.transformer_blocks):
        # During the compression plan decision process,
        # we set the `need_compute_residual` property of all attention modules to `True`,
        # so that all full attention modules will save its residual for convenience.
        # The residual will be saved in the follow-up forward call.
        fast_attn = getattr(block, attn_name).processor.fast_attn
        fast_attn.need_compute_residual[now_stepi] = True
        fast_attn.need_cache_output = True


def select_methods(pipe: DiffusionPipeline):
    blocks = pipe.transformer.transformer_blocks
    transformer: Transformer2DModel = pipe.transformer
    attn_name = get_fast_attn_layer_name()
    n_steps = get_fast_attn_step()
    # reset all processors
    for block in blocks:
        fast_attn: xFuserFastAttention = getattr(block, attn_name).processor.fast_attn
        fast_attn.set_methods(
            [FastAttnMethod.FULL_ATTN] * n_steps,
            selecting=True,
        )

    # Setup loss threshold for each timestep and layer
    loss_thresholds = []
    for step_i in range(n_steps):
        sub_list = []
        for blocki in range(len(blocks)):
            threshold_i = (blocki + 1) / len(blocks) * get_fast_attn_threshold()
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    # calibration
    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds

    seed = 3
    guidance_scale = 4.5
    if not os.path.exists(get_fast_attn_coco_path()):
        raise FileNotFoundError(f"File {get_fast_attn_coco_path()} not found")
    with open(get_fast_attn_coco_path(), "r") as file:
        mscoco_anno = json.load(file)
    np.random.seed(seed)
    slice_ = np.random.choice(mscoco_anno["annotations"], get_fast_attn_calib())
    calib_x = [d["caption"] for d in slice_]
    pipe(
        prompt=calib_x,
        num_inference_steps=n_steps,
        generator=torch.manual_seed(seed),
        output_type="latent",
        negative_prompt="",
        return_dict=False,
        guidance_scale=guidance_scale,
    )

    hook.remove()
    del transformer.loss_thresholds

    blocks_methods = [getattr(block, attn_name).processor.fast_attn.steps_method for block in blocks]
    return blocks_methods


def set_methods(
    pipe: DiffusionPipeline,
    blocks_methods: list,
):
    attn_name = get_fast_attn_layer_name()
    blocks = pipe.transformer.transformer_blocks
    for blocki, block in enumerate(blocks):
        getattr(block, attn_name).processor.fast_attn.set_methods(blocks_methods[blocki])


def statistics(pipe: DiffusionPipeline):
    attn_name = get_fast_attn_layer_name()
    blocks = pipe.transformer.transformer_blocks
    counts = Counter([method for block in blocks for method in getattr(block, attn_name).processor.fast_attn.steps_method])
    total = sum(counts.values())
    for k, v in counts.items():
        logger.info(f"{attn_name} {k} {v/total}")


def fast_attention_compression(pipe: DiffusionPipeline):
    config_file = get_fast_attn_config_file()
    logger.info(f"config file is {config_file}")

    if get_fast_attn_use_cache() and os.path.exists(config_file):
        logger.info(f"load config file {config_file} as DiTFastAttn compression methods.")
        blocks_methods = load_config_file(config_file)
    else:
        if get_fast_attn_use_cache():
            logger.warning(f"config file {config_file} not found.")
        logger.info("start to select DiTFastAttn compression methods.")
        blocks_methods = select_methods(pipe)
        if get_data_parallel_rank() == 0:
            save_config_file(blocks_methods, config_file)
            logger.info(f"save DiTFastAttn compression methods to {config_file}")

    set_methods(pipe, blocks_methods)

    statistics(pipe)
