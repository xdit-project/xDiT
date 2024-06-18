import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from torch import distributed as dist, nn

from .base_model import BaseModel, BaseModule
from pipefuser.modules.dit.tensor_parallel.attention import DistriAttentionTP
from pipefuser.modules.dit.tensor_parallel.conv2d import DistriConv2dTP
from pipefuser.modules.dit.tensor_parallel.feed_forward import DistriFeedForwardTP
from pipefuser.modules.dit.tensor_parallel.resnet import DistriResnetBlock2DTP
from pipefuser.utils import DistriConfig


class DistriSDXLUNetTP(BaseModel):  # for Patch Parallelism
    def __init__(self, model: UNet2DConditionModel, distri_config: DistriConfig):
        assert isinstance(model, UNet2DConditionModel)
        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, Attention):
                        wrapped_submodule = DistriAttentionTP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, FeedForward):
                        wrapped_submodule = DistriFeedForwardTP(
                            submodule, distri_config
                        )
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, ResnetBlock2D):
                        wrapped_submodule = DistriResnetBlock2DTP(
                            submodule, distri_config
                        )
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, nn.Conv2d) and (
                        subname == "conv_out"
                        or "downsamplers" in name
                        or "upsamplers" in name
                    ):
                        wrapped_submodule = DistriConv2dTP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)

        super(DistriSDXLUNetTP, self).__init__(model, distri_config)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor or float or int,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor or None = None,
        timestep_cond: torch.Tensor or None = None,
        attention_mask: torch.Tensor or None = None,
        cross_attention_kwargs: dict[str, any] or None = None,
        added_cond_kwargs: dict[str, torch.Tensor] or None = None,
        down_block_additional_residuals: tuple[torch.Tensor] or None = None,
        mid_block_additional_residual: torch.Tensor or None = None,
        down_intrablock_additional_residuals: tuple[torch.Tensor] or None = None,
        encoder_attention_mask: torch.Tensor or None = None,
        return_dict: bool = True,
        record: bool = False,
    ):
        distri_config = self.distri_config
        b, c, h, w = sample.shape
        assert (
            class_labels is None
            and timestep_cond is None
            and attention_mask is None
            and cross_attention_kwargs is None
            and down_block_additional_residuals is None
            and mid_block_additional_residual is None
            and down_intrablock_additional_residuals is None
            and encoder_attention_mask is None
        )

        if distri_config.use_cuda_graph and not record:
            static_inputs = self.static_inputs

            if (
                distri_config.world_size > 1
                and distri_config.do_classifier_free_guidance
                and distri_config.split_batch
            ):
                assert b == 2
                batch_idx = distri_config.batch_idx()
                sample = sample[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1]
                    if torch.is_tensor(timestep) and timestep.ndim > 0
                    else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                for k in added_cond_kwargs:
                    added_cond_kwargs[k] = added_cond_kwargs[k][
                        batch_idx : batch_idx + 1
                    ]

            assert static_inputs["sample"].shape == sample.shape
            static_inputs["sample"].copy_(sample)
            if torch.is_tensor(timestep):
                if timestep.ndim == 0:
                    for b in range(static_inputs["timestep"].shape[0]):
                        static_inputs["timestep"][b] = timestep.item()
                else:
                    assert static_inputs["timestep"].shape == timestep.shape
                    static_inputs["timestep"].copy_(timestep)
            else:
                for b in range(static_inputs["timestep"].shape[0]):
                    static_inputs["timestep"][b] = timestep
            assert (
                static_inputs["encoder_hidden_states"].shape
                == encoder_hidden_states.shape
            )
            static_inputs["encoder_hidden_states"].copy_(encoder_hidden_states)
            for k in added_cond_kwargs:
                assert (
                    static_inputs["added_cond_kwargs"][k].shape
                    == added_cond_kwargs[k].shape
                )
                static_inputs["added_cond_kwargs"][k].copy_(added_cond_kwargs[k])

            graph_idx = 0

            self.cuda_graphs[graph_idx].replay()
            output = self.static_outputs[graph_idx]
        else:
            if distri_config.world_size == 1:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            elif (
                distri_config.do_classifier_free_guidance and distri_config.split_batch
            ):
                assert b == 2
                batch_idx = distri_config.batch_idx()
                sample = sample[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1]
                    if torch.is_tensor(timestep) and timestep.ndim > 0
                    else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                new_added_cond_kwargs = {}
                for k in added_cond_kwargs:
                    new_added_cond_kwargs[k] = added_cond_kwargs[k][
                        batch_idx : batch_idx + 1
                    ]
                added_cond_kwargs = new_added_cond_kwargs
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty(
                        (b, c, h, w), device=output.device, dtype=output.dtype
                    )
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty_like(output) for _ in range(2)]
                dist.all_gather(
                    self.buffer_list,
                    output.contiguous(),
                    group=distri_config.split_group(),
                    async_op=False,
                )
                torch.cat(self.buffer_list, dim=0, out=self.output_buffer)
                output = self.output_buffer
            else:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty_like(output)
                self.output_buffer.copy_(output)
                output = self.output_buffer
            if record:
                if self.static_inputs is None:
                    self.static_inputs = {
                        "sample": sample,
                        "timestep": timestep,
                        "encoder_hidden_states": encoder_hidden_states,
                        "added_cond_kwargs": added_cond_kwargs,
                    }
                self.synchronize()

        if return_dict:
            output = UNet2DConditionOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
