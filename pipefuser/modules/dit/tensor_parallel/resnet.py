import torch.cuda
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.utils import USE_PEFT_BACKEND
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from pipefuser.modules.base_module import BaseModule
from pipefuser.utils import DistriConfig


class DistriResnetBlock2DTP(BaseModule):
    def __init__(self, module: ResnetBlock2D, distri_config: DistriConfig):
        super(DistriResnetBlock2DTP, self).__init__(module, distri_config)
        assert module.conv1.out_channels % distri_config.n_device_per_batch == 0

        mid_channels = module.conv1.out_channels // distri_config.n_device_per_batch

        sharded_conv1 = nn.Conv2d(
            module.conv1.in_channels,
            mid_channels,
            module.conv1.kernel_size,
            module.conv1.stride,
            module.conv1.padding,
            module.conv1.dilation,
            module.conv1.groups,
            module.conv1.bias is not None,
            module.conv1.padding_mode,
            device=module.conv1.weight.device,
            dtype=module.conv1.weight.dtype,
        )
        sharded_conv1.weight.data.copy_(
            module.conv1.weight.data[
                distri_config.split_idx()
                * mid_channels : (distri_config.split_idx() + 1)
                * mid_channels
            ]
        )
        if module.conv1.bias is not None:
            sharded_conv1.bias.data.copy_(
                module.conv1.bias.data[
                    distri_config.split_idx()
                    * mid_channels : (distri_config.split_idx() + 1)
                    * mid_channels
                ]
            )

        sharded_conv2 = nn.Conv2d(
            mid_channels,
            module.conv2.out_channels,
            module.conv2.kernel_size,
            module.conv2.stride,
            module.conv2.padding,
            module.conv2.dilation,
            module.conv2.groups,
            module.conv2.bias is not None,
            module.conv2.padding_mode,
            device=module.conv2.weight.device,
            dtype=module.conv2.weight.dtype,
        )
        sharded_conv2.weight.data.copy_(
            module.conv2.weight.data[
                :,
                distri_config.split_idx()
                * mid_channels : (distri_config.split_idx() + 1)
                * mid_channels,
            ]
        )
        if module.conv2.bias is not None:
            sharded_conv2.bias.data.copy_(module.conv2.bias.data)

        assert module.time_emb_proj is not None
        assert module.time_embedding_norm == "default"

        sharded_time_emb_proj = nn.Linear(
            module.time_emb_proj.in_features,
            mid_channels,
            bias=module.time_emb_proj.bias is not None,
            device=module.time_emb_proj.weight.device,
            dtype=module.time_emb_proj.weight.dtype,
        )
        sharded_time_emb_proj.weight.data.copy_(
            module.time_emb_proj.weight.data[
                distri_config.split_idx()
                * mid_channels : (distri_config.split_idx() + 1)
                * mid_channels
            ]
        )
        if module.time_emb_proj.bias is not None:
            sharded_time_emb_proj.bias.data.copy_(
                module.time_emb_proj.bias.data[
                    distri_config.split_idx()
                    * mid_channels : (distri_config.split_idx() + 1)
                    * mid_channels
                ]
            )

        sharded_norm2 = nn.GroupNorm(
            module.norm2.num_groups // distri_config.n_device_per_batch,
            mid_channels,
            module.norm2.eps,
            module.norm2.affine,
            device=module.norm2.weight.device,
            dtype=module.norm2.weight.dtype,
        )
        if module.norm2.affine:
            sharded_norm2.weight.data.copy_(
                module.norm2.weight.data[
                    distri_config.split_idx()
                    * mid_channels : (distri_config.split_idx() + 1)
                    * mid_channels
                ]
            )
            sharded_norm2.bias.data.copy_(
                module.norm2.bias.data[
                    distri_config.split_idx()
                    * mid_channels : (distri_config.split_idx() + 1)
                    * mid_channels
                ]
            )

        del module.conv1
        del module.conv2
        del module.time_emb_proj
        del module.norm2
        module.conv1 = sharded_conv1
        module.conv2 = sharded_conv2
        module.time_emb_proj = sharded_time_emb_proj
        module.norm2 = sharded_norm2

        torch.cuda.empty_cache()

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert scale == 1.0

        distri_config = self.distri_config
        module = self.module

        hidden_states = input_tensor
        hidden_states = module.norm1(hidden_states)

        hidden_states = module.nonlinearity(hidden_states)

        if module.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                module.upsample(input_tensor, scale=scale)
                if isinstance(module.upsample, Upsample2D)
                else module.upsample(input_tensor)
            )
            hidden_states = (
                module.upsample(hidden_states, scale=scale)
                if isinstance(module.upsample, Upsample2D)
                else module.upsample(hidden_states)
            )
        elif module.downsample is not None:
            input_tensor = (
                module.downsample(input_tensor, scale=scale)
                if isinstance(module.downsample, Downsample2D)
                else module.downsample(input_tensor)
            )
            hidden_states = (
                module.downsample(hidden_states, scale=scale)
                if isinstance(module.downsample, Downsample2D)
                else module.downsample(hidden_states)
            )

        hidden_states = module.conv1(hidden_states)

        if module.time_emb_proj is not None:
            if not module.skip_time_act:
                temb = module.nonlinearity(temb)
            temb = module.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and module.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = module.norm2(hidden_states)

        if temb is not None and module.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = module.nonlinearity(hidden_states)

        hidden_states = module.dropout(hidden_states)
        hidden_states = F.conv2d(
            hidden_states,
            module.conv2.weight,
            bias=None,
            stride=module.conv2.stride,
            padding=module.conv2.padding,
            dilation=module.conv2.dilation,
            groups=module.conv2.groups,
        )

        dist.all_reduce(
            hidden_states,
            op=dist.ReduceOp.SUM,
            group=distri_config.batch_parallel_group,
            async_op=False,
        )
        if module.conv2.bias is not None:
            hidden_states = hidden_states + module.conv2.bias.view(1, -1, 1, 1)

        if module.conv_shortcut is not None:
            input_tensor = (
                module.conv_shortcut(input_tensor, scale)
                if not USE_PEFT_BACKEND
                else self.conv_shortcut(input_tensor)
            )

        output_tensor = (input_tensor + hidden_states) / module.output_scale_factor

        self.counter += 1

        return output_tensor
