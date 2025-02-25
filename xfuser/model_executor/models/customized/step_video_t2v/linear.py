import torch
import torch.nn as nn
from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_tensor_model_parallel_world_size
)


class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gather_output=True, tp_group=None):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = tp_group or get_tp_group()

        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        x = super().forward(x)
        return x


class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True, tp_group=None):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = tp_group or get_tp_group()
        self.input_is_parallel = input_is_parallel

        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        if not self.input_is_parallel:
            x = torch.chunk(x, self.tp_size, dim=-1)[self.tp_rank]
        x = super().forward(x)
        # 执行All-Reduce聚合结果
        x = self.tp_group.all_reduce(x)
        return x
