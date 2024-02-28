from torch import nn

from distrifuser.utils import DistriConfig


class BaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        distri_config: DistriConfig,
    ):
        super(BaseModule, self).__init__()
        self.module = module
        self.distri_config = distri_config
        self.comm_manager = None

        self.counter = 0

        self.buffer_list = None
        self.idx = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter

    def set_comm_manager(self, comm_manager):
        self.comm_manager = comm_manager
