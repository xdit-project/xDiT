from diffusers import ConfigMixin, ModelMixin
from torch import nn

from legacy.pipefuser.modules.base_module import BaseModule
from ..utils import PatchParallelismCommManager, DistriConfig


class BaseModel(ModelMixin, ConfigMixin):
    def __init__(self, model: nn.Module, distri_config: DistriConfig):
        super(BaseModel, self).__init__()
        self.model = model
        self.distri_config = distri_config
        self.comm_manager = None

        self.buffer_list = None
        self.output_buffer = None
        self.counter = 0

        # for cuda graph
        self.static_inputs = None
        self.static_outputs = None
        self.cuda_graphs = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_counter(counter)

    def set_comm_manager(self, comm_manager: PatchParallelismCommManager):
        self.comm_manager = comm_manager
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_comm_manager(comm_manager)

    def setup_cuda_graph(self, static_outputs, cuda_graphs):
        self.static_outputs = static_outputs
        self.cuda_graphs = cuda_graphs

    @property
    def config(self):
        return self.model.config

    def synchronize(self):
        if self.comm_manager is not None and self.comm_manager.handles is not None:
            for i in range(len(self.comm_manager.handles)):
                if self.comm_manager.handles[i] is not None:
                    self.comm_manager.handles[i].wait()
                    self.comm_manager.handles[i] = None
