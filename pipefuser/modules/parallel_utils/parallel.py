import torch
import torch.nn as nn
import torch.distributed as dist

from pipefuser.modules.pipefusion.pipefusion_pixart_alpha import (
    PipeFusionPixartAlpha,
)
from pipefuser.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

class Parallel():
    def __init__(self, distri_config):
        init_distributed_environment()



    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

        

