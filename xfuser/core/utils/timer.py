import time

import torch
from torch.cuda import synchronize

try:
    import torch_musa
    from torch_musa.core.device import synchronize
except ModuleNotFoundError:
    pass

def gpu_timer_decorator(func):
    def wrapper(*args, **kwargs):
        synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        synchronize()
        end_time = time.time()

        if torch.distributed.get_rank() == 0:
            print(
                f"{func.__name__} took {end_time - start_time} seconds to run on GPU."
            )
        return result

    return wrapper
