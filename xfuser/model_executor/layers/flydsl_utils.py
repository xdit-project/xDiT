from typing import Optional

import torch


def get_device_wave_size(tensor: torch.Tensor) -> Optional[int]:
    properties = torch.cuda.get_device_properties(tensor.device)
    wave_size = getattr(properties, "warp_size", None)
    if not isinstance(wave_size, int) or wave_size < 2 or wave_size & (wave_size - 1):
        return None
    return wave_size
