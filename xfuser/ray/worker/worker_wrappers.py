# Copyright 2024 The xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker_base.py
# Copyright (c) 2023, vLLM team. All rights reserved.
import os
from abc import ABC
from typing import Any, Dict

from xfuser.ray.worker.utils import update_environment_variables, resolve_obj_by_qualname
from xfuser.config.config import ParallelConfig

class BaseWorkerWrapper(ABC):
    def __init__(self, worker_cls: str):
        self.worker_cls = worker_cls
        self.worker = None

    # lazy import
    def init_worker(self, *args, **kwargs):
        worker_class = resolve_obj_by_qualname(
            self.worker_cls)
        self.worker = worker_class(*args, **kwargs)
        assert self.worker is not None

    def execute_method(self, method: str, *args, **kwargs) -> Any:
        method = getattr(self, method, None) or getattr(
            self.worker, method, None)
        if not method:
            raise (AttributeError(
                f"Method {method} not found in Worker class"))
        return method(*args, **kwargs)

    def update_environs(environs: Dict[str, str]):
        if "CUDA_VISIBLE_DEVICES" in environs and "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        update_environment_variables(environs)


class RayWorkerWrapper(BaseWorkerWrapper):
    def __init__(self, parallel_config: ParallelConfig, worker_cls: str, rank: int) -> None:
        super().__init__(worker_cls)
        self.init_worker(parallel_config, rank)