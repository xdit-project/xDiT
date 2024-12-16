import os
import ray
import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from xfuser.worker.utils import update_environment_variables, resolve_obj_by_qualname


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
    def __init__(self, *args, **kwargs) -> None:
        print(f"RayWorkerWrapper init with args: {args} and kwargs: {kwargs}")
        super().__init__(*args, **kwargs)
        self.init_worker()

    def get_node_and_gpu_ids(
        self,
    ) -> Tuple[str, List[int]]:
        gpu_ids = ray.get_gpu_ids()
        node_id = ray.get_runtime_context().get_node_id()
        return node_id, gpu_ids
