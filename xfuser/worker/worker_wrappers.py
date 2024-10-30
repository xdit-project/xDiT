import os
import ray
import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from utils import update_environment_variables


class BaseWorkerWrapper(ABC):
    def __init__(self, worker_module_name: str, worker_class_name: str):
        self.worker_module_name = worker_module_name
        self.worker_class_name = worker_class_name
        self.worker = None

    # lazy import
    def init_worker(self, *args, **kwargs):
        try:
            module = importlib.import_module(self.worker_module_name)
            worker_class = getattr(module, self.worker_class_name)
        except ImportError as e:
            raise (
                ImportError(
                    f"Module {self.worker_module_name} not found, error info: {e}"
                )
            )
        except AttributeError as e:
            raise (
                AttributeError(
                    f"Attribute {self.worker_class_name} in module: {self.worker_module_name} not found, error info: {e}"
                )
            )

        self.worker = worker_class(*args, **kwargs)

    def execute_method(self, method: str, *args, **kwargs) -> Any:
        method = getattr(self, method, None) or getattr(self.worker, method, None)
        if not method:
            raise (AttributeError(f"Method {method} not found in Worker class"))
        return method(*args, **kwargs)

    def update_environs(environs: Dict[str, str]):
        if "CUDA_VISIBLE_DEVICES" in environs and "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        update_environment_variables(environs)


class RayWorkerWrapper(BaseWorkerWrapper):
    def get_node_and_gpu_ids(
        self,
    ) -> Tuple[str, List[int]]:
        gpu_ids = ray.get_gpu_ids()
        node_id = ray.get_runtime_context().get_node_id()
        return node_id, gpu_ids
