"""A GPU worker class."""
import gc
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from abc import ABC, abstractmethod

from xfuser.envs import environment_variables
from xfuser.config.config import EngineConfig, InputConfig


class WorkerBase(ABC):
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        input_config: InputConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
    ):
        raise NotImplementedError


class Worker(WorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
    ) -> None:
        WorkerBase.__init__(self)

    def load_model(self):
        pass

    def execute(self, input_config: InputConfig):
        print("===============")
        print(f"executing model with input_config: {input_config}")
        print("===============")
