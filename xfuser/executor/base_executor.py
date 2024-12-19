# Copyright 2024 The xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/executor/executor_base.py
# Copyright (c) 2023, vLLM team. All rights reserved.
from abc import ABC, abstractmethod

from xfuser.config.config import EngineConfig


class BaseExecutor(ABC):
    def __init__(
        self,
        engine_config: EngineConfig,
    ):
        self.engine_config = engine_config
        self.parallel_config = engine_config.parallel_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self):
        pass
