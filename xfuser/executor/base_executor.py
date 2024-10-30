from abc import ABC, abstractmethod

from xfuser.config.config import EngineConfig


class BaseExecutor(ABC):
    def __init__(
        self,
        engine_config: EngineConfig,
    ):
        self.engine_config = engine_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self):
        pass
