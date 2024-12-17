import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from itertools import islice, repeat
from typing import Any, Dict, List, Optional, Tuple

from xfuser.executor.base_executor import BaseExecutor
from xfuser.executor.ray_utils import initialize_ray_cluster
from xfuser.logger import init_logger
from xfuser.worker.worker_wrappers import RayWorkerWrapper
from xfuser.config.config import InputConfig, EngineConfig

logger = init_logger(__name__)


class GPUExecutor(BaseExecutor):
    def _init_executor(self):
        pass


class RayGPUExecutor(GPUExecutor):
    workers = []
    def _init_executor(self):
        self._init_ray_workers()

    def _init_ray_workers(self):
        placement_group = initialize_ray_cluster(self.engine_config.parallel_config)

        # create placement group and worker wrapper instance for lazy load worker
        self.workers = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            # Skip bundles without GPUs
            if not bundle.get("GPU", 0):
                continue

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_id,
                placement_group_capture_child_tasks=True,
            )
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=scheduling_strategy,
            )(RayWorkerWrapper).remote(self.engine_config,bundle_id)
            self.workers.append(worker)

        self.node_metadata = {}

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """

        count = len(self.workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)

        # Start the ray workers first.
        ray_workers = self.workers
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return ray_worker_outputs

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs
    
    def init_distributed_environment(self):
        self._run_workers("init_worker_distributed_environment")

    def load_model(self,engine_config: EngineConfig):
        self._run_workers("load_model",engine_config)

    def execute(self, input_config: InputConfig):
        self._run_workers("execute", input_config)
