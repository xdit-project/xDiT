# Copyright 2024 The xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
# Copyright (c) 2023, vLLM team. All rights reserved.
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from itertools import islice, repeat
from typing import Any, Dict, List, Optional, Tuple

from xfuser.ray.worker.worker import VAEWorker, DiTWorker
from xfuser.ray.pipeline.base_executor import BaseExecutor
from xfuser.ray.pipeline.ray_utils import initialize_ray_cluster
from xfuser.logger import init_logger
from xfuser.ray.worker.worker_wrappers import RayWorkerWrapper
from xfuser.config.config import InputConfig, EngineConfig
logger = init_logger(__name__)


class GPUExecutor(BaseExecutor):
    def _init_executor(self):
        pass


class RayDiffusionPipeline(GPUExecutor):
    total_workers = []
    dit_workers = []
    vae_workers = []
    def _init_executor(self):
        self._init_ray_workers()
        self._run_workers(self.workers,"init_worker_distributed_environment")

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

            if bundle_id < self.engine_config.parallel_config.dit_parallel_size:
                # DiT workers
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=scheduling_strategy,
                )(RayWorkerWrapper).remote(
                    self.engine_config.parallel_config,
                    "xfuser.ray.worker.worker.DiTWorker",
                    bundle_id,
                )
                self.dit_workers.append(worker)
            else:
                # VAE workers
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=scheduling_strategy,
                )(RayWorkerWrapper).remote(
                    self.engine_config.parallel_config,
                    "xfuser.ray.worker.worker.VAEWorker",
                    bundle_id,
                )
                self.vae_workers.append(worker)
                
            self.workers.append(worker)

    def _run_workers(
        self,
        workers: List[ray.ObjectRef],
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

        count = len(workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)

        # Start the ray workers first.
        ray_workers = workers
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
    
    @classmethod
    def from_pretrained(cls,PipelineClass,pretrained_model_name_or_path: str,engine_config: EngineConfig,**kwargs):
        pipeline = cls(engine_config)
        pipeline._run_workers(pipeline.workers,"from_pretrained",PipelineClass,pretrained_model_name_or_path,engine_config,**kwargs)
        return pipeline

    def prepare_run(self, input_config: InputConfig, steps: int = 3, sync_steps: int = 1):
        self._run_workers(self.workers,"prepare_run",input_config,steps,sync_steps)

    def __call__(self,**kwargs):
        return  self._run_workers(self.workers,"execute",**kwargs)
