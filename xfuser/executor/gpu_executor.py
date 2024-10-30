import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from base_executor import BaseExecutor
from xfuser.executor.ray_utils import initialize_ray_cluster
from xfuser.logger import init_logger
from xfuser.worker.worker_wrappers import RayWorkerWrapper

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
            )(RayWorkerWrapper).remote()
            self.workers.append(worker)

        self.node_metadata = {}
