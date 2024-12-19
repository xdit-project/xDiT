# Copyright 2024 The xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/executor/ray_utils.py
# Copyright (c) 2023, vLLM team. All rights reserved.
import time
import socket
from typing import Dict, List, Optional
from collections import defaultdict

from xfuser.config import ParallelConfig
from xfuser.logger import init_logger
from xfuser.envs import environment_variables

logger = init_logger(__name__)

PG_WAIT_TIMEOUT = 1800

try:
    import ray
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup

except ImportError as e:
    ray = None  # type: ignore
    ray_import_err = e


def ray_is_available() -> bool:
    """Returns True if Ray is available."""
    return ray is not None


def assert_ray_available():
    """Raise an exception if Ray is not available."""
    if ray is None:
        raise ValueError(
            "Failed to import Ray, please install Ray with " "`pip install ray`."
        ) from ray_import_err


def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is
    not created within time.

    """
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check "
            "`ray status` to see if you have enough resources.",
            int(time.time() - s),
            placement_group_specs,
        )

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError(
            "Cannot provide a placement group of "
            f"{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See "
            "`ray status` to make sure the cluster has enough resources."
        ) from None


def _wait_until_pg_removed(current_placement_group: "PlacementGroup"):
    ray.util.remove_placement_group(current_placement_group)
    s = time.time()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        pg = ray.util.get_current_placement_group()
        if pg is None:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            "Waiting for removing a placement group of specs for " "%d seconds.",
            int(time.time() - s),
        )
        time.sleep(wait_interval)


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: Optional[str] = None,
):
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    assert_ray_available()

    # Connect to a ray cluster.
    ray.init(address=ray_address, ignore_reinit_error=True)

    device_str = "GPU"
    # Create placement group for worker processes
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 " f"{device_str}."
                )
            if bundle_devices:
                device_bundles += 1
        if parallel_config.world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group."
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}."
            )
    else:
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        if parallel_config.world_size > num_devices_in_cluster:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group."
            )
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = [
            {device_str: 1.0} for _ in range(parallel_config.world_size)
        ]

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK"
        )
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group, parallel_config, device_str)
    # Set the placement group in the parallel config
    return current_placement_group


def get_num_nodes_in_placement_group() -> int:
    pg_table = ray.util.placement_group_table()
    current_pg = ray.util.get_current_placement_group()
    num_nodes = 0

    if current_pg:
        nodes_in_pg = set()
        for pg_key, pg in pg_table.items():
            if pg_key == current_pg.id.hex():
                for _, node in pg["bundles_to_node_id"].items():
                    nodes_in_pg.add(node)
        num_nodes = len(nodes_in_pg)

    return num_nodes


def get_ip() -> str:
    host_ip = environment_variables["MASTER_ADDR"]()
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    logger.warning(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


def get_open_port() -> int:
    port = environment_variables["MASTER_PORT"]()
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info(
                    "Port %d is already in use, trying port %d", port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def get_distributed_init_method(ip: str, port: int) -> str:
    # Brackets are not permitted in ipv4 addresses,
    # see https://github.com/python/cpython/issues/103848
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"


def _verify_bundles(
    placement_group: "PlacementGroup", parallel_config: ParallelConfig, device_str: str
):
    """Verify a given placement group has bundles located in the right place.

    There are 2 rules.
    - Warn if all tensor parallel workers cannot fit in a single node.
    - Fail if driver node is not included in a placement group.
    """
    assert (
        ray.is_initialized()
    ), "Ray is not initialized although distributed-executor-backend is ray."
    pg_data = placement_group_table(placement_group)
    # bundle_idx -> node_id
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # bundle_idx -> bundle (e.g., {"GPU": 1})
    bundles = pg_data["bundles"]
    # node_id -> List of bundle (e.g., {"GPU": 1})
    node_id_to_bundle: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for bundle_idx, node_id in bundle_to_node_ids.items():
        node_id_to_bundle[node_id].append(bundles[bundle_idx])
    driver_node_id = ray.get_runtime_context().get_node_id()

    if driver_node_id not in node_id_to_bundle:
        raise RuntimeError(
            f"driver node id {driver_node_id} is not included in a placement "
            f"group {placement_group.id}. Node id -> bundles "
            f"{node_id_to_bundle}. "
            "You don't have enough GPUs available in a current node. Check "
            "`ray status` to see if you have available GPUs in a node "
            f"{driver_node_id} before starting an vLLM engine."
        )

    for node_id, bundles in node_id_to_bundle.items():
        if len(bundles) < parallel_config.sp_degree:
            logger.warning(
                "sequence parallel degree=%d "
                "is bigger than a reserved number of %ss (%d "
                "%ss) in a node %s. Sequence parallel workers can be "
                "spread out to 2+ nodes which can degrade the performance "
                "unless you have fast interconnect across nodes, like "
                "Infiniband. To resolve this issue, make sure you have more "
                "than %d GPUs available at each node.",
                parallel_config.sp_degree,
                device_str,
                len(bundles),
                device_str,
                node_id,
                parallel_config.tensor_parallel_size,
            )
