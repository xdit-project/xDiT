"""
xDiT CLI entry point.

This module provides a console script that launches xDiT with distributed
training support, equivalent to:
    torchrun --nproc_per_node=N xfuser/runner.py <args>

Wraps torchrun as a subprocess with proper signal handling to kill all
processes on Ctrl+C.
"""

import math
import sys
import os
import signal
import subprocess
from typing import List, Optional, Tuple

# Torchrun-specific arguments that should be extracted and not passed to the runner
TORCHRUN_ARGS = {
    "--nnodes": "1",
    "--node_rank": "0",
    "--master_addr": "localhost",
    "--master_port": "29500",
    "--nproc_per_node": None,  # Will be computed
}


def get_nproc_from_args(args: List[str]) -> int:
    """
    Infer the number of processes from the command line arguments.

    TODO: Implement proper inference based on:
    - ulysses_degree
    - ring_degree
    - pipefusion_parallel_degree
    - data_parallel_degree
    - tensor_parallel_degree

    For now, returns a default of 8.
    """
    # Placeholder: In the future, parse args to determine optimal nproc
    # For example: nproc = ulysses_degree * ring_degree * pipefusion_parallel_degree * ...

    # Check if --nproc is explicitly passed
    degree_args = ["ulysses", "tensor_parallel", "ring", "pipefusion_parallel", "data_parallel"]
    degree_args = [f"--{arg}_degree" for arg in degree_args]
    degrees = []
    for i, arg in enumerate(args):
        if arg in degree_args and i + 1 < len(args):
            degrees.append(int(args[i + 1]))
        elif any(arg.startswith(f"{d}=") for d in degree_args):
            for d in degree_args:
                if arg.startswith(f"{d}="):
                    degrees.append(int(arg.split("=")[1]))
        if arg == "--use_cfg_parallel" or arg == "--use_cfg_parallel=True":
            degrees.append(2)  # Assume 2 for cfg parallel if enabled

    return math.prod(degrees) if degrees else 1


def extract_torchrun_args(args: List[str]) -> Tuple[dict, List[str]]:
    """
    Extract torchrun-specific arguments from the command line args.

    Returns:
        Tuple of (torchrun_args dict, remaining runner args)
    """
    torchrun_values = dict(TORCHRUN_ARGS)  # Copy defaults
    runner_args = []

    i = 0
    while i < len(args):
        arg = args[i]
        matched = False

        for torchrun_arg in TORCHRUN_ARGS:
            # Handle --arg=value format
            if arg.startswith(f"{torchrun_arg}="):
                torchrun_values[torchrun_arg] = arg.split("=", 1)[1]
                matched = True
                break
            # Handle --arg value format
            elif arg == torchrun_arg and i + 1 < len(args):
                torchrun_values[torchrun_arg] = args[i + 1]
                i += 1  # Skip the value
                matched = True
                break

        if not matched:
            runner_args.append(arg)
        i += 1

    return torchrun_values, runner_args


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the xdit CLI.

    Launches distributed training by wrapping torchrun as a subprocess.
    Handles Ctrl+C gracefully by killing all child processes.
    """
    if args is None:
        args = sys.argv[1:]

    # Extract torchrun args and get runner args
    torchrun_values, runner_args = extract_torchrun_args(args)

    # Infer nproc if not explicitly provided
    if torchrun_values["--nproc_per_node"] is None:
        torchrun_values["--nproc_per_node"] = str(get_nproc_from_args(runner_args))

    # Get the runner module
    runner_module = "xfuser.runner"

    # Build the torchrun command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={torchrun_values['--nproc_per_node']}",
        f"--nnodes={torchrun_values['--nnodes']}",
        f"--node_rank={torchrun_values['--node_rank']}",
        f"--master_addr={torchrun_values['--master_addr']}",
        f"--master_port={torchrun_values['--master_port']}",
        "-m", runner_module,
    ] + runner_args

    # Start the subprocess with a new process group so we can kill all children
    process = subprocess.Popen(
        cmd,
        start_new_session=True,  # Create new process group
    )

    def signal_handler(signum, frame):
        """Handle Ctrl+C by killing the entire process group."""
        print("\nReceived interrupt signal, terminating all processes...")
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already terminated
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Wait for the process to complete
        return_code = process.wait()
        sys.exit(return_code)
    except Exception as e:
        print(f"Error: {e}")
        signal_handler(None, None)


if __name__ == "__main__":
    main()
