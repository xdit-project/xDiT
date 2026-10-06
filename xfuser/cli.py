"""
xDiT CLI entry point.

This module provides a console script that launches xDiT with distributed
training support, equivalent to:
    torchrun --nproc_per_node=N xfuser/runner.py <args>

"""

import argparse
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


# Runner flags whose product is the number of processes the run needs.
DEGREE_ARGS = (
    "--ulysses_degree",
    "--tensor_parallel_degree",
    "--ring_degree",
    "--pipefusion_parallel_degree",
    "--data_parallel_degree",
)


def _spellings(flag: str) -> List[str]:
    """Like the runner's parser, accept each flag with underscores or dashes."""
    return [flag, "--" + flag[2:].replace("_", "-")]


def _degree_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xdit", add_help=False, allow_abbrev=False)
    for flag in DEGREE_ARGS:
        parser.add_argument(*_spellings(flag), dest=flag[2:], type=int, default=1)
    parser.add_argument(*_spellings("--use_cfg_parallel"), dest="use_cfg_parallel", action="store_true")
    return parser


def get_nproc_from_args(args: List[str]) -> int:
    """
    Infer the total number of processes from the runner's parallel degrees.
    """
    parsed, _ = _degree_parser().parse_known_args(args)
    degrees = [getattr(parsed, flag[2:]) for flag in DEGREE_ARGS]
    cfg_degree = 2 if parsed.use_cfg_parallel else 1
    return math.prod(degrees) * cfg_degree


def get_nproc_per_node(args: List[str], nnodes: str) -> int:
    """
    The processes each node starts so that all nodes together run the
    product of the parallel degrees.
    """
    try:
        num_nodes = int(nnodes)
    except ValueError:
        raise ValueError(
            f"Cannot infer --nproc_per_node from --nnodes={nnodes}; pass --nproc_per_node explicitly."
        ) from None
    if num_nodes < 1:
        raise ValueError(f"--nnodes must be at least 1, got {nnodes}.")
    total = get_nproc_from_args(args)
    if total % num_nodes != 0:
        raise ValueError(
            f"The parallel degrees need {total} processes, which cannot be split evenly across --nnodes={num_nodes}."
        )
    return total // num_nodes


def extract_torchrun_args(args: List[str]) -> Tuple[dict, List[str]]:
    """
    Extract torchrun-specific arguments from the command line args.

    Returns:
        Tuple of (torchrun_args dict, remaining runner args)
    """
    torchrun_values = dict(TORCHRUN_ARGS)  # Copy defaults
    flags = {spelling: flag for flag in TORCHRUN_ARGS for spelling in _spellings(flag)}
    runner_args = []

    i = 0
    while i < len(args):
        arg = args[i]
        key, has_value, value = arg.partition("=")
        if has_value and key in flags:  # --arg=value
            torchrun_values[flags[key]] = value
        elif arg in flags and i + 1 < len(args):  # --arg value
            torchrun_values[flags[arg]] = args[i + 1]
            i += 1  # Skip the value
        else:
            runner_args.append(arg)
        i += 1

    return torchrun_values, runner_args


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the xdit CLI.

    Launches distributed training by wrapping torchrun as a subprocess.
    """
    if args is None:
        args = sys.argv[1:]

    # Extract torchrun args and get runner args
    torchrun_values, runner_args = extract_torchrun_args(args)

    # Infer nproc if not explicitly provided
    if torchrun_values["--nproc_per_node"] is None:
        try:
            nproc_per_node = get_nproc_per_node(runner_args, torchrun_values["--nnodes"])
        except ValueError as error:
            sys.exit(f"xdit: {error}")
        torchrun_values["--nproc_per_node"] = str(nproc_per_node)

    runner_module = "xfuser.runner"

    # Build the torchrun command
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={torchrun_values['--nproc_per_node']}",
        f"--nnodes={torchrun_values['--nnodes']}",
        f"--node_rank={torchrun_values['--node_rank']}",
        f"--master_addr={torchrun_values['--master_addr']}",
        f"--master_port={torchrun_values['--master_port']}",
        "-m",
        runner_module,
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

    # Wait for the process to complete
    return_code = process.wait()
    sys.exit(return_code)


if __name__ == "__main__":
    main()
