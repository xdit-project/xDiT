"""
xDiT CLI entry point.

This module provides a console script that launches distributed inference,
equivalent to:
    torchrun --nproc_per_node=N -m xfuser.runner <args>

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


# Keep this help text in step with get_nproc and get_nproc_per_node.
_NPROC_PER_NODE_HELP = (
    "Processes to start on each node. Default: the product of the parallel degrees "
    "(doubled with --use_cfg_parallel), divided by --nnodes."
)


def get_nproc(parsed: argparse.Namespace) -> int:
    """
    The total number of processes that the runner's parallel degrees need.

    Read the degrees from the namespace of _launcher_parser(). Then the
    launcher reads each flag as the runner does, also in shortened form.
    """
    degrees = [getattr(parsed, flag[2:]) for flag in DEGREE_ARGS]
    cfg_degree = 2 if parsed.use_cfg_parallel else 1
    return math.prod(degrees) * cfg_degree


def get_nproc_per_node(parsed: argparse.Namespace) -> int:
    """
    The processes each node starts so that all nodes together run the
    product of the parallel degrees.
    """
    nnodes = parsed.nnodes
    try:
        num_nodes = int(nnodes)
    except ValueError:
        raise ValueError(
            f"Cannot infer --nproc_per_node from --nnodes={nnodes}; pass --nproc_per_node explicitly."
        ) from None
    if num_nodes < 1:
        raise ValueError(f"--nnodes must be at least 1, got {nnodes}.")
    total = get_nproc(parsed)
    if total % num_nodes != 0:
        raise ValueError(
            f"The parallel degrees need {total} processes, which cannot be split evenly across --nnodes={num_nodes}."
        )
    return total // num_nodes


_LAUNCHER_HELP = {
    "--nproc_per_node": _NPROC_PER_NODE_HELP,
    "--nnodes": "Number of nodes. Default: %(default)s.",
    "--node_rank": "Rank of this node. Default: %(default)s.",
    "--master_addr": "Address of the rank 0 node. Default: %(default)s.",
    "--master_port": "Port of the rank 0 node. Default: %(default)s.",
}


def _launcher_parser() -> argparse.ArgumentParser:
    """The runner's parser, with the torchrun options that xdit takes.

    The launcher uses it to show --help and to check the arguments before it
    starts torchrun. The parser loads without torch and the runtime.
    """
    from xfuser.config.args import FlexibleArgumentParser, xFuserArgs

    parser = FlexibleArgumentParser(
        prog="xdit",
        description="Run xDiT inference. xdit starts torchrun with the processes that the parallel "
        "degrees need, and gives the other options to xfuser.runner.",
    )
    launcher = parser.add_argument_group(
        "launcher options", "Options for torchrun. xdit does not give them to the runner."
    )
    for flag, help_text in _LAUNCHER_HELP.items():
        launcher.add_argument(flag, default=TORCHRUN_ARGS[flag], help=help_text)
    return xFuserArgs.add_runner_args(parser)


def extract_torchrun_args(args: List[str]) -> Tuple[dict, List[str]]:
    """
    Extract torchrun-specific arguments from the command line args.

    The scan finds only the full spellings of the launcher flags. main()
    stops on a shortened launcher flag, which argparse accepts.

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


def _shortened_launcher_flag(parser: argparse.ArgumentParser, runner_args: List[str]) -> Optional[Tuple[str, str]]:
    """
    Find a runner argument that argparse reads as a shortened launcher flag.

    argparse accepts a unique prefix of an option, for example --nnode for
    --nnodes. extract_torchrun_args does not find it, so torchrun would use
    the default value and each runner would fail on the flag.

    Returns:
        The argument and the launcher flag it shortens, or None.
    """
    for arg in runner_args:
        if arg == "--":
            break
        if not arg.startswith("--"):
            continue
        # Normalize the name as FlexibleArgumentParser does.
        key = "--" + arg.partition("=")[0][2:].replace("-", "_")
        if key in parser._option_string_actions:
            continue
        for flag in TORCHRUN_ARGS:
            if flag.startswith(key):
                return arg, flag
    return None


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the xdit CLI.

    Launches distributed inference by wrapping torchrun as a subprocess.
    """
    if args is None:
        args = sys.argv[1:]

    # Show --help and report argument errors here, before torchrun starts:
    # one message, and no torch import in torchrun and in each runner process.
    parser = _launcher_parser()
    parsed = parser.parse_args(args)

    # Extract torchrun args and get runner args
    torchrun_values, runner_args = extract_torchrun_args(args)
    shortened = _shortened_launcher_flag(parser, runner_args)
    if shortened is not None:
        arg, flag = shortened
        parser.error(f"write {flag} in full, not {arg}; xdit does not accept a shortened launcher option")

    # Infer nproc if not explicitly provided
    if torchrun_values["--nproc_per_node"] is None:
        try:
            nproc_per_node = get_nproc_per_node(parsed)
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
