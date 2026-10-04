"""Command-line options of the HTTP server in ``entrypoints/launch.py``.

This module does not import ray or fastapi, so the option handling can be
used and tested without the server's optional dependencies.
"""

import argparse
from typing import Optional, Sequence

import torch

from xfuser.config import xFuserArgs

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="xDiT HTTP Service")
    parser.add_argument("--model_path", type=str, help="Path to the model", required=True)
    parser.add_argument("--world_size", type=int, default=1, help="Number of parallel workers")
    parser.add_argument(
        "--pipefusion_parallel_degree", type=int, default=1, help="Degree of pipeline fusion parallelism"
    )
    parser.add_argument("--ulysses_parallel_degree", type=int, default=1, help="Degree of Ulysses parallelism")
    parser.add_argument("--ring_degree", type=int, default=1, help="Degree of ring parallelism")
    parser.add_argument(
        "--save_disk_path",
        type=str,
        default=None,
        help="Directory to save generated images to when a request does not set save_disk_path. "
        "If neither is set, the image is returned base64-encoded in the response.",
    )
    parser.add_argument("--use_cfg_parallel", action="store_true", help="Whether to use CFG parallel")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=sorted(DTYPES),
        default="fp16",
        help="Precision to load and run the model in (default: fp16)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Port the workers use to set up torch.distributed. "
        "The address is taken from MASTER_ADDR, default 127.0.0.1.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Address the HTTP server binds to")
    parser.add_argument("--port", type=int, default=6000, help="Port the HTTP server listens on")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def to_xfuser_args(args: argparse.Namespace) -> xFuserArgs:
    return xFuserArgs(
        model=args.model_path,
        trust_remote_code=True,
        warmup_steps=1,
        use_parallel_vae=False,
        use_torch_compile=False,
        ulysses_degree=args.ulysses_parallel_degree,
        ring_degree=args.ring_degree,
        pipefusion_parallel_degree=args.pipefusion_parallel_degree,
        use_cfg_parallel=args.use_cfg_parallel,
        dit_parallel_size=0,
    )
