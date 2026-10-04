"""The HTTP server's command-line options reach the engine configuration."""

import torch

from entrypoints.server_args import parse_args, to_xfuser_args


def _parse(*argv):
    return parse_args(["--model_path", "/models/FLUX.1-schnell", *argv])


def test_ring_degree_reaches_the_parallel_config(monkeypatch):
    # Before --ring_degree was forwarded, this layout failed with
    # "parallel_world_size 2 must be equal to dit_parallel_size 4".
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    # Only the configuration is built, so the sequence-parallel kit need not be installed.
    monkeypatch.setattr("xfuser.config.config.HAS_LONG_CTX_ATTN", True)
    args = _parse("--world_size", "4", "--ulysses_parallel_degree", "2", "--ring_degree", "2")

    engine_config, _ = to_xfuser_args(args).create_config()

    parallel_config = engine_config.parallel_config
    assert (parallel_config.ulysses_degree, parallel_config.ring_degree) == (2, 2)
    assert parallel_config.sp_degree == 4
    assert parallel_config.dp_degree == 1
