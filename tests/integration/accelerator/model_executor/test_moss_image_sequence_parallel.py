"""MOSS-Image under Ulysses (and CFG parallel) matches one GPU.

Covers both attention paths a rank takes: one sample through ``USP_bshd``
(token-major send buffers, written by the FlyDSL q/k norm + RoPE kernel where
it is available) and the serial-CFG pair through ``USP_pair``, eager and under
``torch.compile(mode="reduce-overhead")``. The joint sequence (256 image + 5
text tokens) needs an SP pad, which ``valid_kv_len`` must keep out of the keys.
Tiny random bf16 weights; nothing is downloaded.
"""

import pytest
import torch

from _spawn import spawn_accelerator_ranks

pytestmark = [pytest.mark.multi_gpu, pytest.mark.slow]

_TINY = dict(dim=1024, ffn_dim=512, num_heads=8, num_layers=2, text_dim=32, freq_dim=32, latent_channels=8)
GRID = (16, 16)
TEXT_LEN = 5
# Largest relative L2 error against one GPU; bf16 GEMMs and the reordered attention reduction stay below ~1e-2.
TOLERANCE = 2e-2


def _inputs(device):
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(1, _TINY["latent_channels"], *GRID, generator=generator)
    text = torch.randn(1, TEXT_LEN, _TINY["text_dim"], generator=generator)
    pair_text = torch.randn(1, TEXT_LEN - 2, _TINY["text_dim"], generator=generator)
    return (
        hidden.to(device, torch.bfloat16),
        text.to(device, torch.bfloat16),
        pair_text.to(device, torch.bfloat16),
    )


def _worker(rank, world_size, init_method, ulysses, cfg, backend, compiled, output_path):
    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.models.transformers.transformer_moss_image import MOSSImageTransformer

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=ulysses, classifier_free_guidance_degree=cfg)
    try:
        args = xFuserArgs(attention_backend=backend, ulysses_degree=ulysses, use_cfg_parallel=cfg > 1)
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        get_runtime_state().set_attention_backend(backend)

        device = torch.device("cuda", rank)
        torch.manual_seed(0)
        model = MOSSImageTransformer(**_TINY).eval().to(device=device, dtype=torch.bfloat16)
        model.fuse_qkv_projections(("q", "k"))
        hidden, text, pair_text = _inputs(device)
        timestep = torch.tensor(500.0, device=device)
        forward = torch.compile(model, mode="reduce-overhead") if compiled else model
        outputs = {}
        with torch.no_grad():
            for _ in range(3 if compiled else 1):
                torch.compiler.cudagraph_mark_step_begin()
                outputs["single"] = forward(hidden, timestep, text, return_dict=False)[0].clone()
                torch.compiler.cudagraph_mark_step_begin()
                outputs["pair"] = forward(
                    hidden, timestep, text, pair_encoder_hidden_states=pair_text, pair_skip_layers=(1,), return_dict=False
                )[0].clone()
        if rank == 0:
            torch.save({name: tensor.float().cpu() for name, tensor in outputs.items()}, output_path)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run(directory, ulysses, cfg, backend, compiled):
    output_path = directory / "output.pt"
    spawn_accelerator_ranks(
        _worker,
        directory,
        world_size=ulysses * cfg,
        # reduce-overhead compiles the single and pair graphs on every rank, which alone takes ~15 min.
        timeout=1800,
        args=(ulysses, cfg, backend, compiled, str(output_path)),
    )
    return torch.load(output_path)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "reduce-overhead"])
@pytest.mark.parametrize("ulysses, cfg", [(4, 1), (4, 2)], ids=["u4", "u4-cfg2"])
@pytest.mark.parametrize("backend", ["AITER", "SDPA_EFFICIENT"] if torch.version.hip else ["SDPA_EFFICIENT"])
def test_sequence_parallel_matches_one_gpu(tmp_path, backend, ulysses, cfg, compiled):
    (tmp_path / "ref").mkdir()
    (tmp_path / "sp").mkdir()
    reference = _run(tmp_path / "ref", 1, 1, backend, False)
    parallel = _run(tmp_path / "sp", ulysses, cfg, backend, compiled)
    for name, expected in reference.items():
        error = (parallel[name] - expected).norm() / expected.norm()
        assert error < TOLERANCE, f"{name}: relative L2 error {error:.2e}"
