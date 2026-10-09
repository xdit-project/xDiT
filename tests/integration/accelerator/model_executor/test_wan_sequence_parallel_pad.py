"""Wan and Wan VACE under Ulysses and ring attention match one GPU, also when the sequence needs an SP pad.

The models pad the sequence so it splits evenly across ranks, and their RoPE
tables too, so the pad keys get a zero logit and would draw attention from
every real token. The models publish the real length (`valid_kv_len`) for USP
to drop those keys; this compares the outputs with a one-GPU run, which has no
pad. Tiny random weights; nothing is downloaded.

Random weights draw no attention sink, so an attended pad errs by only ~1e-3
to 1e-2 here. In an all-bf16 model the GEMMs and norms round by as much, which
would hide it. The models therefore run in fp32: on SDPA_EFFICIENT everywhere,
and on gfx950 also with attention in bf16 on AITER_BF16, while nothing else
rounds. The latter is the only check of the values of the log-sum-exp AITER
returns for ring attention, and it reaches it with the keys a ring step drops.
"""

import pytest
import torch

from _spawn import spawn_accelerator_ranks

pytestmark = [pytest.mark.multi_gpu, pytest.mark.slow]

_TINY_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=128,
    in_channels=4,
    out_channels=4,
    text_dim=32,
    freq_dim=32,
    ffn_dim=256,
    num_layers=2,
    rope_max_seq_len=64,
)
_VACE_CONFIG = dict(vace_layers=[0, 1], vace_in_channels=8)
TEXT_LEN = 16
# Latent (frames, height, width). 9 tokens after the 1x2x2 patching: a pad of 1 on two ranks, 3 on four. Short,
# because an attended pad's error grows with its share of the keys.
PADDED = (3, 2, 6)
# 256 tokens, which split evenly on two and four ranks.
UNPADDED = (4, 16, 16)
MODES = {"ulysses": (2, 1), "ring": (1, 2), "ulysses-ring": (2, 2)}
# Backend, model dtype, and whether only attention runs in bf16.
NUMERICS = {
    "fp32-sdpa": ("SDPA_EFFICIENT", torch.float32, False),
    "fp32-model-bf16-attention": ("AITER_BF16", torch.float32, True),
}
# Largest relative L2 error against one GPU. Measured with the pad dropped / attended: in fp32 <= 3.4e-7 /
# >= 3.3e-3, with bf16 attention <= 1.6e-4 / >= 3.3e-3. Without a pad, a run that drops one real key errs
# by >= 2.2e-4 in fp32.
TOLERANCE = {"fp32-sdpa": 1e-5, "fp32-model-bf16-attention": 5e-4}


def _model(name):
    if name == "wan":
        from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper

        return xFuserWanTransformer3DWrapper(**_TINY_CONFIG)
    from xfuser.model_executor.models.transformers.transformer_wan_vace import xFuserWanVACETransformer3DWrapper

    return xFuserWanVACETransformer3DWrapper(**_TINY_CONFIG, **_VACE_CONFIG)


def _inputs(name, latent_shape):
    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(1, _TINY_CONFIG["in_channels"], *latent_shape, generator=generator),
        encoder_hidden_states=torch.randn(1, TEXT_LEN, _TINY_CONFIG["text_dim"], generator=generator),
    )
    if name == "vace":
        inputs["control_hidden_states"] = torch.randn(
            1, _VACE_CONFIG["vace_in_channels"], *latent_shape, generator=generator
        )
    return inputs


def _attend_in_bf16(backend):
    """Have the backend's kernel take bf16 copies of an fp32 model's Q/K/V, as it would in an all-bf16 model.

    AITER_BF16 serves bf16 only, while the padded runs need the rest of the
    model in fp32 (see the module docstring). Swapping the resolved kernel
    keeps the registry dispatch, the ring and USP's handling of the pad as they
    run in production. Done in the spawned rank, so it never outlives the run.
    """
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType

    spec = registry.get(AttentionBackendType[backend])
    kernel = spec.resolved()

    def kernel_in_bf16(query, key, value, call):
        output, lse = kernel(query.bfloat16(), key.bfloat16(), value.bfloat16(), call)
        return output.to(query.dtype), lse

    # Spec is frozen; run() dispatches to whatever it holds as resolved.
    object.__setattr__(spec, "_resolved", kernel_in_bf16)


def _worker(rank, world_size, init_method, name, ulysses, ring, latent_shape, numerics, output_path):
    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    backend, dtype, attention_in_bf16 = NUMERICS[numerics]
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method
    )
    initialize_model_parallel(ulysses_degree=ulysses, ring_degree=ring)
    try:
        args = xFuserArgs(attention_backend=backend, ulysses_degree=ulysses, ring_degree=ring)
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        runtime_state = get_runtime_state()
        runtime_state.set_attention_backend(backend)
        # Without a pipeline there is no denoising schedule to advance, but the wrappers' forward asks to.
        runtime_state.increment_step_counter = lambda: None
        if attention_in_bf16:
            _attend_in_bf16(backend)

        device = torch.device("cuda", rank)
        # Seeded on the CPU, so every process builds the same weights.
        torch.manual_seed(0)
        model = _model(name).eval().to(device=device, dtype=dtype)
        inputs = {key: tensor.to(device=device, dtype=dtype) for key, tensor in _inputs(name, latent_shape).items()}
        with torch.no_grad():
            output = model(**inputs, timestep=torch.tensor([500.0], device=device), return_dict=False)[0]
        if rank == 0:
            torch.save(output.cpu(), output_path)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run(directory, name, ulysses, ring, latent_shape, numerics):
    output_path = directory / "output.pt"
    spawn_accelerator_ranks(
        _worker,
        directory,
        world_size=ulysses * ring,
        timeout=600,
        args=(name, ulysses, ring, latent_shape, numerics, str(output_path)),
    )
    return torch.load(output_path)


@pytest.fixture(scope="module")
def one_gpu_output(tmp_path_factory):
    """The one-GPU output for a model, latent shape and numerics, run once per module."""
    outputs = {}

    def get(name, latent_shape, numerics):
        key = (name, latent_shape, numerics)
        if key not in outputs:
            outputs[key] = _run(tmp_path_factory.mktemp("one-gpu"), name, 1, 1, latent_shape, numerics)
        return outputs[key]

    return get


def _skip_unless_available(numerics):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    backend = NUMERICS[numerics][0]
    if backend != "AITER_BF16":
        return
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType

    spec = registry.get(AttentionBackendType[backend])
    reason = spec.unavailable() or spec.ring.unmet()
    if reason is not None:
        pytest.skip(reason)


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("name", ["wan", "vace"])
def test_unpadded_ulysses_matches_one_gpu(tmp_path, one_gpu_output, name):
    _skip_unless_available("fp32-sdpa")
    expected = one_gpu_output(name, UNPADDED, "fp32-sdpa")

    actual = _run(tmp_path, name, 2, 1, UNPADDED, "fp32-sdpa")

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE["fp32-sdpa"], f"relative error {error:.3e}"


@pytest.mark.parametrize("numerics", NUMERICS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["wan", "vace"])
def test_padded_sequence_parallel_matches_one_gpu(tmp_path, one_gpu_output, name, mode, numerics):
    _skip_unless_available(numerics)
    ulysses, ring = MODES[mode]
    expected = one_gpu_output(name, PADDED, numerics)

    actual = _run(tmp_path, name, ulysses, ring, PADDED, numerics)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE[numerics], f"relative error {error:.3e}"
