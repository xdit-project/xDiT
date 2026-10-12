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

Wan's FP8 communication runs only under Ulysses, so its one-GPU reference has
the test wrap the attention kernel in the steps FP8 comms takes around it.
Every run also checks that the ranks holding pad rows keep them at zero in
every block's output.
"""

import dataclasses
from types import SimpleNamespace
from typing import NamedTuple

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


class Numerics(NamedTuple):
    backend: str
    dtype: torch.dtype
    # AITER_BF16 serves bf16 only, while the rest of the model stays fp32 (see the module docstring).
    attention_in_bf16: bool = False
    # Self-attention through Wan's FP8 communication; cross-attention on SDPA_EFFICIENT.
    fp8_comms: bool = False


NUMERICS = {
    "fp32-sdpa": Numerics("SDPA_EFFICIENT", torch.float32),
    "fp32-model-bf16-attention": Numerics("AITER_BF16", torch.float32, attention_in_bf16=True),
    "fp8-comms": Numerics("AITER_FP8", torch.float32, fp8_comms=True),
}
# Largest relative L2 error against one GPU. Measured with the pad dropped / attended: in fp32 <= 3.4e-7 /
# >= 3.3e-3, with bf16 attention <= 1.6e-4 / >= 3.3e-3, with FP8 comms <= 3.1e-7 / >= 3.7e-3; FP8 comms gets
# headroom for a value its quantization rounds the other way. Without a pad, a run that drops one real key
# errs by >= 2.2e-4 in fp32.
TOLERANCE = {"fp32-sdpa": 1e-5, "fp32-model-bf16-attention": 5e-4, "fp8-comms": 5e-4}
# A fixed scale, so the quantization does not depend on the data, whose pad rows differ between the runs. Q/K/V
# stay below ~5 here, inside FP8's range at this scale on every architecture.
FP8_COMMS_SCALE = 0.02


def _model(name):
    if name in ("wan", "wan-ti2v"):
        from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper

        return xFuserWanTransformer3DWrapper(**_TINY_CONFIG)
    from xfuser.model_executor.models.transformers.transformer_wan_vace import xFuserWanVACETransformer3DWrapper

    return xFuserWanVACETransformer3DWrapper(**_TINY_CONFIG, **_VACE_CONFIG)


def _seq_len(latent_shape):
    return int(torch.Size(latent_shape).numel()) // int(torch.Size(_TINY_CONFIG["patch_size"]).numel())


def _inputs(name, latent_shape, batch):
    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(batch, _TINY_CONFIG["in_channels"], *latent_shape, generator=generator),
        encoder_hidden_states=torch.randn(batch, TEXT_LEN, _TINY_CONFIG["text_dim"], generator=generator),
        timestep=torch.full((batch,), 500.0),
    )
    if name == "vace":
        inputs["control_hidden_states"] = torch.randn(
            batch, _VACE_CONFIG["vace_in_channels"], *latent_shape, generator=generator
        )
    if name == "wan-ti2v":
        # Wan2.2 TI2V: a timestep per token, 0 on the conditioning first frame. The model then pads and
        # splits the per-token modulation along with the tokens.
        tokens_per_frame = _seq_len(latent_shape) // latent_shape[0]
        inputs["timestep"] = torch.full((batch, _seq_len(latent_shape)), 500.0)
        inputs["timestep"][:, :tokens_per_frame] = 0
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


def _attend_as_fp8_comms(backend):
    """Have the kernel take what FP8 comms would hand it, for the one-GPU reference.

    FP8 comms runs only under Ulysses: USP rotates Q/K, quantizes Q/K/V with the
    layer's scales before the exchange, and quantizes the attention output for
    the exchange back. The exchanges move values unchanged, so the same steps
    with the same fixed scale around the kernel give one GPU the same numbers.
    """
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.distributed.fp8_comms import _per_tensor_quant, _rotate_for_backend

    backend = AttentionBackendType[backend]
    spec = registry.get(backend)
    kernel = spec.resolved()

    def kernel_as_fp8_comms(query, key, value, call):
        scale = torch.full((1,), FP8_COMMS_SCALE, device=query.device)
        query, key = _rotate_for_backend(query, key, backend)
        (query, q_descale), (key, k_descale), (value, v_descale) = (
            _per_tensor_quant(tensor, scale) for tensor in (query, key, value)
        )
        attention_kwargs = {
            **call.attention_kwargs,
            "pre_quantized": True,
            "q_descale": q_descale,
            "k_descale": k_descale,
            "v_descale": v_descale,
        }
        output, lse = kernel(query, key, value, dataclasses.replace(call, attention_kwargs=attention_kwargs))
        output_fp8, o_descale = _per_tensor_quant(output, scale)
        return (output_fp8.float() * o_descale).to(output.dtype), lse

    object.__setattr__(spec, "_resolved", kernel_as_fp8_comms)


def _watch_pad_rows(model, rows):
    """Record, for every block output, whether its last `rows` rows (this rank's SP pad) are zero."""
    zero = []

    def record(module, args, output):
        for tensor in output if isinstance(output, tuple) else (output,):
            if tensor is not None:
                zero.append(bool(tensor[:, -rows:].eq(0).all()))

    for block in [*model.blocks, *getattr(model, "vace_blocks", [])]:
        block.register_forward_hook(record)
    return zero


def _worker(rank, world_size, init_method, name, ulysses, ring, latent_shape, numerics, batch, output_path):
    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        get_sequence_parallel_rank,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    numerics = NUMERICS[numerics]
    fp8_comms = numerics.fp8_comms and ulysses > 1
    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=ulysses, ring_degree=ring)
    try:
        args = xFuserArgs(
            attention_backend=numerics.backend,
            ulysses_degree=ulysses,
            ring_degree=ring,
            use_fp8_comms=fp8_comms,
            fp8_comms_scale=FP8_COMMS_SCALE if fp8_comms else None,
        )
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        runtime_state = get_runtime_state()
        runtime_state.set_attention_backend(numerics.backend)
        # Without a pipeline there is no denoising schedule to advance, but the wrappers' forward asks to.
        runtime_state.increment_step_counter = lambda: None
        if numerics.attention_in_bf16:
            _attend_in_bf16(numerics.backend)
        if numerics.fp8_comms:
            runtime_state.set_cross_attention_backend("SDPA_EFFICIENT")
            if not fp8_comms:
                _attend_as_fp8_comms(numerics.backend)

        device = torch.device("cuda", rank)
        # Seeded on the CPU, so every process builds the same weights.
        torch.manual_seed(0)
        model = _model(name).eval().to(device=device, dtype=numerics.dtype)
        if fp8_comms:
            runtime_state.fp8_comms.register_models(SimpleNamespace(transformer=model))
        seq_len, sp_size = _seq_len(latent_shape), ulysses * ring
        local_len = -(-seq_len // sp_size)
        sp_rank = get_sequence_parallel_rank()
        pad_rows = sum(row >= seq_len for row in range(sp_rank * local_len, (sp_rank + 1) * local_len))
        pad_rows_zero = _watch_pad_rows(model, pad_rows) if pad_rows else None

        inputs = {
            key: tensor.to(device=device, dtype=tensor.dtype if key == "timestep" else numerics.dtype)
            for key, tensor in _inputs(name, latent_shape, batch).items()
        }
        with torch.no_grad():
            output = model(**inputs, return_dict=False)[0]

        if pad_rows_zero is not None:
            assert pad_rows_zero and all(pad_rows_zero), (
                f"SP rank {sp_rank}: its {pad_rows} pad rows are non-zero in "
                f"{pad_rows_zero.count(False)} of {len(pad_rows_zero)} block outputs"
            )
        if rank == 0:
            torch.save(output.cpu(), output_path)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run(directory, name, ulysses, ring, latent_shape, numerics, batch=1):
    output_path = directory / "output.pt"
    spawn_accelerator_ranks(
        _worker,
        directory,
        world_size=ulysses * ring,
        timeout=600,
        args=(name, ulysses, ring, latent_shape, numerics, batch, str(output_path)),
    )
    return torch.load(output_path)


@pytest.fixture(scope="module")
def one_gpu_output(tmp_path_factory):
    """The one-GPU output for a model, latent shape, numerics and batch size, run once per module."""
    outputs = {}

    def get(name, latent_shape, numerics, batch=1):
        key = (name, latent_shape, numerics, batch)
        if key not in outputs:
            outputs[key] = _run(tmp_path_factory.mktemp("one-gpu"), name, 1, 1, latent_shape, numerics, batch)
        return outputs[key]

    return get


def _skip_unless_available(numerics, ring=1):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import AttentionBackendType

    spec = registry.get(AttentionBackendType[NUMERICS[numerics].backend])
    reason = spec.unavailable() or (spec.ring.unmet() if ring > 1 else None)
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


@pytest.mark.parametrize("numerics", ["fp32-sdpa", "fp32-model-bf16-attention"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["wan", "vace", "wan-ti2v"])
def test_padded_sequence_parallel_matches_one_gpu(tmp_path, one_gpu_output, name, mode, numerics):
    ulysses, ring = MODES[mode]
    _skip_unless_available(numerics, ring)
    expected = one_gpu_output(name, PADDED, numerics)

    actual = _run(tmp_path, name, ulysses, ring, PADDED, numerics)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE[numerics], f"relative error {error:.3e}"


@pytest.mark.parametrize("numerics", ["fp32-sdpa", "fp32-model-bf16-attention"])
@pytest.mark.parametrize("name", ["wan", "vace"])
def test_padded_batch_of_two_matches_one_gpu(tmp_path, one_gpu_output, name, numerics):
    """Different samples per batch row, under both exchanges and the ring merge."""
    ulysses, ring = MODES["ulysses-ring"]
    _skip_unless_available(numerics, ring)
    expected = one_gpu_output(name, PADDED, numerics, batch=2)

    actual = _run(tmp_path, name, ulysses, ring, PADDED, numerics, batch=2)

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE[numerics], f"relative error {error:.3e}"


@pytest.mark.parametrize("ulysses", [2, 4])
def test_padded_ulysses_with_fp8_comms_matches_one_gpu(tmp_path, one_gpu_output, ulysses):
    """VACE registers no FP8-comms layers, and AITER_FP8 joins no ring, so Wan under Ulysses only."""
    _skip_unless_available("fp8-comms")
    expected = one_gpu_output("wan", PADDED, "fp8-comms")

    actual = _run(tmp_path, "wan", ulysses, 1, PADDED, "fp8-comms")

    error = _relative_error(actual, expected)
    assert error <= TOLERANCE["fp8-comms"], f"relative error {error:.3e}"
