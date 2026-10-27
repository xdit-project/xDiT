"""USP over gloo ignores a declared trailing sequence-parallel pad.

The pad is what a model appends so the sequence splits evenly across ranks. Its
keys must not be attended: in Wan they get a zero logit and act as an
attention sink. USP drops them given the real length (`valid_kv_len`); in ring
attention it does so inside the step that attends to the last rank's shard.
That needs the real Ulysses and ring exchanges, hence processes over gloo, and
torch.compile has to trace it without a graph break.

Joint tokens (a replicated prompt joined after the exchange) are not part of
the declared length. The kernel here crops K to `valid_kv_len` when the kwargs
still carry it, as the dense fallback of AITER_VSA does, so a length that
outlived the trim would cut the joint keys' share of K off again.

A key-padding mask (`key_padding_mask_attention_kwargs`, for batched prompts of
different lengths) is over the keys USP attends to: the trimmed main keys, with
the joint keys where they are joined. Its cases check that order against SDPA
over the same keys with the same mask, eagerly and under torch.compile.

Replicated joint text must remain visible exactly once while the declared real
length excludes only image padding. Check image and text queries independently
of the compiler assertions against unpadded CPU attention.
"""

import os
import queue
import shutil
import time
import traceback
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

HEADS, HEAD_DIM, PADDED_LEN = 4, 16, 48
CASES = {
    "no-pad": dict(real_len=PADDED_LEN),
    "declared-pad": dict(real_len=45, declared=True),
    "declared-pad-causal": dict(real_len=45, declared=True, causal=True),
}
JOINT_LEN = 5
JOINT_CASES = {
    "joint-front-no-pad": dict(real_len=PADDED_LEN, joint="front", joint_query=True),
    "joint-front-declared-pad": dict(real_len=45, declared=True, joint="front", joint_query=True),
    "joint-rear-declared-pad": dict(real_len=45, declared=True, joint="rear", joint_query=True),
    # Only the keys are joint, as for the understanding tokens of Cosmos3.
    "joint-front-keys-only": dict(real_len=45, declared=True, joint="front"),
}
# A mask hides keys 7-10 and the third from last of the keys USP attends to (joint ones included).
KEY_MASK_CASES = {
    "key-mask-no-pad": dict(real_len=PADDED_LEN, key_mask=True),
    "key-mask-declared-pad": dict(real_len=45, declared=True, key_mask=True),
    "key-mask-joint-front": dict(real_len=45, declared=True, joint="front", joint_query=True, key_mask=True),
    "key-mask-joint-rear": dict(real_len=45, declared=True, joint="rear", joint_query=True, key_mask=True),
}
# (Ulysses, ring) degrees. At ring degree 4 only one of a rank's four steps attends to the pad.
MODES = {"ulysses": (2, 1), "ring": (1, 2), "hybrid": (2, 2), "ring4": (1, 4), "hybrid-ring4": (2, 4)}
# Every variant computes in fp32 on the same CPU kernel; the leak of an attended pad is several orders larger.
TOLERANCE = 1e-5


def _inputs(torch, real_len):
    generator = torch.Generator().manual_seed(0)
    query, key, value = (torch.randn(1, HEADS, PADDED_LEN, HEAD_DIM, generator=generator) for _ in range(3))
    # As in Wan: the zero-padded RoPE tables zero the pad rows of Q and K, while V keeps a projection of them.
    query[:, :, real_len:] = 0
    key[:, :, real_len:] = 0
    value[:, :, real_len:] *= 3
    return query, key, value


def _wire_usp_to_gloo(torch, ulysses, ring):
    """Point usp at yunchang's gloo groups and a CPU kernel that returns the log-sum-exp ring attention merges on.

    xDiT hands yunchang's groups to usp only on CUDA or NPU machines. Each
    getter returns a precomputed value, as xDiT's own do: one that built or
    queried anything while being traced would be a graph break of the test's
    own making.
    """
    import torch.distributed as dist
    from yunchang import set_seq_parallel_pg
    from yunchang.globals import PROCESS_GROUP

    from xfuser.core.attention.constraints import ANY_CALL
    from xfuser.core.attention.requirements import ALWAYS
    from xfuser.core.attention.spec import AttentionBackendType, Spec
    from xfuser.model_executor.layers import usp

    set_seq_parallel_pg(ulysses, ring, dist.get_rank(), dist.get_world_size())

    def cpu_flash(query, key, value, call):
        valid_kv_len = call.attention_kwargs.get("valid_kv_len")
        if valid_kv_len is not None and valid_kv_len < key.shape[2]:
            key, value = key[:, :, :valid_kv_len], value[:, :, :valid_kv_len]
        attn_mask = call.attention_kwargs.get("attn_mask")
        if attn_mask is not None:
            # Like SDPA, which applies the mask: no log-sum-exp, so no ring attention.
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask), None
        return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            query, key, value, dropout_p=0.0, is_causal=call.is_causal
        )[:2]

    spec = Spec(AttentionBackendType.SDPA_FLASH, impl=cpu_flash, ring=ALWAYS, accepts=ANY_CALL, requires=ALWAYS)
    attention_function = usp.concat_joint_tensors_decorator(usp._spec_adapter(spec))
    ulysses_rank = dist.get_rank(PROCESS_GROUP.ULYSSES_PG)
    ring_rank = dist.get_rank(PROCESS_GROUP.RING_PG)
    runtime_state = SimpleNamespace(
        fp8_comms=None,
        # Selected for the mask: the registry's SDPA applies attn_mask, as the kernel above does.
        attention_backend=AttentionBackendType.SDPA,
        runtime_config=SimpleNamespace(use_spargeattn_head_balance=False),
    )
    usp.PROCESS_GROUP = PROCESS_GROUP
    usp.get_sequence_parallel_world_size = lambda: ulysses * ring
    usp.get_ulysses_parallel_world_size = lambda: ulysses
    usp.get_ring_parallel_world_size = lambda: ring
    usp.get_ulysses_parallel_rank = lambda: ulysses_rank
    usp.get_ring_parallel_rank = lambda: ring_rank
    usp.get_runtime_state = lambda: runtime_state
    usp._get_attention_function = lambda backend=None: attention_function
    return usp


def _joint_inputs(torch):
    generator = torch.Generator().manual_seed(1)
    return [torch.randn(1, HEADS, JOINT_LEN, HEAD_DIM, generator=generator) for _ in range(3)]


def _key_mask(torch, case):
    """The [1, keys] mask of a key-mask case over the keys USP attends to, or None."""
    if not case.get("key_mask"):
        return None
    keys = case["real_len"] + (JOINT_LEN if case.get("joint") else 0)
    mask = torch.ones(1, keys, dtype=torch.bool)
    mask[:, 7:11] = False
    mask[:, -3] = False
    return mask


def _padded_key_mask(torch, case):
    """The mask over the keys the backend sees: the real keys' mask with the False pad rows of the main keys in it."""
    mask = _key_mask(torch, case)
    if mask is None:
        return None
    pad = torch.zeros(1, PADDED_LEN - case["real_len"], dtype=torch.bool)
    joint = JOINT_LEN if case.get("joint") else 0
    if case.get("joint") == "front":
        return torch.cat([mask[:, :joint], mask[:, joint:], pad], dim=1)
    return torch.cat([mask[:, : mask.shape[1] - joint], pad, mask[:, mask.shape[1] - joint :]], dim=1)


def _real_rows_error(torch, rank, world_size, case, output):
    query, key, value = (tensor[:, :, : case["real_len"]] for tensor in _inputs(torch, case["real_len"]))
    if case.get("joint"):
        front = case["joint"] == "front"
        _, joint_key, joint_value = _joint_inputs(torch)
        key, value = (
            torch.cat((joint, main) if front else (main, joint), dim=2)
            for joint, main in ((joint_key, key), (joint_value, value))
        )
        if case.get("joint_query"):
            # With joint queries the output also holds their rows, ahead of or behind the main ones.
            output = output[:, :, JOINT_LEN:] if front else output[:, :, :-JOINT_LEN]
    mask = _key_mask(torch, case)
    if mask is not None:
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None, :])
    else:
        expected = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            query, key, value, is_causal=case.get("causal", False)
        )[0]
    chunk = PADDED_LEN // world_size
    rows = range(rank * chunk, min((rank + 1) * chunk, case["real_len"]))
    if not rows:
        return 0.0
    local_rows = slice(rows.start - rank * chunk, rows.stop - rank * chunk)
    return (output[:, :, local_rows] - expected[:, :, rows.start : rows.stop]).abs().max().item()


def _local_inputs(torch, rank, world_size, real_len):
    chunk = PADDED_LEN // world_size
    return [tensor[:, :, rank * chunk : (rank + 1) * chunk].contiguous() for tensor in _inputs(torch, real_len)]


def _run_cases(torch, usp, rank, world_size, attention, cases):
    errors = {}
    for name, case in cases.items():
        local = _local_inputs(torch, rank, world_size, case["real_len"])
        attention_kwargs = {"valid_kv_len": case["real_len"]} if case.get("declared") else None
        joint = {}
        if case.get("joint"):
            joint_query, joint_key, joint_value = _joint_inputs(torch)
            joint = dict(joint_key=joint_key, joint_value=joint_value, joint_strategy=case["joint"])
            if case.get("joint_query"):
                joint["joint_query"] = joint_query
        output = attention(
            *local,
            is_causal=case.get("causal", False),
            attention_kwargs=attention_kwargs,
            key_mask=_padded_key_mask(torch, case),
            **joint,
        )
        errors[name] = _real_rows_error(torch, rank, world_size, case, output)
    return errors


def _usp_with_key_mask(usp):
    """USP as a wrapper calls it: the key padding first (the mask over every key, the pad included), then USP."""
    from xfuser.model_executor.layers.attention_mask import MaskMetaCache

    cache = MaskMetaCache()

    def call(query, key, value, key_mask=None, attention_kwargs=None, **kwargs):
        padding = None if key_mask is None else usp.key_padding_mask(cache, None, lambda _key: key_mask)
        attention_kwargs = usp.key_padding_attention_kwargs(attention_kwargs, padding)
        return usp.USP(query, key, value, attention_kwargs=attention_kwargs, **kwargs)

    return call


def _run_joint_cases(torch, usp, rank, world_size, attention):
    """A FLUX-style replicated prompt attends to the same unpadded image as local image queries.

    The prompt goes in front of the image keys, or behind them as in HunyuanVideo.
    """
    text_tokens = 7
    generator = torch.Generator().manual_seed(17)
    text = [torch.randn(1, HEADS, text_tokens, HEAD_DIM, generator=generator) for _ in range(3)]
    chunk = PADDED_LEN // world_size
    errors = {}
    for strategy in ("front", "rear"):
        for name, real_len in (("no-pad", PADDED_LEN), ("declared-pad", 45)):
            image = _inputs(torch, real_len)
            local = [tensor[:, :, rank * chunk : (rank + 1) * chunk].contiguous() for tensor in image]
            real_local_len = min(chunk, real_len - rank * chunk)
            real = [local[0][:, :, :real_local_len], image[1][:, :, :real_len], image[2][:, :, :real_len]]
            joined = [
                torch.cat([text[i], real[i]] if strategy == "front" else [real[i], text[i]], dim=2) for i in range(3)
            ]
            expected = torch.nn.functional.scaled_dot_product_attention(*joined)
            output = attention(
                *local,
                joint_query=text[0],
                joint_key=text[1],
                joint_value=text[2],
                joint_strategy=strategy,
                combine_qkv_a2a=True,
                attention_kwargs={"valid_kv_len": real_len},
            )
            assert output.shape == (1, HEADS, text_tokens + chunk, HEAD_DIM)
            if strategy == "front":
                rows = {"text": (slice(0, text_tokens), slice(0, text_tokens))}
                rows["image"] = (slice(text_tokens, text_tokens + real_local_len), slice(text_tokens, None))
            else:
                rows = {"text": (slice(chunk, None), slice(real_local_len, None))}
                rows["image"] = (slice(0, real_local_len), slice(0, real_local_len))
            for stream, (got, want) in rows.items():
                errors[f"{strategy}-{name}-{stream}"] = (output[:, :, got] - expected[:, :, want]).abs().max().item()
    return errors


def _packed_keys_error(torch, usp, rank, world_size):
    """What USP raises for packed keys with ring attention, which it cannot slice per step."""
    local = _local_inputs(torch, rank, world_size, 45)
    attention_kwargs = {
        "valid_kv_len": 45,
        "indices_k": torch.arange(45),
        "cu_seqlens_k": torch.tensor([0, 45], dtype=torch.int32),
        "max_seqlen_k": 45,
    }
    try:
        usp.USP(*local, attention_kwargs=attention_kwargs)
    except ValueError as error:
        return str(error)
    return None


def _worker(rank, world_size, init_method, result_queue, ulysses, ring, compile_backends, cases, scenario="plain"):
    # The CPU path is the one under test, whatever accelerators the machine has.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    dist = None
    try:
        import torch
        import torch.distributed as dist

        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
        usp = _wire_usp_to_gloo(torch, ulysses, ring)
        entry = _usp_with_key_mask(usp)
        if scenario == "plain":

            def run_cases(attention):
                return _run_cases(torch, usp, rank, world_size, attention, cases)

            wrapped = entry
        else:

            def run_cases(attention):
                return _run_joint_cases(torch, usp, rank, world_size, attention)

            wrapped = usp.USP

        report = {"errors": {"eager": run_cases(wrapped)}, "graph_breaks": {}}
        for backend in compile_backends:
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            compiled = torch.compile(wrapped, backend=backend)
            report["errors"][backend] = run_cases(compiled)
            report["graph_breaks"][backend] = sum(torch._dynamo.utils.counters["graph_break"].values())
        if ring > 1 and scenario == "plain":
            report["packed_keys_error"] = _packed_keys_error(torch, usp, rank, world_size)
        result_queue.put(("returned", rank, report))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, world_size, args, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, world_size, init_method, result_queue, *args))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)
    results = []
    while len(results) < world_size:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return [process.exitcode for process in processes], results


def _compile_backends():
    """aot_eager always: graph breaks are Dynamo's, whatever compiles the graph. Inductor too where it can build."""
    from torch._inductor import config

    return ["aot_eager"] + (["inductor"] if any(cxx and shutil.which(cxx) for cxx in config.cpp.cxx) else [])


@pytest.mark.parametrize("mode", MODES)
def test_usp_matches_attention_over_the_real_tokens(tmp_path, mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("yunchang", reason="yunchang builds the Ulysses and ring process groups")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")
    ulysses, ring = MODES[mode]
    world_size = ulysses * ring
    compile_backends = _compile_backends()

    exit_codes, results = _run_spawned(
        torch, f"file://{tmp_path / 'usp-init'}", world_size, (ulysses, ring, compile_backends, CASES), timeout=900
    )

    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert exit_codes == [0] * world_size
    assert sorted(result[1] for result in results) == list(range(world_size))
    for _, rank, report in results:
        for variant, case_errors in report["errors"].items():
            for case, error in case_errors.items():
                assert error <= TOLERANCE, f"rank {rank}, {variant}, {case}: max error {error:.3e}"
        assert report["graph_breaks"] == {backend: 0 for backend in compile_backends}, f"rank {rank}"
        if ring > 1:
            assert report["packed_keys_error"] is not None and "valid_kv_len must be in" in report["packed_keys_error"]


@pytest.mark.parametrize("mode", ["ulysses", "ring", "hybrid"])
def test_usp_with_joint_tokens_attends_them_once_and_drops_only_the_main_pad(tmp_path, mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("yunchang", reason="yunchang builds the Ulysses and ring process groups")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")
    ulysses, ring = MODES[mode]
    world_size = ulysses * ring

    exit_codes, results = _run_spawned(
        torch, f"file://{tmp_path / 'usp-init'}", world_size, (ulysses, ring, [], JOINT_CASES), timeout=300
    )

    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert exit_codes == [0] * world_size
    for _, rank, report in results:
        for case, error in report["errors"]["eager"].items():
            assert error <= TOLERANCE, f"rank {rank}, {case}: max error {error:.3e}"


def test_usp_with_a_key_padding_mask_attends_the_keys_in_the_order_the_mask_describes(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("yunchang", reason="yunchang builds the Ulysses and ring process groups")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")
    ulysses, ring = MODES["ulysses"]
    world_size = ulysses * ring
    compile_backends = _compile_backends()

    exit_codes, results = _run_spawned(
        torch,
        f"file://{tmp_path / 'usp-init'}",
        world_size,
        (ulysses, ring, compile_backends, KEY_MASK_CASES),
        timeout=600,
    )

    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert exit_codes == [0] * world_size
    for _, rank, report in results:
        for backend, backend_errors in report["errors"].items():
            for case, error in backend_errors.items():
                assert error <= TOLERANCE, f"rank {rank}, {backend}, {case}: max error {error:.3e}"
        assert report["graph_breaks"] == {backend: 0 for backend in compile_backends}, f"rank {rank}"


@pytest.mark.parametrize("mode", MODES)
def test_usp_joint_text_and_padded_image_match_unpadded_attention(tmp_path, mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("yunchang", reason="yunchang builds the Ulysses and ring process groups")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")
    ulysses, ring = MODES[mode]
    world_size = ulysses * ring

    exit_codes, results = _run_spawned(
        torch,
        f"file://{tmp_path / 'usp-joint-init'}",
        world_size,
        (ulysses, ring, ["aot_eager"], {}, "joint"),
        timeout=900,
    )

    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert exit_codes == [0] * world_size
    assert sorted(result[1] for result in results) == list(range(world_size))
    for _, rank, report in results:
        for variant, case_errors in report["errors"].items():
            for case, error in case_errors.items():
                assert error <= TOLERANCE, f"rank {rank}, {variant}, {case}: max error {error:.3e}"
        assert report["graph_breaks"] == {"aot_eager": 0}, f"rank {rank}"
