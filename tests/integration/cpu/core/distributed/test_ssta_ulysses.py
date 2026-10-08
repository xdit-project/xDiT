"""SSTA block masks under Ulysses must match the single-process masks.

After the Ulysses all-to-all each rank holds the full sequence for its share of
the heads. With ``attn_mask_share_within_head`` the mask is built from the
average over all heads, so every rank has to produce the mask that one process
holding every head produces.
"""

import queue
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gloo

_WORLD_SIZE = 2
_HEADS = 4
_HEAD_DIM = 8
_THW = (4, 4, 4)
_TILE = (2, 2, 2)
_TEXT_LEN = 16  # two text blocks
_TEXT_VALID = 11

# (sparse type, text tokens, sparse_text_to_image)
_CASES = [
    ("ssta", 0, False),
    ("ssta", _TEXT_LEN, False),
    ("ssta", _TEXT_LEN, True),
    ("moba", _TEXT_LEN, True),
]


def _attn_kwargs(*, share, sparse_type, encoder_sequence_length, sp_size, sparse_text_to_image, text_mask):
    return {
        "ssta_threshold": 0.0,
        "ssta_lambda": 0.7,
        "ssta_sampling_type": "importance",
        "ssta_adaptive_pool": None,
        "attn_pad_type": "zero",
        "attn_use_text_mask": 1,
        "text_mask": text_mask,
        "attn_mask_share_within_head": share,
        "attn_sparse_type": sparse_type,
        "encoder_sequence_length": encoder_sequence_length,
        "ssta_topk": 6,  # halved to 3 for short videos
        "thw": _THW,
        "tile_size": list(_TILE),
        "win_size": [[1, 1, 1]],
        "sp_size": sp_size,
        "sparse_text_to_image": sparse_text_to_image,
    }


def _masks(rank, world_size):
    """Return {(case, share): (reference mask, this rank's Ulysses mask)}."""
    import torch

    from xfuser.core.distributed.ssta import get_sparse_mask, setup_ssta

    generator = torch.Generator().manual_seed(0)
    image_len = _THW[0] * _THW[1] * _THW[2]
    heads_per_rank = _HEADS // world_size
    head_slice = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)

    results = {}
    for case in _CASES:
        sparse_type, text_len, sparse_text_to_image = case
        q, k, v = (torch.randn(1, _HEADS, image_len + text_len, _HEAD_DIM, generator=generator) for _ in range(3))
        text_mask = None
        if text_len:
            text_mask = torch.zeros(1, text_len, dtype=torch.int64)
            text_mask[:, :_TEXT_VALID] = 1

        def ulysses_layout(x):
            # What the Ulysses all-to-all hands this rank: its heads, with the
            # sequence as [img_r0, txt_r0, img_r1, txt_r1, ...].
            x = x[:, head_slice]
            if not text_len:
                return x
            img = x[:, :, :image_len].chunk(world_size, dim=2)
            txt = x[:, :, image_len:].chunk(world_size, dim=2)
            return torch.cat([part for pair in zip(img, txt) for part in pair], dim=2)

        for share in (0, 1):
            common = dict(share=share, sparse_type=sparse_type, sparse_text_to_image=sparse_text_to_image)
            ref_kwargs = _attn_kwargs(encoder_sequence_length=text_len, sp_size=1, text_mask=text_mask, **common)
            *_, ref_config, _ = setup_ssta(q, k, v, ref_kwargs)
            reference = get_sparse_mask(ref_config, sparse_type=sparse_type)

            sp_kwargs = _attn_kwargs(
                encoder_sequence_length=text_len // world_size,
                sp_size=world_size,
                text_mask=text_mask,
                **common,
            )
            *_, sp_config, _ = setup_ssta(ulysses_layout(q), ulysses_layout(k), ulysses_layout(v), sp_kwargs)
            ulysses = get_sparse_mask(sp_config, sparse_type=sparse_type)

            if not share:
                # Per-head masks: this rank owns a slice of the reference heads.
                reference = reference[:, head_slice]
            results[(case, share)] = (reference.numpy(), ulysses.numpy())
    return results


def _ssta_worker(rank, world_size, init_method, result_queue):
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
        # xFuser registers its Ulysses group only when yunchang can run (it needs
        # an accelerator); on CPU the whole world is the Ulysses group.
        sp_group = SimpleNamespace(ulysses_group=dist.group.WORLD)
        with patch("xfuser.core.distributed.parallel_state.get_sp_group", return_value=sp_group):
            result_queue.put(("returned", rank, _masks(rank, world_size)))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, *, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_ssta_worker, args=(rank, _WORLD_SIZE, init_method, result_queue))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()

    results = []
    deadline = time.monotonic() + timeout
    while len(results) < _WORLD_SIZE and time.monotonic() < deadline:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break

    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            process.kill()
            process.join(5)
    return processes, results


@pytest.mark.slow
def test_ssta_masks_under_ulysses_match_single_process(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is required for distributed process-group tests")
    np = pytest.importorskip("numpy")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, results = _run_spawned(torch, f"file://{tmp_path / 'ssta-ulysses-init'}", timeout=120)

    errors = [result for result in results if result[0] == "error"]
    assert not errors, errors
    assert len(results) == _WORLD_SIZE, f"missing worker results: {results}"
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE

    mismatches = []
    for _, rank, masks in sorted(results, key=lambda result: result[1]):
        for (case, share), (reference, ulysses) in masks.items():
            if reference.shape != ulysses.shape or not np.array_equal(reference, ulysses):
                mismatches.append((rank, case, f"share={share}"))
    assert not mismatches, f"Ulysses masks differ from the single-process masks: {mismatches}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
