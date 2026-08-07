"""Checkpoint file helpers for the memory-efficient FSDP load path.

Pure, model-agnostic utilities: resolve checkpoint tensor keys to local shard files (index-only,
no tensor read), enumerate a component's shard files, report the container's memory footprint, and
evict clean page-cache after a checkpoint is consumed. Used by ``meta_load`` to stream weights
per block/rank without materializing a full copy on host.

Checkpoint discovery delegates to the request-driven loading.checkpoint module; the public helpers
here retain the legacy string-based API for existing callers.
"""

import os

from xfuser.model_executor.models.runner_models.loading.checkpoint import (
    CheckpointRequest,
    component_shard_paths as _component_shard_paths,
    discover_checkpoint,
    resolve_checkpoint_file,
)


def host_mem_gb() -> str:
    """Container memory footprint (the number the OOM killer watches) as 'cur/anon/file GB'.

    cgroup-v2 memory.current + memory.stat {anon,file} (falls back to v1 total-only, then '?').
    Splitting anon vs reclaimable file cache tells apart a real ×N tensor blowup (anon, which
    page-cache eviction cannot touch) from mmap checkpoint cache (file). Host RAM, not VRAM, is
    the binding constraint on the memory-efficient FSDP load path.
    """

    def _read_int(path: str):
        try:
            with open(path) as f:
                return int(f.read())
        except OSError:
            return None

    cur = _read_int("/sys/fs/cgroup/memory.current")
    if cur is not None:
        anon = file = None
        try:
            with open("/sys/fs/cgroup/memory.stat") as f:
                for line in f:
                    k, _, v = line.partition(" ")
                    if k == "anon":
                        anon = int(v)
                    elif k == "file":
                        file = int(v)
        except OSError:
            pass
        a = f"{anon/1e9:.1f}" if anon is not None else "?"
        fl = f"{file/1e9:.1f}" if file is not None else "?"
        return f"{cur/1e9:.1f}/{a}/{fl}"
    v1 = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    return f"{v1/1e9:.1f}/?/?" if v1 is not None else "?"


def drop_file_page_cache(paths) -> None:
    """Drop clean page-cache for fully-read checkpoint files (Linux; no-op elsewhere).

    Under a cgroup-v2 memory limit, memory.current counts reclaimable file cache. mmap-reading
    the multi-GB checkpoints on every rank fills it faster than the kernel reclaims -> OOMKill
    even though the pages are clean. POSIX_FADV_DONTNEED evicts them so memory.current tracks
    only the live working set.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def warm_file_page_cache(path, chunk_bytes: int = 8 << 20) -> None:
    """Stream a checkpoint file into page cache so a later mmap read faults against warm pages.

    safetensors reads through mmap, whose faults are synchronous and one readahead window deep. On
    local NVMe that measured 0.62 GB/s against 3.21 GB/s for the same file streamed with pread and a
    ~6 GB/s device ceiling, so the fill was paying fault latency rather than bandwidth. Streaming the
    shard first turns those faults into cache hits: reading a 10GB shard to device went from 9.05s to
    4.39s, and it is why spreading reads across ranks is not needed to go faster.

    Costs no memory over doing nothing, which is the point: these are the same pages mmap would have
    faulted in anyway, they are clean and file-backed, and drop_file_page_cache still releases each
    shard once nothing needs it. Sequential pread is also why this is safe to do blind -- the kernel
    can drop what it cannot fit, leaving the mmap read correct but slower.

    Best-effort by design: a failure here only forfeits the speedup.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        offset = 0
        while True:
            block = os.pread(fd, chunk_bytes, offset)
            if not block:
                break
            offset += len(block)
    except OSError:
        pass
    finally:
        os.close(fd)


def resolve_repo_file(model_name: str | CheckpointRequest, relpath: str) -> str | None:
    """Local path of ``relpath`` within a checkpoint, or None when it is not there.

    ``model_name`` is either a Hub repo id or a path to a local checkpoint directory; a local
    directory is a normal deployment shape (pre-downloaded, air-gapped, or a converted export),
    and hf_hub_download rejects one outright with HFValidationError. Resolve the local case by
    path and only ask the Hub otherwise.

    Returns None rather than raising for a genuinely absent file, since callers probe for optional
    files (a shard index that only exists for sharded checkpoints) to decide which layout they are
    reading. Absence is the only thing that maps to None. Network and auth failures propagate.

    LocalEntryNotFoundError needs its own re-raise: it subclasses EntryNotFoundError but means the
    file could not be reached (offline, HF_HUB_OFFLINE, an unresolvable host). Swallow it and a
    sharded checkpoint reads as unsharded, then the run fails against the single-file name, blaming
    a file that was never the problem.
    """
    request = (
        model_name
        if isinstance(model_name, CheckpointRequest)
        else CheckpointRequest(model_name)
    )
    return resolve_checkpoint_file(request, relpath)


def resolve_checkpoint_weight_map(
    model_name: str | CheckpointRequest, subfolder: str | None = None
) -> dict:
    """Map every checkpoint tensor key -> local safetensors file path for a component.

    Downloads only the index + shard files (cached if already present), never loads tensors.
    Handles both sharded (index.json + shards) and single-file checkpoints, from a Hub repo id or
    a local checkpoint directory. Used by the transformer self-fill path to read individual tensors
    lazily per block/rank.
    """
    request = (
        model_name
        if isinstance(model_name, CheckpointRequest)
        else CheckpointRequest(model_name, subfolder=subfolder)
    )
    if subfolder is not None and request.subfolder != subfolder:
        request = request.with_subfolder(subfolder)
    return discover_checkpoint(request).weight_map


def _require_repo_file(model_name: str | CheckpointRequest, relpath: str) -> str:
    """resolve_repo_file for a file the checkpoint layout says must exist."""
    path = resolve_repo_file(model_name, relpath)
    if path is None:
        raise FileNotFoundError(
            f"checkpoint file '{relpath}' not found in {model_name}"
        )
    return path


def component_shard_paths(
    model_name: str | CheckpointRequest,
    subfolder: str | None = None,
    basename: str | None = None,
) -> set:
    """Local safetensors file paths for a component, no tensor read (index-only).

    basename distinguishes diffusers ("diffusion_pytorch_model") from transformers ("model")
    checkpoint naming. Downloads only the index + shard files (cached if present); resolves a local
    checkpoint directory by path. Used to drop a component's page cache after it has been consumed.

    Returns an empty set when the component has no safetensors of that name (e.g. a .bin-only
    checkpoint), which only costs the page-cache drop for it.
    """
    if isinstance(model_name, CheckpointRequest) and basename is None:
        basename, subfolder = subfolder, None
    if basename is None:
        raise TypeError("basename is required")
    request = (
        model_name
        if isinstance(model_name, CheckpointRequest)
        else CheckpointRequest(model_name, subfolder=subfolder)
    )
    if subfolder is not None and request.subfolder != subfolder:
        request = request.with_subfolder(subfolder)
    return _component_shard_paths(request, basename)
