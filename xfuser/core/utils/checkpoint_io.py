"""Checkpoint file helpers for the memory-efficient FSDP load path.

Pure, model-agnostic utilities: resolve checkpoint tensor keys to local shard files (index-only,
no tensor read), enumerate a component's shard files, report the container's memory footprint, and
evict clean page-cache after a checkpoint is consumed. Used by ``meta_load`` to stream weights
per block/rank without materializing a full copy on host.
"""

import json
import os


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


def resolve_checkpoint_weight_map(model_name: str, subfolder: str) -> dict:
    """Map every checkpoint tensor key -> local safetensors file path for a component.

    Downloads only the index + shard files (cached if already present), never loads tensors.
    Handles both sharded (index.json + shards) and single-file checkpoints. Used by the
    transformer self-fill path to read individual tensors lazily per block/rank.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    index_name = f"{subfolder}/diffusion_pytorch_model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    try:
        idx_path = hf_hub_download(model_name, index_name)
    except Exception:
        idx_path = None
    if idx_path is not None:
        with open(idx_path) as f:
            key_to_file = json.load(f)["weight_map"]
        file_local: dict[str, str] = {}
        for key, fname in key_to_file.items():
            if fname not in file_local:
                file_local[fname] = hf_hub_download(model_name, f"{subfolder}/{fname}")
            weight_map[key] = file_local[fname]
    else:
        single = hf_hub_download(
            model_name, f"{subfolder}/diffusion_pytorch_model.safetensors"
        )
        with safe_open(single, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = single
    return weight_map


def component_shard_paths(model_name: str, subfolder: str, basename: str) -> set:
    """Local safetensors file paths for a component, no tensor read (index-only).

    basename distinguishes diffusers ("diffusion_pytorch_model") from transformers ("model")
    checkpoint naming. Downloads only the index + shard files (cached if present). Used to drop
    a component's page cache after it has been consumed.
    """
    from huggingface_hub import hf_hub_download
    try:
        idx_path = hf_hub_download(model_name, f"{subfolder}/{basename}.safetensors.index.json")
    except Exception:
        idx_path = None
    if idx_path is not None:
        with open(idx_path) as f:
            fnames = set(json.load(f)["weight_map"].values())
        return {hf_hub_download(model_name, f"{subfolder}/{fn}") for fn in fnames}
    try:
        return {hf_hub_download(model_name, f"{subfolder}/{basename}.safetensors")}
    except Exception:
        return set()
