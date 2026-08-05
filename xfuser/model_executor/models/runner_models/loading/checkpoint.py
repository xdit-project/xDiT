"""Checkpoint requests and file discovery, with no model/runtime dependencies."""

from dataclasses import dataclass, replace
import json
import os
from typing import Callable, Iterable


@dataclass(frozen=True)
class CheckpointRequest:
    model_name_or_path: str
    subfolder: str | None = None
    revision: str | None = None
    variant: str | None = None
    token: str | bool | None = None
    cache_dir: str | os.PathLike | None = None
    local_files_only: bool = False

    def with_subfolder(self, subfolder: str | None) -> "CheckpointRequest":
        return replace(self, subfolder=subfolder)

    def from_pretrained_kwargs(self, *, include_subfolder: bool = True) -> dict:
        values = {
            "revision": self.revision,
            "variant": self.variant,
            "token": self.token,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if include_subfolder:
            values = {"subfolder": self.subfolder, **values}
        return {key: value for key, value in values.items() if value is not None}

    def hub_kwargs(self) -> dict:
        return {
            key: value
            for key, value in {
                "revision": self.revision,
                "token": self.token,
                "cache_dir": self.cache_dir,
                "local_files_only": self.local_files_only,
            }.items()
            if value is not None
        }

    def config_kwargs(self, *, include_subfolder: bool = True) -> dict:
        """from_pretrained kwargs accepted by config-only loading APIs."""
        values = self.from_pretrained_kwargs(
            include_subfolder=include_subfolder
        )
        values.pop("variant", None)
        return values


@dataclass(frozen=True)
class CheckpointManifest:
    """Tensor-key mapping produced by discovery, before any tensor is read."""

    weight_map: dict[str, str]

    @property
    def shard_paths(self) -> frozenset[str]:
        return frozenset(self.weight_map.values())


def _is_within(directory: str, path: str) -> bool:
    try:
        return os.path.commonpath([directory, path]) == directory
    except ValueError:
        return False


def resolve_checkpoint_file(
    request: CheckpointRequest, filename: str
) -> str | None:
    """Resolve one request-relative file; only genuine absence maps to None."""

    root = os.fspath(request.model_name_or_path)
    if os.path.isdir(root):
        model_root = os.path.realpath(root)
        checkpoint_dir = os.path.realpath(
            os.path.join(model_root, request.subfolder or "")
        )
        if not _is_within(model_root, checkpoint_dir):
            raise ValueError(
                f"local checkpoint subfolder {request.subfolder!r} resolves "
                f"outside model root {model_root!r}"
            )
        local = os.path.realpath(os.path.join(checkpoint_dir, filename))
        if not _is_within(checkpoint_dir, local):
            raise ValueError(
                f"local checkpoint file {filename!r} resolves outside "
                f"checkpoint directory {checkpoint_dir!r}"
            )
        return local if os.path.isfile(local) else None

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=root,
            filename=filename,
            subfolder=request.subfolder,
            **request.hub_kwargs(),
        )
    except LocalEntryNotFoundError:
        raise
    except EntryNotFoundError:
        return None


def _require_checkpoint_file(
    request: CheckpointRequest, filename: str
) -> str:
    path = resolve_checkpoint_file(request, filename)
    if path is None:
        location = request.model_name_or_path
        if request.subfolder:
            location = f"{location}/{request.subfolder}"
        raise FileNotFoundError(
            f"checkpoint file '{filename}' not found in {location}"
        )
    return path


def _variant_filename(request: CheckpointRequest, filename: str) -> str:
    if not request.variant:
        return filename
    marker = ".safetensors"
    stem, separator, suffix = filename.partition(marker)
    if separator:
        return f"{stem}.{request.variant}{separator}{suffix}"
    stem, extension = filename.rsplit(".", 1)
    return f"{stem}.{request.variant}.{extension}"


def _safetensor_keys(path: str) -> Iterable[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return tuple(handle.keys())


def discover_checkpoint(
    request: CheckpointRequest,
    *,
    basename: str = "diffusion_pytorch_model",
    key_reader: Callable[[str], Iterable[str]] = _safetensor_keys,
) -> CheckpointManifest:
    """Discover safetensor shards and keys without reading tensor payloads."""

    index = resolve_checkpoint_file(
        request,
        _variant_filename(
            request, f"{basename}.safetensors.index.json"
        ),
    )
    if index is not None:
        with open(index) as handle:
            key_to_file = json.load(handle)["weight_map"]
        local_files: dict[str, str] = {}
        weight_map = {}
        for key, filename in key_to_file.items():
            if filename not in local_files:
                local_files[filename] = _require_checkpoint_file(
                    request, filename
                )
            weight_map[key] = local_files[filename]
        return CheckpointManifest(weight_map)

    single = _require_checkpoint_file(
        request, _variant_filename(request, f"{basename}.safetensors")
    )
    return CheckpointManifest({key: single for key in key_reader(single)})


def component_shard_paths(
    request: CheckpointRequest, basename: str
) -> set[str]:
    """Resolve component shard paths from an index, without reading tensors."""

    index = resolve_checkpoint_file(
        request,
        _variant_filename(
            request, f"{basename}.safetensors.index.json"
        ),
    )
    if index is not None:
        with open(index) as handle:
            filenames = set(json.load(handle)["weight_map"].values())
        return {
            _require_checkpoint_file(request, filename)
            for filename in filenames
        }
    single = resolve_checkpoint_file(
        request, _variant_filename(request, f"{basename}.safetensors")
    )
    return {single} if single is not None else set()
