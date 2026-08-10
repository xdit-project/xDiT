"""Checkpoint requests and file discovery, with no model/runtime dependencies."""

from dataclasses import dataclass, field, replace
import json
import os
from typing import Callable, Iterable, Mapping


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
        values = self.from_pretrained_kwargs(include_subfolder=include_subfolder)
        values.pop("variant", None)
        return values


@dataclass(frozen=True)
class CheckpointTensorRef:
    path: str
    checkpoint_key: str


@dataclass(frozen=True)
class DerivedTensor:
    """A live tensor computed from checkpoint tensors rather than copied from one.

    Covers the two mappings a name alone cannot express: a checkpoint that fuses
    what the model keeps separate, where three live tensors are slices of one
    stored tensor, and a checkpoint that stores a quantized weight beside the
    scale needed to read it, where one live tensor needs two stored ones.

    ``sources`` are checkpoint keys and ``build`` receives them in that order. It
    must be pure and cheap: it runs inside the per-block fill, once per tensor,
    while the fill holds at most one block, so anything it allocates is paid for
    at every block.
    """

    sources: tuple[str, ...]
    build: Callable[..., "torch.Tensor"]
    description: str = ""


@dataclass(frozen=True)
class CheckpointManifest:
    """Tensor-key mapping produced by discovery, before any tensor is read."""

    weight_map: dict[str, str]
    checkpoint_keys: dict[str, str] = field(default_factory=dict)
    strict: bool = False
    label: str | None = None
    derived: Mapping[str, DerivedTensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Every source has to sit in the shard the live key maps to. The filler keeps
        # at most one shard mapped, so a source elsewhere would either read through a
        # second handle, retaining a whole extra shard, or reopen per tensor. Checked
        # here, where the map is known, rather than discovered mid-fill.
        for live_key, derived in self.derived.items():
            path = self.weight_map.get(live_key)
            if path is None:
                raise ValueError(
                    f"derived tensor {live_key} names no shard in the weight map"
                )
            elsewhere = [
                source
                for source in derived.sources
                if self.weight_map.get(source, path) != path
            ]
            if elsewhere:
                raise ValueError(
                    f"derived tensor {live_key} reads {', '.join(elsewhere)} from "
                    "another shard; a derived tensor must be built from one shard"
                )

    @property
    def shard_paths(self) -> frozenset[str]:
        return frozenset(self.weight_map.values())

    def tensor_ref(self, live_key: str) -> CheckpointTensorRef:
        return CheckpointTensorRef(
            path=self.weight_map[live_key],
            checkpoint_key=self.checkpoint_keys.get(live_key, live_key),
        )


def _is_within(directory: str, path: str) -> bool:
    try:
        return os.path.commonpath([directory, path]) == directory
    except ValueError:
        return False


def _cached_component_config(request: CheckpointRequest) -> str | None:
    """This component's cached config, or None when its snapshot is not cached."""

    from huggingface_hub import try_to_load_from_cache

    filename = "config.json"
    if request.subfolder:
        filename = f"{request.subfolder}/{filename}"
    cached = try_to_load_from_cache(
        repo_id=os.fspath(request.model_name_or_path),
        filename=filename,
        cache_dir=request.cache_dir,
        revision=request.revision,
    )
    return cached if isinstance(cached, str) else None


def resolve_checkpoint_file(request: CheckpointRequest, filename: str) -> str | None:
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
        # With no network, a name the repo does not carry and a name that was never
        # cached raise the same error, since neither can be checked against the hub.
        # The component's own config settles it: if that is cached then this snapshot
        # is here, and the missing name is one this checkpoint does not use.
        if _cached_component_config(request) is None:
            raise
        return None
    except EntryNotFoundError:
        return None


def _require_checkpoint_file(request: CheckpointRequest, filename: str) -> str:
    path = resolve_checkpoint_file(request, filename)
    if path is None:
        location = request.model_name_or_path
        if request.subfolder:
            location = f"{location}/{request.subfolder}"
        raise FileNotFoundError(f"checkpoint file '{filename}' not found in {location}")
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


def resolve_mapped_checkpoint(
    path: str | os.PathLike,
    *,
    live_key: Callable[[str], str],
    key_reader: Callable[[str], Iterable[str]] = _safetensor_keys,
) -> CheckpointManifest:
    """Map one safetensors file's source keys to strict live model keys."""

    resolved = os.path.realpath(os.fspath(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"mapped checkpoint file not found: {resolved}")
    weight_map = {}
    checkpoint_keys = {}
    for checkpoint_key in key_reader(resolved):
        mapped = live_key(checkpoint_key)
        if not isinstance(mapped, str) or not mapped:
            raise ValueError(
                f"mapped checkpoint key {checkpoint_key!r} produced an empty live key"
            )
        previous = checkpoint_keys.get(mapped)
        if previous is not None:
            raise ValueError(
                f"mapped checkpoint collision for live key {mapped!r}: "
                f"{previous!r} and {checkpoint_key!r}"
            )
        weight_map[mapped] = resolved
        checkpoint_keys[mapped] = checkpoint_key
    return CheckpointManifest(
        weight_map=weight_map,
        checkpoint_keys=checkpoint_keys,
        strict=True,
        label=resolved,
    )


def discover_checkpoint(
    request: CheckpointRequest,
    *,
    basename: str = "diffusion_pytorch_model",
    key_reader: Callable[[str], Iterable[str]] = _safetensor_keys,
) -> CheckpointManifest:
    """Discover safetensor shards and keys without reading tensor payloads."""

    index = resolve_checkpoint_file(
        request,
        _variant_filename(request, f"{basename}.safetensors.index.json"),
    )
    if index is not None:
        with open(index) as handle:
            key_to_file = json.load(handle)["weight_map"]
        local_files: dict[str, str] = {}
        weight_map = {}
        for key, filename in key_to_file.items():
            if filename not in local_files:
                local_files[filename] = _require_checkpoint_file(request, filename)
            weight_map[key] = local_files[filename]
        return CheckpointManifest(weight_map)

    single = _require_checkpoint_file(
        request, _variant_filename(request, f"{basename}.safetensors")
    )
    return CheckpointManifest({key: single for key in key_reader(single)})


def component_shard_paths(request: CheckpointRequest, basename: str) -> set[str]:
    """Resolve component shard paths from an index, without reading tensors."""

    index = resolve_checkpoint_file(
        request,
        _variant_filename(request, f"{basename}.safetensors.index.json"),
    )
    if index is not None:
        with open(index) as handle:
            filenames = set(json.load(handle)["weight_map"].values())
        return {_require_checkpoint_file(request, filename) for filename in filenames}
    single = resolve_checkpoint_file(
        request, _variant_filename(request, f"{basename}.safetensors")
    )
    return {single} if single is not None else set()
