import os
import csv
import logging
import torch
import functools
import numpy as np
from PIL.Image import Image
from typing import Callable, Optional, Tuple
from xfuser.envs import _is_cuda, _is_hip, PACKAGES_CHECKER

logger = logging.getLogger(__name__)


def _use_aiter_fp8_rdna4() -> bool:
    """True on ROCm gfx1200 (Navi 44) or gfx1201 (Navi 48) with AITER present."""
    if not _is_hip():
        return False
    try:
        import aiter  # noqa: F401
    except ImportError:
        return False
    return PACKAGES_CHECKER._on_rdna4()

def log(message: str, debug=False, log_from_all_processes: bool = False) -> None:
    """Log message. By default, only from the last process to avoid duplicates."""
    if log_from_all_processes or is_last_process():
        if debug:
            logger.debug(message)
        else:
            logger.info(message)

def is_last_process() -> bool:
    """
    Checks based on env rank and world size if this is last process in
    Has to be the last process, as legacy xDiT models only produce the
    output on the last GPU.
    """
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    return rank == world_size - 1

def resize_image_to_max_area(image: Image, input_height: int, input_width: int, mod_value: int) -> Image:
    """ Resize image to fit within max area while retaining aspect ratio """

    max_area = input_height * input_width
    width, height = image.size
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area /aspect_ratio)) // mod_value * mod_value

    image = image.resize((width, height))
    log(f"Resized image to {image.width}x{image.height} to fit within max area {width}x{height}")
    return image

def resize_and_crop_image(image: Image, target_height: int, target_width: int, mod_value: int) -> Image:
        """ Resize and center-crop image to target dimensions """

        target_height_aligned = target_height // mod_value * mod_value
        target_width_aligned = target_width // mod_value * mod_value

        log("Force output size mode enabled.")
        log(f"Input image resolution: {image.height}x{image.width}")
        log(f"Requested output resolution: {target_height}x{target_width}")
        log(f"Aligned output resolution (multiple of {mod_value}): {target_height_aligned}x{target_width_aligned}")

        # Step 1: Resize image maintaining aspect ratio so both dimensions >= target
        img_width, img_height = image.size

        # Calculate scale factor to ensure both dimensions are at least target size
        scale_width = target_width_aligned / img_width
        scale_height = target_height_aligned / img_height
        scale = max(scale_width, scale_height)  # Use max to ensure both dims are >= target

        # Resize with aspect ratio preserved
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        image = image.resize((new_width, new_height))

        log(f"Resized image to: {new_height}x{new_width} (maintaining aspect ratio)")

        # Step 2: Crop from center to get exact target dimensions
        left = (new_width - target_width_aligned) // 2
        top = (new_height - target_height_aligned) // 2
        image = image.crop((left, top, left + target_width_aligned, top + target_height_aligned))

        log(f"Cropped from center to: {target_height_aligned}x{target_width_aligned}")
        return image

def _get_fp8_kernel_preference():
    """Select FP8 kernel preference based on GPU architecture.

    AUTO selects CUTLASS sm90a kernels which crash on Blackwell (sm100a)
    under torch.compile, so we force TORCH (_scaled_mm) on Blackwell+.
    """
    from torchao.quantization.quantize_.common import KernelPreference
    if torch.cuda.is_available() and _is_cuda() and torch.cuda.get_device_capability()[0] >= 10:
        return KernelPreference.TORCH
    return KernelPreference.AUTO


# Smallest value the dynamic activation scale may be derived from. Without it a tensor whose values
# are all zero gets scale = max_abs / 448 = 0, and quantizing divides by that scale, so the layer
# returns all NaN instead of the bias. Zeros arrive in normal use: a model that zero-pads its text
# sequence up to a multiple of the sequence-parallel world size hands whole padding-only chunks to
# late ranks — Qwen-Image at eight ranks is one — and the NaN then spreads through the attention
# all-to-all until every render is black. Matches torchao's own EPS on its training
# path, and only binds when the largest value in the tensor is below it, so nothing else moves:
# measured bit-identical output on ordinary activations.
FP8_ACTIVATION_SCALE_FLOOR = 1e-12


def _float8_tensor_op_table(float8_cls: type) -> dict:
    """Return the mutable per-class op table (torchao 0.14+ naming varies)."""
    for attr in ("_ATEN_OP_TABLE", "_ATEN_OP_OR_TORCH_FN_TABLE"):
        root = getattr(float8_cls, attr, None)
        if root is not None:
            if float8_cls not in root:
                root[float8_cls] = {}
            return root[float8_cls]
    raise AttributeError(f"{float8_cls!r} has no _ATEN_OP_* op table")


def _patch_torchao_float8_fsdp2() -> list[str]:
    """
    Patch torchao's inference Float8Tensor for FSDP2 (fully_shard) compatibility.

    Two separate problems require two sets of patches:

    1. Init-time aten op bugs:
       FSDP2 calls torch.chunk which passes aten.split.Tensor with dim as a kwarg,
       but the existing Float8Tensor handler expects 3 positional args which raises
       a ValueError. We fix split and add new_empty/new_zeros/copy_/view handlers for
       the param-management ops FSDP2 calls during _init_sharded_param.

    2. Forward-time subclass loss:
       FSDP2 all-gather reconstructs sharded params as plain torch.Tensors, 
       stripping the Float8Tensor subclass. F.linear then sees raw fp8 bytes
       interpreted as bf16, resulting in garbage output. Fix: implement
       fsdp_pre/post_all_gather so FSDP2 gathers qdata and reconstructs a proper
       Float8Tensor on the other side. This mirrors WeightWithDynamicFloat8CastTensor
       in torchao/float8/fsdp_utils.py (the training path).

    Returns list of patch names applied (logged on first call).
    """
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

    aten = torch.ops.aten
    table = _float8_tensor_op_table(Float8Tensor)
    patched = []

    def _make_float8(tensor, new_qdata, new_block_size=None):
        _, attr_dict = tensor.__tensor_flatten__()
        if new_block_size is not None:
            attr_dict = {**attr_dict, 'block_size': new_block_size}
        return Float8Tensor.__tensor_unflatten__(
            {'qdata': new_qdata, 'scale': tensor.scale}, attr_dict, None, None
        )

    # aten.split.Tensor: torch.chunk passes dim as kwarg; existing handler
    # expects 3 positional args which raises a ValueError. Also handles PerTensor
    # scale when FSDP2 flattens weights to 1D before making them a DTensor.
    split_op = aten.split.Tensor
    _orig_split = table.get(split_op)

    def _split(func, types, args, kwargs):
        if len(args) == 2:
            args = args + (kwargs.pop("dim", 0),)
        tensor, split_size, dim = args
        if isinstance(tensor, Float8Tensor) and tensor.scale.numel() == 1:
            new_qdatas = aten.split.Tensor(tensor.qdata, split_size, dim)
            return tuple(_make_float8(tensor, qd, list(qd.shape)) for qd in new_qdatas)
        if _orig_split is not None:
            return _orig_split(func, types, args, kwargs)
        # Fall back to the same unwrap-and-dispatch behavior as _patched_dispatch
        # for anything not special-cased above (matches the pre-existing fallthrough).
        # Limitation: for a Float8Tensor with a non-scalar (block/row) scale this
        # unwraps to raw qdata and drops the Float8Tensor subclass + scale, so the
        # split result is no longer fp8-aware. That scale layout does not occur on
        # the supported static per-tensor-scale inference path (handled above);
        # revisit if block-scaled weights are ever sharded via FSDP2.
        def _unwrap(t):
            return t.qdata if isinstance(t, Float8Tensor) else t
        return func(
            *torch.utils._pytree.tree_map(_unwrap, args),
            **torch.utils._pytree.tree_map(_unwrap, kwargs),
        )

    table[split_op] = _split
    patched.append("aten.split.Tensor")

    # aten.new_empty: FSDP2 calls _chunk_with_empty which pads short chunk lists with size-0 tensors
    def _new_empty(_, _t, args, kwargs):
        tensor = args[0]
        size = list(args[1]) if len(args) > 1 else list(kwargs.get("size", []))
        new_qdata = tensor.qdata.new_empty(size, pin_memory=kwargs.get("pin_memory", False))
        return _make_float8(tensor, new_qdata, size if size else [0])
    table[aten.new_empty.default] = _new_empty
    patched.append("aten.new_empty.default")

    # aten.new_zeros: FSDP2 allocates a padded sharded-param buffer
    def _new_zeros(_, _t, args, kwargs):
        tensor = args[0]
        size = list(args[1]) if len(args) > 1 else list(kwargs.get("size", []))
        new_qdata = tensor.qdata.new_zeros(size, pin_memory=kwargs.get("pin_memory", False))
        return _make_float8(tensor, new_qdata, size)
    table[aten.new_zeros.default] = _new_zeros
    patched.append("aten.new_zeros.default")

    # aten.copy_: FSDP2 fills the padded buffer from the local fp8 shard
    def _copy_(_, _t, args, _kw):
        dst, src = args[0], args[1]
        dst.qdata.copy_(src.qdata if isinstance(src, Float8Tensor) else src)
        return dst
    table[aten.copy_.default] = _copy_
    patched.append("aten.copy_.default")

    # aten.view.default: existing handler only supports 2D↔3D; FSDP2 may call
    # view(-1) to flatten to 1D before sharding.
    _orig_view = table.get(aten.view.default)
    def _view(func, types, args, kwargs):
        tensor, size = args
        if len(size) == 1:
            numel = tensor.numel()
            return _make_float8(tensor, tensor.qdata.reshape(numel), [numel])
        if _orig_view is not None:
            return _orig_view(func, types, args, kwargs)
        raise NotImplementedError(f"Float8Tensor view patch: unhandled {tensor.shape} -> {size}")
    table[aten.view.default] = _view
    patched.append("aten.view.default")

    # aten.as_strided.default: FSDP2 uses this during init_unsharded_param to
    # create a strided view into the all-gathered buffer.
    def _as_strided(_, _t, args, kwargs):
        tensor, size, stride = args[0], args[1], args[2]
        storage_offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)
        new_qdata = aten.as_strided.default(tensor.qdata, size, stride, storage_offset)
        return _make_float8(tensor, new_qdata, list(size))
    table[aten.as_strided.default] = _as_strided
    patched.append("aten.as_strided.default")

    # __torch_dispatch__ fallthrough: for any remaining FSDP structural ops not
    # in the table (e.g. torch.ops.fsdp.*), unwrap to qdata and call through.
    # Compute ops (linear, mm, addmm) ARE in the table so they won't hit this.
    @classmethod
    def _patched_dispatch(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        inner_table = _float8_tensor_op_table(cls)
        if func in inner_table:
            return inner_table[func](func, types, args, kwargs)
        def _unwrap(t):
            return t.qdata if isinstance(t, cls) else t
        unwrapped_args = torch.utils._pytree.tree_map(_unwrap, args)
        unwrapped_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
        return func(*unwrapped_args, **unwrapped_kwargs)
    Float8Tensor.__torch_dispatch__ = _patched_dispatch
    patched.append("__torch_dispatch__ fallthrough")

    # fsdp_pre_all_gather / fsdp_post_all_gather: preserve Float8Tensor subclass
    # through FSDP2 all-gather. Without these, all-gather returns a plain Tensor,
    # F.linear bypasses fp8 dispatch and interprets e4m3fn bytes as bf16, resulting
    # in noise output. For static-scale inference Float8Tensor, scale is per-tensor
    # (scalar) and identical across all ranks, so only qdata needs to be gathered.

    def _fsdp_pre_all_gather(self, _mesh):
        _, attr_dict = self.__tensor_flatten__()
        return (self.qdata,), (self.scale, attr_dict)

    def _fsdp_post_all_gather(_self, all_gather_outputs, metadata, _param_dtype, *, out=None):
        (qdata,) = all_gather_outputs
        scale, attr_dict = metadata
        if out is not None:
            if isinstance(out, Float8Tensor):
                out.qdata.copy_(qdata)
                return
            raise RuntimeError(
                f"fsdp_post_all_gather: out must be Float8Tensor, got {type(out)}"
            )
        # attr_dict was captured before sharding, so block_size reflects the 1D
        # shard shape. Patch it to the all-gathered shape before reconstructing.
        attr_dict = {**attr_dict, 'block_size': list(qdata.shape)}
        # Return (tensor, inner_tensors): FSDP2 keeps inner_tensors alive until
        # reshard; without this second element FSDP2 unpacks the tensor itself.
        fp8 = Float8Tensor.__tensor_unflatten__(
            {'qdata': qdata, 'scale': scale}, attr_dict, None, None
        )
        return fp8, (qdata,)

    Float8Tensor.fsdp_pre_all_gather = _fsdp_pre_all_gather
    Float8Tensor.fsdp_post_all_gather = _fsdp_post_all_gather
    patched.append("fsdp_pre_all_gather")
    patched.append("fsdp_post_all_gather")

    return patched


_TORCHAO_FLOAT8_FSDP2_PATCHES: list[str] = []
_REQUIRED_TORCHAO_FLOAT8_FSDP2_PATCHES = frozenset(
    {
        "aten.split.Tensor",
        "aten.new_empty.default",
        "aten.new_zeros.default",
        "aten.copy_.default",
        "aten.view.default",
        "aten.as_strided.default",
        "__torch_dispatch__ fallthrough",
        "fsdp_pre_all_gather",
        "fsdp_post_all_gather",
    }
)

try:
    _TORCHAO_FLOAT8_FSDP2_PATCHES = _patch_torchao_float8_fsdp2()
    logger.debug("torchao Float8Tensor FSDP2 patches applied: %s", _TORCHAO_FLOAT8_FSDP2_PATCHES)
except Exception as e:
    logger.debug("torchao Float8Tensor FSDP2 patches skipped (%s): %s", type(e).__name__, e)


def torchao_float8_fsdp2_patches_available() -> tuple[bool, str | None]:
    """Report whether every tensor-subclass patch required by FSDP2 applied."""

    missing = _REQUIRED_TORCHAO_FLOAT8_FSDP2_PATCHES.difference(
        _TORCHAO_FLOAT8_FSDP2_PATCHES
    )
    if missing:
        return (
            False,
            "missing TorchAO FSDP patches: " + ", ".join(sorted(missing)),
        )
    return True, None


def quantize_linear_layers_to_int8(
    module_or_module_list: torch.nn.Module | torch.nn.ModuleList,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    device: Optional[torch.device] = None,
    min_layer_size: int = 0,
) -> None:
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quant_primitives import MappingType
    from torchao.quantization.quant_api import (
        Int8DynamicActivationInt8WeightConfig,
        quantize_,
        _is_linear,
    )

    requested_filter = filter_fn

    def filter_fn(mod, fqn):
        if not _is_linear(mod, fqn):
            return False
        if requested_filter is not None and not requested_filter(mod, fqn):
            return False
        return (
            min_layer_size <= 0
            or min(mod.out_features, mod.in_features) >= min_layer_size
        )

    config = Int8DynamicActivationInt8WeightConfig(
        granularity=PerRow(),
        act_mapping_type=MappingType.SYMMETRIC,
        set_inductor_config=False,
    )

    if isinstance(module_or_module_list, torch.nn.Module):
        module_or_module_list = [module_or_module_list]

    for module in module_or_module_list:
        quantize_(
            module,
            config=config,
            filter_fn=filter_fn,
            device=device,
        )

def quantize_linear_layers_to_fp8(module_or_module_list_to_quantize: torch.nn.Module | torch.nn.ModuleList,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
    include_suffixes: Optional[Tuple[str, ...]] = None,
    device: Optional[torch.device] = None) -> None:
    """Quantize all linear layers in the given module or module list to FP8."""
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_, _is_linear

    requested_filter = filter_fn

    def filter_fn(mod, fqn):
        if not _is_linear(mod, fqn):
            return False
        if requested_filter is not None:
            return requested_filter(mod, fqn)
        return not include_suffixes or fqn.endswith(include_suffixes)
    config = Float8DynamicActivationFloat8WeightConfig(
                granularity=PerTensor(),
                set_inductor_config=False,
                kernel_preference=_get_fp8_kernel_preference(),
                activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
        )
    if isinstance(module_or_module_list_to_quantize, torch.nn.Module):
        module_or_module_list_to_quantize = [module_or_module_list_to_quantize]
    for module in module_or_module_list_to_quantize:
        quantize_(
            module,
            config=config,
            filter_fn=filter_fn,
            device=device
        )


def quantize_linear_layers_to_fp8_blockscale(
    model: torch.nn.Module,
    parent_name: str = "",
    include_suffixes: Optional[Tuple[str, ...]] = None,
    device: Optional[torch.device] = None,
    offload_to_cpu: bool = False,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> int:
    """Replace nn.Linear layers with xFuserFP8BlockScaleLinear (AITER gemm_a8w8_blockscale).

    Mirrors quantize_linear_layers_to_fp4 structure: recursive tree walk, in-place
    setattr replacement. Pre-quantizes weights to FP8 block-128 at call time.
    Returns the number of nn.Linear leaves replaced (0 when leaves are already FP8, e.g.
    after streaming quantize-on-load).
    """
    from xfuser.model_executor.layers.fp8_linear import xFuserFP8BlockScaleLinear

    replaced = 0
    for name, module in list(model.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(module, torch.nn.Linear):
            if filter_fn is not None:
                if not filter_fn(module, full_name):
                    continue
            elif include_suffixes and not full_name.endswith(include_suffixes):
                continue
            weight = module.weight.data
            bias = module.bias.data if module.bias is not None else None
            fp8_layer = xFuserFP8BlockScaleLinear(
                module.in_features,
                module.out_features,
                bias=(bias is not None),
                device=weight.device,
                dtype=weight.dtype,
            )
            # Free BF16 weight before quantization to avoid holding two copies on GPU
            module.weight = None
            if module.bias is not None:
                module.bias = None
            fp8_layer.load_and_quantize_weights(weight, bias, device=device)
            del weight, bias
            # Under offload, quantize on GPU then evict to CPU immediately so peak VRAM
            # during quantization stays at one layer, not the whole component. The offload
            # manager streams these fp8 leaves back on demand.
            if offload_to_cpu:
                fp8_layer.to("cpu")
            setattr(model, name, fp8_layer)
            replaced += 1
        elif next(module.children(), None) is not None:
            replaced += quantize_linear_layers_to_fp8_blockscale(
                module,
                parent_name=full_name,
                include_suffixes=include_suffixes,
                device=device,
                offload_to_cpu=offload_to_cpu,
                filter_fn=filter_fn,
            )
    return replaced


def load_dataset_prompts(dataset_path: str) -> list[str]:
    """ load prompts from a csv dataset file """
    prompts = []
    with open(dataset_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts.append(row['prompt'])
    log(f"Loaded {len(prompts)} prompts from dataset at {dataset_path}")
    return prompts

def rsetattr(obj: object, attr: str, value: object) -> None:
    """ Recursive setattr to set nested attributes """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, value)

def rgetattr(obj: object, attr: str) -> object:
    """ Recursive getattr to get nested attributes """
    return functools.reduce(getattr, [obj] + attr.split("."))

def _layer_uses_fp8_override(
    layer_fqn: str,
    fp8_layers: tuple[str] | None,
    fp8_suffix_layers: tuple[str] | None,
) -> bool:
    if fp8_layers and layer_fqn.startswith(fp8_layers):
        return True
    if fp8_suffix_layers and layer_fqn.endswith(fp8_suffix_layers):
        return True
    return False


def quantize_linear_layers_to_fp4(
    model,
    parent_name='',
    fp8_layers=None,
    fp8_suffix_layers: tuple[str] | None = None,
    use_hybrid_schedule: bool = False,
    device: Optional[torch.device] = None,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
):
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig, quantize_
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear, xFuserHybridMXFP4Linear

    for name, module in list(model.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, torch.nn.Linear):
            if filter_fn is not None and not filter_fn(module, full_name):
                continue
            if _layer_uses_fp8_override(full_name, fp8_layers, fp8_suffix_layers):
                quantize_(
                      module,
                      config=Float8DynamicActivationFloat8WeightConfig(
                          granularity=PerTensor(),
                          set_inductor_config=False,
                          kernel_preference=_get_fp8_kernel_preference(),
                          activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
                    ),
                    device=device,
                )
            else:
                low_precision_layer = xFuserMXFP4Linear(
                    module.in_features,
                    module.out_features,
                    bias=(module.bias is not None),
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )

                with torch.no_grad():
                    low_precision_layer.load_and_quantize_weights(module.weight, module.bias)

                if use_hybrid_schedule:
                    high_precision_layer = torch.nn.Linear(
                        module.in_features,
                        module.out_features,
                        bias=(module.bias is not None),
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                    with torch.no_grad():
                        high_precision_layer.weight.copy_(module.weight)
                        if module.bias is not None:
                            high_precision_layer.bias.copy_(module.bias)
                    quantize_(
                        high_precision_layer,
                        config=Float8DynamicActivationFloat8WeightConfig(
                            granularity=PerTensor(),
                            set_inductor_config=False,
                            kernel_preference=_get_fp8_kernel_preference(),
                            activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
                        ),
                        device=device,
                    )
                    new_layer = xFuserHybridMXFP4Linear(
                        high_precision_linear=high_precision_layer,
                        low_precision_linear=low_precision_layer,
                    )
                else:
                    new_layer = low_precision_layer

                setattr(model, name, new_layer)

        elif len(list(module.children())) > 0:
            quantize_linear_layers_to_fp4(
                module,
                full_name,
                fp8_layers=fp8_layers,
                fp8_suffix_layers=fp8_suffix_layers,
                use_hybrid_schedule=use_hybrid_schedule,
                device=device,
                filter_fn=filter_fn,
            )


def quantize_linear_layers_to_nvfp4(
    module_or_module_list_to_quantize: torch.nn.Module | torch.nn.ModuleList,
    fp8_layers: tuple[str] = None,
    fp8_suffix_layers: tuple[str] | None = None,
    device: Optional[torch.device] = None,
    min_layer_size: int = 0,
    use_triton_kernel: bool = True,
    filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
) -> None:
    """Quantize linear layers to NVFP4 using torchao on NVIDIA Blackwell GPUs.

    Args:
        module_or_module_list_to_quantize: Module(s) whose linear layers will be quantized.
        fp8_layers: FQN prefixes of layers that should use FP8 instead of NVFP4.
        fp8_suffix_layers: FQN suffixes of layers that should use FP8 instead of NVFP4.
        device: Target device.
        min_layer_size: Skip NVFP4 for layers where min(out_features, in_features)
            is below this threshold (quantization overhead may exceed the speedup).
        use_triton_kernel: Whether to use the Triton-based NVFP4 kernel.
    """
    from torchao.prototype.mx_formats.inference_workflow import (
        NVFP4DynamicActivationNVFP4WeightConfig,
    )
    from torchao.quantization.quant_api import quantize_, _is_linear

    nvfp4_config = NVFP4DynamicActivationNVFP4WeightConfig(
        use_dynamic_per_tensor_scale=True,
        use_triton_kernel=use_triton_kernel,
    )

    if isinstance(module_or_module_list_to_quantize, torch.nn.Module):
        module_or_module_list_to_quantize = [module_or_module_list_to_quantize]

    quantized_count = 0
    skipped_fp8_count = 0
    skipped_small_count = 0

    for module in module_or_module_list_to_quantize:
        for fqn, submodule in module.named_modules():
            if not isinstance(submodule, torch.nn.Linear):
                continue
            if filter_fn is not None and not filter_fn(submodule, fqn):
                continue

            if _layer_uses_fp8_override(fqn, fp8_layers, fp8_suffix_layers):
                skipped_fp8_count += 1
                continue

            layer_min = min(submodule.out_features, submodule.in_features)
            if min_layer_size > 0 and layer_min < min_layer_size:
                skipped_small_count += 1
                continue

            quantized_count += 1

        def nvfp4_filter_fn(mod, fqn):
            if not _is_linear(mod, fqn):
                return False
            if filter_fn is not None and not filter_fn(mod, fqn):
                return False
            if _layer_uses_fp8_override(fqn, fp8_layers, fp8_suffix_layers):
                return False
            if min_layer_size > 0:
                layer_min = min(mod.out_features, mod.in_features)
                if layer_min < min_layer_size:
                    return False
            return True

        quantize_(module, config=nvfp4_config, filter_fn=nvfp4_filter_fn, device=device)

        if fp8_layers or fp8_suffix_layers:
            from torchao.quantization.granularity import PerTensor
            from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig

            fp8_config = Float8DynamicActivationFloat8WeightConfig(
                granularity=PerTensor(),
                set_inductor_config=False,
                kernel_preference=_get_fp8_kernel_preference(),
                activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
            )

            def fp8_filter_fn(mod, fqn):
                if not _is_linear(mod, fqn):
                    return False
                if filter_fn is not None and not filter_fn(mod, fqn):
                    return False
                return _layer_uses_fp8_override(fqn, fp8_layers, fp8_suffix_layers)

            quantize_(module, config=fp8_config, filter_fn=fp8_filter_fn, device=device)

    log(f"  [NVFP4] Summary: {quantized_count} layers quantized to NVFP4, "
        f"{skipped_fp8_count} overridden to FP8, {skipped_small_count} skipped (too small)")


def _tokenizer_directory(
    model_name_or_path, component_name, from_pretrained_kwargs
) -> str | None:
    """The directory holding a tokenizer, when its own fast tokenizer file is there to load.

    Reloading by the model's own name makes transformers v5 parse the *root* config.json for an
    unrelated Mistral regex fix, and it does that for every model once HF_HUB_OFFLINE is set.
    HunyuanVideo ships a root config.json with a trailing comma, which is not JSON, so the reload
    raised JSONDecodeError and the model could not load at all. Loading from the tokenizer's own
    directory leaves that file unread. None means keep the name we were given: without a
    tokenizer.json beside it the directory is not enough to build a fast tokenizer from.
    """
    import os

    from xfuser.model_executor.models.runner_models.loading.checkpoint import (
        CheckpointRequest,
        resolve_checkpoint_file,
    )

    hub_keys = {"revision", "token", "cache_dir", "local_files_only"}
    request = CheckpointRequest(
        model_name_or_path,
        subfolder=component_name,
        **{
            key: value
            for key, value in from_pretrained_kwargs.items()
            if key in hub_keys
        },
    )
    try:
        resolved = resolve_checkpoint_file(request, "tokenizer.json")
    except Exception:
        return None
    return None if resolved is None else os.path.dirname(resolved)


def fix_llama_tokenizer_pretokenizer(pipeline, model_name_or_path, **from_pretrained_kwargs) -> None:
    """
    Workaround for transformers v5 bug where LlamaTokenizer.__init__ unconditionally
    installs a SentencePiece Metaspace pre-tokenizer, silently breaking newer models
    that use ByteLevel BPE under the ``tokenizer_class="LlamaTokenizerFast"`` label.

    Reloads affected tokenizer components as ``PreTrainedTokenizerFast``, which
    respects the pre-tokenizer defined in ``tokenizer.json`` without the override.

    See https://github.com/huggingface/transformers/pull/45345
    """
    import transformers
    from xfuser.compat import version_at_least
    if not version_at_least(transformers.__version__, "5.0.0"):
        return

    from transformers import PreTrainedTokenizerFast

    for component_name, component in pipeline.components.items():
        if component is None or not component_name.startswith("tokenizer"):
            continue

        if "Llama" not in type(component).__name__:
            continue

        log(f"Replacing tokenizer '{component_name}' (type={type(component).__name__}) "
            f"with PreTrainedTokenizerFast...", debug=True)

        source, source_kwargs = model_name_or_path, {
            "subfolder": component_name,
            **from_pretrained_kwargs,
        }
        local_dir = _tokenizer_directory(
            model_name_or_path, component_name, from_pretrained_kwargs
        )
        if local_dir is not None:
            source, source_kwargs = local_dir, {}

        fixed = PreTrainedTokenizerFast.from_pretrained(source, **source_kwargs)
        setattr(pipeline, component_name, fixed)

        log(f"Fixed tokenizer '{component_name}': "
            f"reloaded as PreTrainedTokenizerFast (transformers v5 LlamaTokenizer bug workaround)")


def convert_model_convs_to_channels_last(model: torch.nn.Module) -> None:
    """
    Manually convert 2D and 3D convolutional layer weights to channels_last format.
     - Conv3d weights: (out_channels, in_channels, D, H, W) -> channels_last_3d
     - Conv2d weights: (out_channels, in_channels, H, W) -> channels_last
     - Biases and non-conv parameters are left unchanged (they are 1D and not affected by memory format)
     - This is done in-place to avoid unnecessary copying of the entire model and to ensure we only change what is needed.
    """
    for param in model.parameters():
        if param.dim() == 5:
            param.data = param.data.to(memory_format=torch.channels_last_3d)
        elif param.dim() == 4:
            param.data = param.data.to(memory_format=torch.channels_last)
