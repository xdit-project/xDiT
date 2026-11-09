"""One registered adapter class per ``(format, implementation)`` pair.

An adapter names the concrete thing a format is stored as here, and installs
it: the layer each format finally swaps in, the config a streamed load carries,
which side of the device move it converts on. ``QuantAdapter`` in
``quant_adapter`` declares what every adapter has to answer; this module
answers it, once per pair, and registers the class under that pair.

Whether this machine can store a pair at all is ``backend_capabilities``'
question. Nothing here imports torch, AITER or torchao at module scope: each
adapter reaches for its kernels when it converts.
"""

from .contracts import AITER_FP8, TORCHAO_FP8
from .quant_adapter import LinearOwnership, QuantAdapter, stores

MXFP4_STREAMING_FALLBACK = (
    "AITER MXFP4 conversion requires each full-precision weight before creating "
    "xFuserMXFP4Linear packed state; no safe Diffusers per-weight loader exists"
)


@stores("fp8", AITER_FP8)
class AiterFp8BackendAdapter(QuantAdapter):
    # AiterFp8BlockScaleTEConfig to stream through, and the
    # xFuserFP8BlockScaleLinear layout _swap_meta_te_to_fp8 fills.
    places_text_encoder = True
    storage_semantics = "block_128_scaled"
    # A plain nn.Parameter, which is why this one needs no FSDP2 patches.
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "persistent_scale_buffer"
    serialization = "packed_state_supported_not_portable"
    converts_before_device_move = True
    loads_to_host_under_offload = True
    meta_layout_matches_storage = True
    streams_by_exclusion = False

    def _stream_config_factory(self, targets):
        from xfuser.model_executor.quant.aiter_load import stream_config

        return stream_config(list(targets))

    def transformer_stream_plan(self, targets, *, model_factory=None, residual_match=None):
        """Built from the targets themselves; this one needs no model structure."""
        targets = tuple(targets)
        return self._stream_config_factory(targets), LinearOwnership(exclusions=(), streamed=targets)

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import packed_layer_factory
        from xfuser.model_executor.layers.fp8_linear import xFuserFP8BlockScaleLinear

        return packed_layer_factory(xFuserFP8BlockScaleLinear, device)

    def convert_module(
        self,
        module,
        *,
        device,
        offload_to_cpu=False,
        filter_fn=None,
        companion=None,
    ):
        from xfuser.core.utils.runner_utils import replace_linears

        return replace_linears(
            module,
            self.layer_factory(device=device, companion=companion),
            filter_fn=filter_fn,
            offload_to_cpu=offload_to_cpu,
        )


@stores("fp8", TORCHAO_FP8)
class TorchaoFp8BackendAdapter(QuantAdapter):
    # a Transformers TorchAoConfig carrying the same
    # Float8DynamicActivationFloat8WeightConfig this stores a transformer with.
    places_text_encoder = True
    default_unavailable_reason = "Diffusers TorchAoConfig API is unavailable"
    storage_semantics = "tensorwise_dynamic"
    parameter_semantics = "torchao_float8_tensor_subclass"

    def _quant_config(self):
        return _torchao_quant_config("fp8")

    def _stream_config_factory(self, exclusions):
        return torchao_stream_config("fp8", exclusions)

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import torchao_layer_factory

        return torchao_layer_factory(self._quant_config(), device)

    def convert_module(
        self,
        module,
        *,
        device,
        offload_to_cpu=False,
        filter_fn=None,
        companion=None,
    ):
        if offload_to_cpu:
            raise ValueError("torchao FP8 conversion does not support immediate CPU offload")
        if companion is not None:
            from xfuser.core.utils.runner_utils import replace_linears

            return replace_linears(
                module,
                self.layer_factory(device=device, companion=companion),
                filter_fn=filter_fn,
            )
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp8

        return quantize_linear_layers_to_fp8(module, device=device, filter_fn=filter_fn)


@stores("fp4", "torchao")
class TorchaoNvfp4BackendAdapter(QuantAdapter):
    default_unavailable_reason = "Diffusers TorchAoConfig API is unavailable"
    storage_semantics = "torchao_nvfp4_dynamic_per_tensor"
    parameter_semantics = "torchao_nvfp4_tensor_subclass"
    supports_precision_overrides = True

    def _stream_config_factory(self, exclusions):
        return torchao_stream_config("nvfp4", exclusions)

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import torchao_layer_factory

        return torchao_layer_factory(_torchao_quant_config("nvfp4"), device)

    def convert_module(self, module, *, device, filter_fn=None, offload_to_cpu=False, companion=None):
        if companion is not None:
            from xfuser.core.utils.runner_utils import replace_linears

            return replace_linears(
                module,
                self.layer_factory(device=device, companion=companion),
                filter_fn=filter_fn,
                offload_to_cpu=offload_to_cpu,
            )
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_nvfp4

        return quantize_linear_layers_to_nvfp4(module, device=device, filter_fn=filter_fn)


@stores("fp4", "aiter")
class AiterMxfp4BackendAdapter(QuantAdapter):
    meta_layout_matches_storage = True
    streams_by_exclusion = False
    # Group offloading moves a module's parameters between host and device
    # around each call, and these weights survive neither leg. Both failures
    # land below Python -- one inside the hook, one as AITER's own abort with
    # no traceback -- so neither can be caught where it happens.
    group_offload_refusal = {
        "pinned": "torch cannot pin a Float4_e2m1fn_x2 tensor",
        "unpinned": (
            "AITER binds a device from the parameter it is given, and a host parameter resolves to an invalid ordinal"
        ),
    }
    storage_semantics = "aiter_mxfp4_per_1x32"
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "replicated_scale_buffer"
    serialization = "packed_state_supported_not_portable"
    supports_precision_overrides = True

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import mxfp4_layer_factory

        return mxfp4_layer_factory(device)

    def convert_module(self, module, *, device, filter_fn=None, offload_to_cpu=False, companion=None):
        from xfuser.core.utils.runner_utils import replace_linears

        return replace_linears(
            module,
            self.layer_factory(device=device, companion=companion),
            filter_fn=filter_fn,
            offload_to_cpu=offload_to_cpu,
        )


@stores("fp6", "aiter")
class AiterMxfp6BackendAdapter(QuantAdapter):
    meta_layout_matches_storage = True
    streams_by_exclusion = False
    storage_semantics = "aiter_mxfp6_e2m3_per_1x32"
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "persistent_scale_buffer"
    serialization = "packed_state_supported_not_portable"

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import packed_layer_factory
        from xfuser.model_executor.layers.mxfp6_linear import xFuserMXFP6Linear

        return packed_layer_factory(xFuserMXFP6Linear, device)

    def convert_module(
        self,
        module,
        *,
        device,
        filter_fn=None,
        offload_to_cpu=False,
        companion=None,
        **kwargs,
    ):
        from xfuser.core.utils.runner_utils import replace_linears

        return replace_linears(
            module,
            self.layer_factory(device=device, companion=companion),
            filter_fn=filter_fn,
            offload_to_cpu=offload_to_cpu,
        )


@stores("a6w4", "aiter")
class AiterA6W4BackendAdapter(QuantAdapter):
    meta_layout_matches_storage = True
    streams_by_exclusion = False
    # Deliberately takes no ``companion``: the kernel computes one leaf's FP4
    # weight against its own FP6 activation and has no seam for a second
    # layer, so ``leads_a_pair`` reads False off this signature and a per-step
    # schedule asking A6W4 to lead is refused where every other pair is.
    fills_blocks_locally = False
    # Neither leg survives the hook: packing reads the source weight on the
    # device it will compute on, and a host-resident parameter has nothing to
    # pack from. Pinning is not what fails, so both legs carry one reason.
    group_offload_refusal = {
        "pinned": "mixed-MXFP packing requires GPU-resident source weights",
        "unpinned": "mixed-MXFP packing requires GPU-resident source weights",
    }
    packs_from_device_resident_source = True
    storage_semantics = "aiter_a6w4_per_1x32"
    parameter_semantics = "packed_weight_parameter"
    auxiliary_state_semantics = "persistent_scale_buffer"
    serialization = "packed_state_supported_not_portable"

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import packed_layer_factory
        from xfuser.model_executor.layers.mixed_mxfp_linear import xFuserA6W4Linear

        return packed_layer_factory(xFuserA6W4Linear, device)

    def convert_module(self, module, *, device, filter_fn=None, offload_to_cpu=False, **kwargs):
        from xfuser.core.utils.runner_utils import replace_linears

        return replace_linears(
            module,
            self.layer_factory(device=device),
            filter_fn=filter_fn,
            offload_to_cpu=offload_to_cpu,
        )


@stores("int8", "torchao")
class TorchaoInt8BackendAdapter(QuantAdapter):
    default_unavailable_reason = "Diffusers TorchAoConfig API is unavailable"
    storage_semantics = "torchao_w8a8_dynamic_per_row_symmetric"
    parameter_semantics = "torchao_int8_tensor_subclass"
    min_layer_size = 512

    def _stream_config_factory(self, exclusions):
        return torchao_stream_config("int8", exclusions)

    def _single_layer_factory(self, *, device):
        from xfuser.core.utils.runner_utils import torchao_layer_factory

        return torchao_layer_factory(_torchao_quant_config("int8"), device)

    def convert_module(self, module, *, device, filter_fn=None, companion=None, **kwargs):
        if companion is not None:
            from xfuser.core.utils.runner_utils import replace_linears

            return replace_linears(
                module,
                self.layer_factory(device=device, companion=companion),
                filter_fn=filter_fn,
            )
        from xfuser.core.utils.runner_utils import quantize_linear_layers_to_int8

        return quantize_linear_layers_to_int8(
            module,
            device=device,
            min_layer_size=self.min_layer_size,
            filter_fn=filter_fn,
        )


def torchao_stream_config(kind: str, exclusions):
    """The Diffusers wrapper around the quant config, for a module-wide load.

    Public because the Diffusers API probe builds one too: what it measures is
    whether Diffusers accepts the config a streamed load would carry, so it
    has to ask for that config rather than for a stand-in.
    """

    from diffusers import TorchAoConfig

    return TorchAoConfig(_torchao_quant_config(kind), modules_to_not_convert=list(exclusions))


def _torchao_quant_config(kind: str):
    """The torchao config itself, which `quantize_` takes for one leaf or a tree.

    Split out so a single leaf can be built with exactly the settings a
    module-wide load would have used -- that is what gives each torchao format
    a per-leaf seam, and with it a place in a per-step pair.
    """

    from xfuser.model_executor.quant.torchao_quantizer import (
        register_torchao_fp32_policy,
    )

    register_torchao_fp32_policy()
    if kind == "fp8":
        from torchao.quantization.granularity import PerTensor
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
        )
        from xfuser.core.utils.runner_utils import (
            FP8_ACTIVATION_SCALE_FLOOR,
            _get_fp8_kernel_preference,
        )

        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            set_inductor_config=False,
            kernel_preference=_get_fp8_kernel_preference(),
            activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
        )
    elif kind == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )

        config = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        )
    else:
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quant_primitives import MappingType
        from torchao.quantization.quant_api import (
            Int8DynamicActivationInt8WeightConfig,
        )

        config = Int8DynamicActivationInt8WeightConfig(
            granularity=PerRow(),
            act_mapping_type=MappingType.SYMMETRIC,
            set_inductor_config=False,
        )
    return config
