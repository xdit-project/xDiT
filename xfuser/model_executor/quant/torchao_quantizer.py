"""Keep diffusers' ``_keep_in_fp32_modules`` policy on for TorchAO streaming loads.

Diffusers turns that policy off whenever a quantizer is active unless the quantizer opts back in by
declaring ``use_keep_in_fp32_modules``. Its bitsandbytes, quanto, gguf and modelopt quantizers all
opt in; TorchAO does not, even though it only ever converts ``nn.Linear`` weights. So streaming a
transformer through ``TorchAoConfig`` loads the modules a model pins to fp32 in the requested compute
dtype instead — for Wan2.2 that is every ``norm2``, every ``scale_shift_table`` and the time
embedder, 125 tensors per transformer.

The post-load quantization walk in ``runner_utils`` never had this problem: it quantizes after an
ordinary ``from_pretrained``, so no quantizer is in scope while the weights load and the policy
stays on. Streaming would otherwise load different weights than the walk for the same checkpoint and
flags. Subclass the quantizer to opt in so both routes agree.
"""

import functools


@functools.lru_cache(maxsize=1)
def register_torchao_fp32_policy() -> None:
    """Point diffusers' torchao quantizer entry at a subclass that keeps fp32 modules.

    Registered under diffusers' own ``quant_method`` key so an unmodified ``TorchAoConfig`` still
    routes here and every ``quant_method == "torchao"`` check elsewhere in diffusers still holds.
    Called when a streaming config is built, so a run that never streams TorchAO leaves the
    global mapping untouched. A no-op once diffusers opts in upstream.
    """

    from diffusers.quantizers.auto import AUTO_QUANTIZER_MAPPING
    from diffusers.quantizers.quantization_config import QuantizationMethod

    key = QuantizationMethod.TORCHAO
    installed = AUTO_QUANTIZER_MAPPING.get(key)
    if installed is None or getattr(installed, "use_keep_in_fp32_modules", False):
        return

    class xFuserTorchAoHfQuantizer(installed):
        """TorchAO quantizer that honours the model's ``_keep_in_fp32_modules``."""

        use_keep_in_fp32_modules = True

    AUTO_QUANTIZER_MAPPING[key] = xFuserTorchAoHfQuantizer
