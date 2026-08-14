"""What the load already quantized, recorded so the post-load walks do not do it again.

Every route that can quantize a component on the way in writes here, and every post-load
conversion walk reads here. Getting it wrong is quiet in both directions: an unrecorded path is
quantized a second time, and an over-recorded one is never quantized at all. Routes take the
ledger as an argument rather than reaching for it on the model, so a route that records nothing
is a visible omission.

Two things are tracked, and they answer different questions. The described components say whose
plan has already been logged, so a walk that finds an undescribed component knows it is looking
at a fallback nobody announced. The streamed paths say which module paths already hold quantized
weights, so a walk skips them.
"""

from dataclasses import dataclass, field


def component_target_paths(component_name, targets):
    """Targets are component-relative; the walks match against full pipeline paths."""

    return {
        component_name if not target else f"{component_name}.{target}"
        for target in targets
    }


@dataclass
class QuantizationLedger:
    """The load's record of what it quantized, per component and per module path.

    Each pair is kept twice, once for FP8 and once for any format, because the FP8 walk and the
    FP4/INT8 walks ask separately and a component can be described to one and not the other.
    """

    descriptor_components: set = field(default_factory=set)
    fp8_descriptor_components: set = field(default_factory=set)
    streaming_targets: set = field(default_factory=set)
    fp8_streaming_targets: set = field(default_factory=set)

    def describe(self, component_name, *, fp8, any_format=True):
        """Record that this component's quantization plan has been logged.

        ``any_format=False`` records only the FP8 half, for the text-encoder route: a text encoder
        never appears in the FP4 or INT8 module lists the format-agnostic walks iterate.
        """
        if any_format:
            self.descriptor_components.add(component_name)
        if fp8:
            self.fp8_descriptor_components.add(component_name)

    def claim_description(self, component_name, *, fp8=False):
        """True when nothing has described this component yet, so the caller should.

        A post-load walk uses this to announce the fallback it is about to perform, exactly once.
        """
        described = (
            self.fp8_descriptor_components if fp8 else self.descriptor_components
        )
        if component_name in described:
            return False
        described.add(component_name)
        return True

    def record_streamed(self, component_name, targets, *, fp8, any_format=True):
        """Record the module paths that will hold quantized weights once this route finishes."""

        paths = component_target_paths(component_name, targets)
        if any_format:
            self.streaming_targets.update(paths)
        if fp8:
            self.fp8_streaming_targets.update(paths)

    def already_quantized(self, *, fp8=False):
        """The paths a walk should skip, as one set."""

        return (
            self.streaming_targets | self.fp8_streaming_targets
            if fp8
            else set(self.streaming_targets)
        )
