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

    #: (component, format) pairs already announced. Keyed by format, not by
    #: "is it FP8": a run placing two non-FP8 formats -- fp4 with an fp6 tier --
    #: shared one slot under the boolean, so whichever walk ran second was
    #: silently never logged and its tier was invisible in the run's output.
    described: set = field(default_factory=set)
    streaming_targets: set = field(default_factory=set)
    #: Streamed paths per format, for the one consumer that needs to know which
    #: format a path holds rather than merely that it holds one. Keyed by
    #: format for the same reason `described` is: two non-FP8 formats in one
    #: run shared a slot under a boolean.
    streamed_by_format: dict = field(default_factory=dict)

    def describe(self, component_name, *, format_name):
        """Record that this component's plan for one format has been logged."""

        self.described.add((component_name, format_name))

    def claim_description(self, component_name, *, format_name):
        """True when this component's `format_name` plan has not been announced yet.

        A post-load walk uses this to announce the fallback it is about to
        perform, exactly once per component and format.
        """
        if (component_name, format_name) in self.described:
            return False
        self.described.add((component_name, format_name))
        return True

    def record_streamed(self, component_name, targets, *, format_name):
        """Record the module paths that will hold quantized weights once this route finishes.

        Recorded for every format, not only the one that did the quantizing. A
        walk skips these paths because the weights are already quantized, and
        that is true whatever format the walk is placing -- the text encoder is
        quantized to FP8 by its own route and a pure FP4 run must still leave
        it alone.
        """

        paths = component_target_paths(component_name, targets)
        self.streaming_targets.update(paths)
        self.streamed_by_format.setdefault(format_name, set()).update(paths)

    def already_quantized(self):
        """The paths a walk should skip, whatever format it is placing.

        One set, with no format in the question. It used to take an ``fp8``
        flag choosing between ``streaming_targets`` and its union with the FP8
        subset -- but every FP8 path was recorded in both, so the union was
        always the first set and the flag never changed an answer.
        """

        return set(self.streaming_targets)

    def streamed_at(self, format_name):
        """The paths already holding one particular format."""

        return set(self.streamed_by_format.get(format_name, ()))
