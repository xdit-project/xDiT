"""Configuration defaults that argument parsing needs.

Keep this module free of runtime imports. The argument parser loads it before
the runtime, for example to show ``--help``.
"""

DEFAULT_FP8_COMMS_SAFETY_FACTOR = 0.85
