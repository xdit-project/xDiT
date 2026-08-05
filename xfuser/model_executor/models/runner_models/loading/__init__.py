"""Load-time infrastructure shared by the runner models.

``meta_load`` builds components on meta and fills them without a full host copy; ``fp8_plan``
decides what a run quantizes on the way in. The per-model-family runners that call both live in the
parent package.

Deliberately imports nothing: the parent package's ``__init__`` imports every module it finds to
trigger the model-family registration decorators, and would otherwise pull this infrastructure in
eagerly, defeating the lazy imports its callers use to keep quantizer and loader dependencies off the
import path of runs that never need them.
"""
