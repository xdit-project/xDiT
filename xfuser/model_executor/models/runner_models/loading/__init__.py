"""Load-time infrastructure shared by the runner models.

``meta_load.ModelLoader`` owns each run's contract, quantization and routes; ``quantization_plan``
resolves backend-neutral targets. The per-model-family runners live in the parent package.

Deliberately imports nothing: the parent package's ``__init__`` imports every module it finds to
trigger the model-family registration decorators, and would otherwise pull this infrastructure in
eagerly, defeating the lazy imports its callers use to keep quantizer and loader dependencies off the
import path of runs that never need them.
"""
