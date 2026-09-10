import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL.Image import Image


def _index_path(path: str, index: int) -> str:
    return f"{path}[{index}]" if path else f"[{index}]"


def _mapping_key_path(path: str, key) -> str:
    key_part = f"['{key}']" if isinstance(key, str) else f"[{key!r}]"
    return f"{path}{key_part}" if path else key_part


def _payload_family(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, np.generic):
        return "np.generic"
    if isinstance(value, (int, float, complex)):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, np.ndarray):
        return "ndarray"
    if isinstance(value, Image):
        return "pil"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if isinstance(value, (list, tuple)):
        return "sequence"
    if isinstance(value, Mapping):
        return "mapping"
    instance_dict = getattr(value, "__dict__", None)
    if isinstance(instance_dict, dict):
        return "object"
    return None


def _float_values_equal(a, b) -> bool:
    return (math.isnan(a) and math.isnan(b)) or (a == b)


def _numeric_values_equal(a, b) -> bool:
    if isinstance(a, complex) or np.issubdtype(type(a), np.complexfloating):
        return _float_values_equal(a.real, b.real) and _float_values_equal(a.imag, b.imag)
    if isinstance(a, float) or np.issubdtype(type(a), np.floating):
        return _float_values_equal(a, b)
    return a == b


def _tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    left = a.detach()
    right = b.detach()
    if left.device != right.device:
        left = left.cpu()
        right = right.cpu()
    left = left.contiguous()
    right = right.contiguous()
    if left.is_complex():
        return _tensors_equal(torch.view_as_real(left), torch.view_as_real(right))
    if left.is_floating_point():
        equal = left == right
        both_nan = torch.isnan(left) & torch.isnan(right)
        return bool(torch.all(equal | both_nan).item())
    return bool(torch.equal(left, right))


def _ndarrays_equal(a: np.ndarray, b: np.ndarray, path: str) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == object:
        for index, (left, right) in enumerate(zip(a.flat, b.flat)):
            if not _payloads_equal(left, right, _index_path(path, index)):
                return False
        return True
    return bool(np.array_equal(a, b, equal_nan=True))


def _pil_palette_data(image: Image):
    palette = image.palette
    if palette is None:
        return None
    return (palette.mode, palette.rawmode, bytes(palette.palette))


def _pil_images_equal(a: Image, b: Image) -> bool:
    if type(a) is not type(b):
        return False
    if a.size != b.size or a.mode != b.mode:
        return False
    if _pil_palette_data(a) != _pil_palette_data(b):
        return False
    return a.tobytes() == b.tobytes()


def _payloads_equal(a, b, path: str) -> bool:
    family_a = _payload_family(a)
    family_b = _payload_family(b)
    if family_a is None or family_b is None:
        unsupported = a if family_a is None else b
        raise TypeError(
            f"Unsupported payload type {type(unsupported).__name__} at {path or '<root>'}"
        )
    if family_a != family_b:
        return False
    if family_a == "none":
        return True
    if family_a in ("bool", "np.generic", "number", "str"):
        if type(a) is not type(b):
            return False
        if family_a == "str":
            return a == b
        return _numeric_values_equal(a, b)
    if family_a == "tensor":
        return _tensors_equal(a, b)
    if family_a == "ndarray":
        return _ndarrays_equal(a, b, path)
    if family_a == "pil":
        return _pil_images_equal(a, b)
    if family_a == "bytes":
        return type(a) is type(b) and a == b
    if family_a == "sequence":
        if type(a) is not type(b) or len(a) != len(b):
            return False
        for index, (left, right) in enumerate(zip(a, b)):
            if not _payloads_equal(left, right, _index_path(path, index)):
                return False
        return True
    if family_a == "mapping":
        if type(a) is not type(b):
            return False
        keys_a = list(a.keys())
        keys_b = list(b.keys())
        if keys_a != keys_b:
            return False
        for key in keys_a:
            if not _payloads_equal(a[key], b[key], _mapping_key_path(path, key)):
                return False
        return True
    if type(a) is not type(b):
        return False
    keys_a = list(a.__dict__.keys())
    keys_b = list(b.__dict__.keys())
    if keys_a != keys_b:
        return False
    for name in keys_a:
        if not _payloads_equal(
            a.__dict__[name], b.__dict__[name], f"{path}.{name}" if path else name
        ):
            return False
    return True


def outputs_equal(expected: Any, actual: Any) -> bool:
    """Compare generated payloads of two DiffusionOutput-like objects.

    Used by the determinism check in ``xFuserModel.run()`` to decide whether a
    later timed iteration matches the first-iteration snapshot.

    Assumptions:
    - Both operands are instances of ``DiffusionOutput`` or a subclass, exposed
      only as ``Any`` so this module does not import ``base_model``.
    - Operands are compared by concrete type and instance ``__dict__`` field
      names in insertion order.
    - The root ``pipe_args`` field is ignored; every other field, including
      subclass extras such as audio, is compared recursively.
    - Comparison is exact: same types, shapes, and dtypes, with IEEE float
      equality plus ``NaN == NaN`` and ``+0 == -0``. Unsupported nested types
      raise ``TypeError`` rather than falling back to ``==``.

    This is a freestanding function rather than a ``DiffusionOutput`` method
    because that class's ``get_outputs()`` pairs generated items with
    ``pipe_args`` and does not expose subclass payload fields. Determinism
    equality is not whole-object equality, so attaching it to the class would
    imply ``__eq__`` semantics that exclude ``pipe_args``. Additionally, this
    function has to operate on any subclass of ``DiffusionOutput``, so putting
    it into the class itself seems weird, though possible.
    """
    if type(expected) is not type(actual):
        return False
    expected_fields = [name for name in expected.__dict__ if name != "pipe_args"]
    actual_fields = [name for name in actual.__dict__ if name != "pipe_args"]
    if expected_fields != actual_fields:
        return False
    for name in expected_fields:
        if not _payloads_equal(expected.__dict__[name], actual.__dict__[name], name):
            return False
    return True
