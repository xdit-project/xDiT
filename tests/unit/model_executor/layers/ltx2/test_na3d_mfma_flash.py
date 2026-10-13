import sys

import pytest

from xfuser.model_executor.layers.ltx2.na3d_mfma_flash import LTX2VideoVaeMfmaAttnProcessor


def test_construction_raises_import_error_without_aiter_na3d_flash(monkeypatch):
    # The LTX-2.5 runner falls back to tiled SDPA on ImportError, so an AITER
    # build without na3d_flash must fail at construction, not on first decode.
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.attention.na3d_flash", None)

    with pytest.raises(ImportError):
        LTX2VideoVaeMfmaAttnProcessor()
