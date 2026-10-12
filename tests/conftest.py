import os


def pytest_configure():
    """Keep a GPU fault from writing a core dump into the working tree.

    The ROCm HSA runtime writes ``gpucore.<pid>.gpu`` on a GPU exception unless
    this is set before the first device call. pytest_configure runs before
    collection imports those tests.
    """
    os.environ.setdefault("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1")
