import pytest


def pytest_collection_modifyitems(items):
    accelerator = pytest.mark.accelerator
    for item in items:
        item.add_marker(accelerator)
