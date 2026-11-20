import pytest

from xfuser.model_executor.layers.wan_vae_sharded.layout import bands


@pytest.mark.parametrize("height,world", [(90, 8), (90, 4), (16, 2), (11, 11), (45, 7)])
def test_bands_cover_the_rows_once_and_evenly(height, world):
    split = bands(height, world)
    assert len(split) == world
    covered = [row for first, rows in split for row in range(first, first + rows)]
    assert covered == list(range(height))
    sizes = [rows for _, rows in split]
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)


def test_bands_refuse_fewer_rows_than_ranks():
    with pytest.raises(ValueError):
        bands(3, 4)
