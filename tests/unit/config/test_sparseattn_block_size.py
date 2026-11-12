import pytest

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs


def _parse(*argv):
    return xFuserArgs.add_runner_args(FlexibleArgumentParser()).parse_args(["--model", "m", *argv])


@pytest.mark.parametrize(("value", "tile"), [("64x64", (64, 64)), ("256X128", (256, 128))])
def test_block_size_is_read_as_query_rows_by_keys(value, tile):
    assert _parse("--sparseattn_block_size", value).sparseattn_block_size == tile


def test_block_size_defaults_to_each_recipes_own_tile():
    assert _parse().sparseattn_block_size is None


@pytest.mark.parametrize("value", ["64", "64x", "x64", "64x64x64", "sixtyfour", "0x64", "64x-64"])
def test_a_malformed_block_size_is_a_usage_error(value):
    with pytest.raises(SystemExit):
        _parse("--sparseattn_block_size", value)


def test_a_block_size_given_in_code_as_text_reaches_the_runtime_config():
    args = xFuserArgs(model="m", sparseattn_block_size="64x64")

    assert args.sparseattn_block_size == (64, 64)
