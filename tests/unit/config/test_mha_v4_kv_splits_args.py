import pytest

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs


def test_kv_splits_default_to_unset():
    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser())

    assert parser.parse_args(["--model", "test/model"]).mha_v4_kv_splits is None
    assert xFuserArgs().mha_v4_kv_splits is None


@pytest.mark.parametrize(("value", "expected"), [("auto", "auto"), ("1", 1), ("4", 4), (4, 4)])
def test_kv_splits_accept_auto_or_a_positive_count(value, expected):
    assert xFuserArgs(mha_v4_kv_splits=value).mha_v4_kv_splits == expected


@pytest.mark.parametrize("value", ["0", "-2", "abc", "", "AUTO", 0, True, 2.5])
def test_kv_splits_reject_anything_else(value):
    with pytest.raises(ValueError, match="--mha_v4_kv_splits"):
        xFuserArgs(mha_v4_kv_splits=value)
