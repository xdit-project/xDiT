import warnings

import pytest

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs


def _from_cli(*argv):
    parsed = xFuserArgs.add_runner_args(FlexibleArgumentParser()).parse_args(["--model", "m", *argv])
    return xFuserArgs.from_cli_args(parsed)


def test_head_balancing_is_off_by_default():
    assert not _from_cli().use_sparseattn_head_balance


def test_the_flag_turns_head_balancing_on_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert _from_cli("--use_sparseattn_head_balance").use_sparseattn_head_balance


def test_the_sparge_spelling_still_works_but_says_it_is_deprecated():
    """Head balancing serves Sol-Attn as well as Sparge now, so the flag lost the Sparge name.
    Scripts written against the old one keep running."""
    with pytest.warns(FutureWarning, match="use --use_sparseattn_head_balance"):
        args = _from_cli("--use_spargeattn_head_balance")

    assert args.use_sparseattn_head_balance


def test_the_deprecated_field_works_when_set_in_code():
    with pytest.warns(FutureWarning, match="deprecated"):
        args = xFuserArgs(model="m", use_spargeattn_head_balance=True)

    assert args.use_sparseattn_head_balance
