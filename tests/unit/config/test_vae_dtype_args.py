import pytest
import torch

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs


def _from_cli(*argv):
    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser())
    return xFuserArgs.from_cli_args(parser.parse_args(["--model", "m", *argv]))


@pytest.mark.parametrize(("value", "dtype"), [("bf16", torch.bfloat16), ("fp32", torch.float32)])
def test_the_vae_dtype_reaches_the_model_as_a_torch_dtype(value, dtype):
    assert _from_cli("--vae_dtype", value).vae_dtype is dtype


def test_without_the_flag_each_model_keeps_its_own_vae_dtype():
    assert _from_cli().vae_dtype is None


def test_an_unknown_vae_dtype_is_a_usage_error():
    with pytest.raises(SystemExit):
        _from_cli("--vae_dtype", "fp16")


def test_a_vae_dtype_given_in_code_as_text_is_converted_or_refused():
    assert xFuserArgs(model="m", vae_dtype="fp32").vae_dtype is torch.float32
    assert xFuserArgs(model="m", vae_dtype=torch.bfloat16).vae_dtype is torch.bfloat16
    with pytest.raises(ValueError, match="--vae_dtype must be one of bf16, fp32"):
        xFuserArgs(model="m", vae_dtype="fp16")
