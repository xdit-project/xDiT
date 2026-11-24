import pytest
import torch

from xfuser.model_executor.layers import transposed_linear
from xfuser.model_executor.layers.transposed_linear import TransposedLinear, transpose_plain_linears
from xfuser.model_executor.models.transformers.transformers_utils import share_fused_weights


class _WrappedLinear(torch.nn.Linear):
    pass


def _model():
    """Two plain linears, one that is not plain, and a fused projection with the two linears it stacks."""
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.up = torch.nn.Linear(6, 10)
    model.down = torch.nn.Linear(10, 6, bias=False)
    model.wrapped = _WrappedLinear(6, 4)
    model.to_q, model.to_k = torch.nn.Linear(6, 4), torch.nn.Linear(6, 4)
    model.to_qk = torch.nn.Linear(6, 8)
    with torch.no_grad():
        model.to_qk.weight.copy_(torch.cat([model.to_q.weight, model.to_k.weight]))
        model.to_qk.bias.copy_(torch.cat([model.to_q.bias, model.to_k.bias]))
    share_fused_weights(model.to_qk, (model.to_q, model.to_k))
    return model


def _outputs(model):
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        return {
            name: linear(torch.randn(2, 5, linear.in_features, generator=generator))
            for name, linear in model.named_children()
        }


def _weight_bytes(model):
    """Bytes of the distinct storages behind a model's parameters."""
    storages = {
        parameter.untyped_storage().data_ptr(): parameter.untyped_storage().nbytes() for parameter in model.parameters()
    }
    return sum(storages.values())


@pytest.mark.parametrize(
    ("arch", "expected"),
    [("gfx942:sramecc+:xnack-", True), ("gfx942", True), ("gfx950:sramecc+:xnack-", False), ("", False)],
)
def test_only_gfx942_prefers_untransposed_weights(monkeypatch, arch, expected):
    monkeypatch.setattr(transposed_linear, "device_arch", lambda: arch)

    assert transposed_linear.prefers_untransposed_weights() is expected


def test_plain_linears_store_their_weight_transposed_and_compute_the_same():
    model = _model()
    expected = _outputs(model)

    assert transpose_plain_linears(model) == 3

    for name in ("up", "down", "to_qk"):
        linear = getattr(model, name)
        assert type(linear) is TransposedLinear
        assert linear.weight.shape == (linear.in_features, linear.out_features)
        assert linear.weight.is_contiguous()
    for name, output in _outputs(model).items():
        torch.testing.assert_close(output, expected[name], rtol=1e-6, atol=1e-6, msg=name)


def test_views_of_a_fused_weight_keep_their_shape_and_follow_it():
    model = _model()

    transpose_plain_linears(model)

    fused = model.to_qk.weight
    for row, view in ((0, model.to_q), (4, model.to_k)):
        assert type(view) is torch.nn.Linear
        assert view.weight.shape == (4, 6)
        assert view.weight.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
        assert torch.equal(view.weight, fused.t()[row : row + 4])


def test_transposing_costs_no_memory():
    model = _model()
    before = _weight_bytes(model)

    transpose_plain_linears(model)

    assert _weight_bytes(model) == before


def test_a_linear_that_is_not_plain_is_left_alone():
    model = _model()
    weight = model.wrapped.weight.detach().clone()

    transpose_plain_linears(model)

    assert type(model.wrapped) is _WrappedLinear
    assert torch.equal(model.wrapped.weight, weight)


def test_transposing_twice_changes_nothing():
    model = _model()
    transpose_plain_linears(model)
    expected = _outputs(model)

    assert transpose_plain_linears(model) == 0

    for name, output in _outputs(model).items():
        assert torch.equal(output, expected[name]), name
