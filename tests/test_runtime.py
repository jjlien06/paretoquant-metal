import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import mlx_lm.models.qwen2 as qwen2  # noqa: E402
from mlx_lm.models.qwen2 import MLP as Qwen2MLP  # noqa: E402

from paretoquant.runtime import FusedMLP, install_fusion  # noqa: E402


class ReluMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 128, bias=False)
        self.up_proj = nn.Linear(64, 128, bias=False)
        self.down_proj = nn.Linear(128, 64, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.relu(self.gate_proj(x)) * self.up_proj(x))


class GeluMLP(ReluMLP):
    def __call__(self, x):
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class OverriddenMLP(Qwen2MLP):
    def __call__(self, x):
        return self.down_proj(nn.relu(self.gate_proj(x)) * self.up_proj(x))


class OverriddenLinear(nn.Linear):
    def __call__(self, x):
        return super().__call__(x) * 2


class OverriddenQuantizedLinear(nn.QuantizedLinear):
    def __call__(self, x):
        return super().__call__(x) * 2


def mlp(bits=4):
    m = Qwen2MLP(64, 128)
    m.set_dtype(mx.float16)
    nn.quantize(m, group_size=64, bits=bits)
    return m


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj"])
@pytest.mark.parametrize("entry", ["wrapper", "stock", "fused"])
def test_overridden_quantized_projection_rejected(projection, entry):
    original = mlp()
    linear = nn.Linear(64, 128, bias=False)
    linear.set_dtype(mx.float16)
    original[projection] = OverriddenQuantizedLinear.from_linear(linear, group_size=64, bits=4)
    with pytest.raises(ValueError, match="unsupported MLP projection"):
        if entry == "wrapper":
            FusedMLP(original)
        else:
            model = nn.Module()
            model.mlp = original
            install_fusion(model, {"mlp": {"backend": entry}})


@pytest.mark.parametrize("factory", [ReluMLP, GeluMLP, lambda: OverriddenMLP(64, 128)])
def test_unverified_mlp_semantics_rejected_by_fused_wrapper(factory):
    original = factory()
    original.set_dtype(mx.float16)
    nn.quantize(original, group_size=64, bits=4)
    with pytest.raises(ValueError, match="unsupported MLP"):
        FusedMLP(original)


@pytest.mark.parametrize("factory", [ReluMLP, GeluMLP, lambda: OverriddenMLP(64, 128)])
@pytest.mark.parametrize("backend", ["stock", "fused"])
def test_installation_rejects_unverified_semantics_without_replacing_modules(factory, backend):
    model = nn.Module()
    unsupported = factory()
    unsupported.set_dtype(mx.float16)
    nn.quantize(unsupported, group_size=64, bits=4)
    supported = mlp()
    model.layers = [supported, unsupported]
    with pytest.raises(ValueError, match="unsupported MLP"):
        install_fusion(
            model,
            {"layers.0": {"backend": "fused"}, "layers.1": {"backend": backend}},
        )
    assert model.layers[0] is supported
    assert model.layers[1] is unsupported


@pytest.mark.parametrize("bits", [3, 4, 6])
@pytest.mark.parametrize("shape", [(1, 1, 64), (1, 4, 64), (2, 1, 64)])
def test_adapter_decode_and_prefill_match(bits, shape):
    m = mlp(bits)
    fused = FusedMLP(m, rows_per_group=4)
    x = mx.random.normal(shape).astype(mx.float16) * 0.1
    actual, expected = fused(x), m(x)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0.03, atol=2e-4)
    assert fused.stats["fused_calls"] == int(np.prod(shape[:-1]) == 1)
    assert fused.stats["stock_calls"] == int(np.prod(shape[:-1]) != 1)


@pytest.mark.parametrize("replace_projections", [False, True])
def test_prefill_uses_verified_original_forward_with_current_parameters(
    replace_projections, monkeypatch
):
    mx.random.seed(0)
    swiglu_calls = []
    verified_swiglu = qwen2.swiglu

    def tracked_swiglu(gate, up):
        swiglu_calls.append((gate, up))
        return verified_swiglu(gate, up)

    monkeypatch.setattr(qwen2, "swiglu", tracked_swiglu)
    original = mlp()
    fused = FusedMLP(original)
    expected_module = original
    if replace_projections:
        expected_module = mlp()
        fused.update_modules(
            {name: expected_module[name] for name in ("gate_proj", "up_proj", "down_proj")}
        )
        assert fused.gate_proj is not original.gate_proj
        assert fused.up_proj is not original.up_proj
        assert fused.down_proj is not original.down_proj
    x = mx.random.normal((1, 4, 64)).astype(mx.float16)
    actual = fused(x)
    assert len(swiglu_calls) == 1, "fallback must execute the verified original forward"
    expected = expected_module(x)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    assert fused.stats == {"fused_calls": 0, "stock_calls": 1}
    assert set(fused.children()) == {"gate_proj", "up_proj", "down_proj"}


def test_incompatible_gate_up_rejected():
    m = mlp(3)
    m.up_proj = mlp(4).up_proj
    with pytest.raises(ValueError, match="compatible"):
        FusedMLP(m)


def test_installation_uses_named_paths_and_rejects_unknown_paths():
    model = nn.Module()
    model.layers = [mlp(), mlp()]
    installed = install_fusion(
        model,
        {"layers.0": {"backend": "fused", "rows_per_group": 2}, "layers.1": {"backend": "stock"}},
    )
    assert installed == ["layers.0"]
    assert isinstance(model.layers[0], FusedMLP)
    assert not isinstance(model.layers[1], FusedMLP)
    with pytest.raises(ValueError, match="Unknown"):
        install_fusion(model, {"missing": {"backend": "fused"}})
