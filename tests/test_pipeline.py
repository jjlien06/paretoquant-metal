"""Synthetic small-model fixtures test plumbing, not LLM quality claims."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.qwen2 import MLP as Qwen2MLP  # noqa: E402
from test_runtime import GeluMLP, OverriddenLinear, OverriddenMLP, ReluMLP  # noqa: E402

import paretoquant.pipeline as pipeline  # noqa: E402
from paretoquant.allocator import Option  # noqa: E402
from paretoquant.pipeline import (  # noqa: E402
    apply_plan,
    calibrate_inputs,
    model_bytes,
    profile_units,
)
from paretoquant.runtime import FusedMLP  # noqa: E402


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Qwen2MLP(64, 128)

    def __call__(self, x):
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 64)
        self.layers = [Layer(), Layer()]
        self.head = nn.Linear(64, 16, bias=False)
        self.set_dtype(mx.float16)

    def __call__(self, tokens):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class TokenizerFixture:
    def encode(self, text):
        return [1, 2, 3, 4]


def mlp_with_overridden_projection(projection):
    module = Qwen2MLP(64, 128)
    module[projection] = OverriddenLinear(64, 128, bias=False)
    return module


@pytest.mark.parametrize(
    "factory",
    [
        ReluMLP,
        GeluMLP,
        lambda: OverriddenMLP(64, 128),
        lambda: mlp_with_overridden_projection("gate_proj"),
        lambda: mlp_with_overridden_projection("up_proj"),
    ],
)
@pytest.mark.parametrize("entry", ["calibrate", "profile", "apply"])
def test_unverified_semantics_rejected_before_pipeline_work(factory, entry, monkeypatch):
    model = TinyModel()
    unsupported = factory()
    unsupported.set_dtype(mx.float16)
    model.layers[1].mlp = unsupported
    originals = [layer.mlp for layer in model.layers]
    weights = [module.gate_proj.weight for module in originals]

    def must_not_start(*args, **kwargs):
        pytest.fail("pipeline work started before rejecting unsupported MLP semantics")

    with pytest.raises(ValueError, match="unsupported MLP"):
        if entry == "calibrate":
            tokenizer = TokenizerFixture()
            monkeypatch.setattr(tokenizer, "encode", must_not_start)
            calibrate_inputs(model, tokenizer, ["fixture"])
        elif entry == "profile":
            monkeypatch.setattr(pipeline, "environment", must_not_start)
            inputs = {f"layers.{i}.mlp": mx.ones((2, 64), mx.float16) for i in range(2)}
            profile_units(model, inputs, repeats=2, warmup=0, verbose=False)
        else:
            monkeypatch.setattr(pipeline, "quantize_model", must_not_start)
            choices = {
                f"layers.{i}.mlp": Option("q4:stock", 4, 1, 1.0, 0.0, "stock")
                for i in range(2)
            }
            apply_plan(model, {"model_type": "fixture"}, choices)
    for layer, original, weight in zip(model.layers, originals, weights):
        assert layer.mlp is original
        assert layer.mlp.gate_proj.weight is weight


def test_calibration_collects_actual_layer_inputs_and_restores_modules():
    model = TinyModel()
    originals = [layer.mlp for layer in model.layers]
    inputs = calibrate_inputs(model, TokenizerFixture(), ["fixture"], samples_per_text=3)
    assert set(inputs) == {"layers.0.mlp", "layers.1.mlp"}
    assert inputs["layers.0.mlp"].shape == (3, 64)
    assert all(layer.mlp is original for layer, original in zip(model.layers, originals))


def test_calibration_restores_supported_modules_after_forward_failure():
    model = TinyModel()
    originals = [layer.mlp for layer in model.layers]
    head = model.head
    tokens = mx.array([[1, 2, 3, 4]])
    expected = model(tokens)
    mx.eval(expected)

    class FailingHead(nn.Module):
        def __call__(self, x):
            raise RuntimeError("fixture forward failure")

    model.head = FailingHead()
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        calibrate_inputs(model, TokenizerFixture(), ["fixture"])
    assert all(layer.mlp is original for layer, original in zip(model.layers, originals))
    model.head = head
    actual = model(tokens)
    mx.eval(actual)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_profile_contains_actual_bytes_errors_and_measurements():
    model = TinyModel()
    inputs = calibrate_inputs(model, TokenizerFixture(), ["fixture"])
    profile = profile_units(model, inputs, repeats=2, warmup=0, verbose=False)
    assert len(profile["units"]) == 2
    for unit in profile["units"]:
        assert {opt["bits"] for opt in unit["options"]} == {3, 4, 6}
        assert all(
            opt["memory_bytes"] > 0 and opt["latency_ms"] > 0 and opt["loss"] >= 0
            for opt in unit["options"]
        )
        for bit in (3, 4, 6):
            stock = next(o for o in unit["options"] if o["bits"] == bit and o["backend"] == "stock")
            assert stock["memory_bytes"] == 2 * (128 * 64 * bit // 8 + 128 * 2 * 2)
        assert unit["measurements"]


def test_precision_and_fused_dispatch_survive_saved_weights_layout():
    model = TinyModel()
    choices = {
        "layers.0.mlp": Option("q3:fused:rpg4", 3, 1, 1.0, 0.0, "fused"),
        "layers.1.mlp": Option("q6:stock", 6, 1, 1.0, 0.0, "stock"),
    }
    model, config, dispatch = apply_plan(model, {"model_type": "fixture"}, choices)
    assert isinstance(model.layers[0].mlp, FusedMLP)
    assert model.layers[0].mlp.gate_proj.bits == model.layers[0].mlp.up_proj.bits == 3
    assert model.layers[1].mlp.gate_proj.bits == model.layers[1].mlp.up_proj.bits == 6
    assert model.layers[1].mlp.down_proj.bits == 4
    assert config["quantization"]["layers.0.mlp.gate_proj"]["bits"] == 3
    assert dispatch["layers.0.mlp"]["backend"] == "fused"
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert np.isfinite(np.array(out)).all()
    assert model_bytes(model) > 0


@pytest.mark.parametrize("texts", [[], [""]])
def test_empty_calibration_rejected(texts):
    with pytest.raises(ValueError):
        calibrate_inputs(TinyModel(), TokenizerFixture(), texts)
