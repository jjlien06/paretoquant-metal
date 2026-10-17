import mlx.core as mx
import pytest
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from test_qwen38 import tiny_config


def test_compiler_error_is_not_hidden_or_certified(monkeypatch):
    from compiler import GDNCompiler
    from phased_decode import phased_forward

    model = Model(ModelArgs.from_dict(tiny_config()))
    model.eval()
    mx.eval(model.parameters())
    cache = model.make_cache()
    phased_forward(model, mx.array([[1, 2]], mx.int32), cache, eval_group_size=4)
    compiler = GDNCompiler(model)
    events = []

    def broken_compile(*args, **kwargs):
        raise RuntimeError("sentinel compiler failure")

    monkeypatch.setattr(mx, "compile", broken_compile)
    with pytest.raises(RuntimeError, match="sentinel compiler failure"):
        phased_forward(
            model,
            mx.array([[3]], mx.int32),
            cache,
            eval_group_size=4,
            compiler=compiler,
            progress=events.append,
        )
    assert not any(e["phase"] in ("layer_done", "block_done") for e in events)
    assert compiler.compiled_calls == 0


def test_model_binding_is_explicit():
    from compiler import GDNCompiler
    from phased_decode import phased_forward

    model = Model(ModelArgs.from_dict(tiny_config()))
    other = Model(ModelArgs.from_dict(tiny_config()))
    with pytest.raises(ValueError, match="another model"):
        phased_forward(
            other, mx.array([[1]], mx.int32), other.make_cache(), compiler=GDNCompiler(model)
        )
