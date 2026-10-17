import mlx.core as mx
import pytest
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from test_qwen38 import assert_caches, tiny_config


def test_compile_cli_admission_sources_and_receipts():
    import argparse
    from pathlib import Path

    import probe
    import supervise
    from remote_actor import SOURCES, source_hashes

    args = supervise.parser().parse_args(
        [
            "--remote-root",
            "/remote/spikes/014-qwen38-compiled-gdn",
            "--remote-source",
            "/model",
            "--local-metadata",
            "/metadata",
            "--output",
            "/output",
            "--ssh-key",
            "/key",
            "--split",
            "4",
            "5",
            "--compile-gdn",
            "--eval-group-size",
            "4",
        ]
    )
    assert "--compile-gdn" in supervise.probe_options(args)
    assert "compiler.py" in SOURCES
    assert set(source_hashes(Path(__file__).parent)) == set(SOURCES)
    assert probe.phased_options(
        argparse.Namespace(phased=True, eval_group_size=4, compile_gdn=True)
    ) == {"eval_group_size": 4}
    args.execution = "upstream"
    with pytest.raises(ValueError, match="compile"):
        supervise.validate_args(args)
    with pytest.raises(ValueError, match="compile"):
        probe.phased_options(argparse.Namespace(phased=False, eval_group_size=1, compile_gdn=True))
    from supervise import validate_compile_receipt

    validate_compile_receipt({"compile_gdn": False, "gdn_compiler": None}, False)
    validate_compile_receipt(
        {
            "compile_gdn": True,
            "gdn_compiler": {"trace_count": 3, "compiled_calls": 6, "compiled_functions": 3},
        },
        True,
    )
    for record in [
        {},
        {"compile_gdn": False},
        {
            "compile_gdn": True,
            "gdn_compiler": {"trace_count": 0, "compiled_calls": 0, "compiled_functions": 0},
        },
    ]:
        with pytest.raises(ValueError):
            validate_compile_receipt(record, True)


def test_compiler_fallback_and_group_one():
    from compiler import GDNCompiler
    from phased_decode import phased_forward, phased_generate

    config = tiny_config()
    model = Model(ModelArgs.from_dict(config))
    model.eval()
    mx.eval(model.parameters())
    compiler = GDNCompiler(model)
    left, right = model.make_cache(), model.make_cache()
    x = mx.array([[1]], mx.int32)
    expected = phased_forward(model, x, left, eval_group_size=1)
    actual = phased_forward(model, x, right, eval_group_size=1, compiler=compiler)
    assert compiler.trace_count == 0
    assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()
    assert_caches(left, right)
    actual = phased_forward(model, x, right, eval_group_size=1, compiler=compiler)
    expected = phased_forward(model, x, left, eval_group_size=1)
    assert compiler.trace_count == 3
    assert mx.allclose(expected, actual, atol=1e-6, rtol=1e-5).item()
    assert_caches(left, right)
    h = model.model.embed_tokens(x)
    layer, cache = model.model.layers[0], right[0]
    for field in ["lengths", "left_padding"]:
        setattr(cache, field, mx.array([1]))
        assert not compiler.eligible(h, [layer], [cache], None)
        setattr(cache, field, None)
    assert not compiler.eligible(h, [layer], [cache], mx.ones((1, 1), mx.bool_))

    class Tokenizer:
        eos_token_ids = []

        def decode(self, ids, **kwargs):
            return str(ids)

    native = phased_generate(
        model, Tokenizer(), [1, 2, 3, 4, 5], max_tokens=4, prefill_step_size=2, eval_group_size=4
    )
    for prompt in [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]:
        compiled = phased_generate(
            model,
            Tokenizer(),
            prompt,
            max_tokens=4,
            prefill_step_size=2,
            eval_group_size=4,
            compiler=compiler,
        )
        assert compiled["token_ids"] == native["token_ids"]
        assert compiled["compile_gdn"] is True
