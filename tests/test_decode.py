"""Explicit-state decoding must preserve Qwen2 logits and the live KV prefix."""

import os

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402


def tiny_model():
    mx.random.seed(42)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=128,
            max_position_embeddings=512,
            tie_word_embeddings=True,
        )
    )
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("compiled", [False, True])
def test_fixed_decode_matches_dynamic_logits_and_cache_for_every_token(compiled):
    from paretoquant.decode import FixedDecoder

    with mx.stream(mx.cpu):
        model = tiny_model()
        ids = [1, 7, 4]
        decoder = FixedDecoder(model, capacity=9, compiled=compiled)
        reference = make_prompt_cache(model)
        logits = model(mx.array([ids]), cache=reference)
        initial = decoder.prefill(ids)
        assert mx.allclose(initial, logits, atol=1e-5, rtol=1e-5).item()
        for position, token in enumerate([8, 2, 10, 9, 11, 12], start=len(ids) + 1):
            expected = model(mx.array([[token]]), cache=reference)
            actual = decoder.step(token)
            mx.eval(actual, expected, decoder.state)
            assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
            assert decoder.position == position
            for fixed, dynamic in zip(decoder.state, reference):
                keys, values, offset = fixed
                assert offset.item() == position
                assert mx.allclose(
                    keys[:, :, :position], dynamic.state[0][:, :, :position], atol=1e-6, rtol=1e-5
                ).item()
                assert mx.allclose(
                    values[:, :, :position], dynamic.state[1][:, :, :position], atol=1e-6, rtol=1e-5
                ).item()
        if compiled:
            assert decoder.trace_count == 1


@pytest.mark.parametrize("compiled", [False, True])
def test_future_storage_is_masked_even_when_it_contains_large_values(compiled):
    from paretoquant.decode import FixedDecoder

    with mx.stream(mx.cpu):
        model = tiny_model()
        decoder = FixedDecoder(model, capacity=32, compiled=compiled)
        reference = make_prompt_cache(model)
        mx.eval(model(mx.array([[1, 2]]), cache=reference), decoder.prefill([1, 2]))
        decoder.state = tuple(
            (
                mx.concatenate([k[:, :, :2], mx.full(k[:, :, 2:].shape, 1000)], axis=2),
                mx.concatenate([v[:, :, :2], mx.full(v[:, :, 2:].shape, -1000)], axis=2),
                offset,
            )
            for k, v, offset in decoder.state
        )
        actual = decoder.step(3)
        expected = model(mx.array([[3]]), cache=reference)
        assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()


def test_reset_with_different_prompt_length_reuses_graph_and_resets_offset():
    from paretoquant.decode import FixedDecoder

    with mx.stream(mx.cpu):
        model = tiny_model()
        decoder = FixedDecoder(model, capacity=32)
        for prompt in ([1, 2, 3, 4], [7], [5, 6]):
            reference = make_prompt_cache(model)
            mx.eval(
                model(mx.array([prompt]), cache=reference), decoder.prefill(prompt), decoder.state
            )
            for token in (8, 9, 10):
                actual = decoder.step(token)
                expected = model(mx.array([[token]]), cache=reference)
                assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
                assert all(offset.item() == decoder.position for _, _, offset in decoder.state)
        assert decoder.trace_count == 1


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, 513])
def test_invalid_capacity_rejected(capacity):
    from paretoquant.decode import FixedDecoder

    with pytest.raises(ValueError, match="capacity"):
        FixedDecoder(tiny_model(), capacity=capacity)


def test_empty_oversized_and_invalid_prompts_rejected():
    from paretoquant.decode import FixedDecoder

    decoder = FixedDecoder(tiny_model(), capacity=3)
    for prompt in ([], [1, 2, 3, 4], [-1], [128], [True], [1.5]):
        with pytest.raises(ValueError):
            decoder.prefill(prompt)


def test_decode_requires_prefill_and_rejects_overflow_without_changing_state():
    from paretoquant.decode import FixedDecoder

    decoder = FixedDecoder(tiny_model(), capacity=3)
    with pytest.raises(ValueError, match="prefill"):
        decoder.step(1)
    decoder.prefill([1, 2])
    for token in (-1, 128, True, 1.5):
        with pytest.raises(ValueError, match="token"):
            decoder.step(token)
    assert decoder.position == 2
    mx.eval(decoder.step(3), decoder.state)
    state = decoder.state
    with pytest.raises(ValueError, match="exhausted"):
        decoder.step(4)
    assert decoder.state is state
    assert decoder.position == 3


def test_semantic_lookalike_model_subclass_rejected():
    from paretoquant.decode import FixedDecoder

    class ChangedForward(Model):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs) + 1

    with pytest.raises(ValueError, match="exact"):
        FixedDecoder(ChangedForward(tiny_model().args), capacity=32)


def test_paired_benchmark_retains_all_cache_and_compilation_controls():
    from paretoquant.decode import benchmark_decoders

    with mx.stream(mx.cpu):
        result = benchmark_decoders(
            {"stock": tiny_model()}, [1, 2, 3], [4, 5, 6], repeats=2, warmup=1
        )
    assert set(result["timing"]) == {"stock_dynamic", "stock_fixed_eager", "stock_compiled"}
    for name, timing in result["timing"].items():
        assert len(timing["decode_samples_ms"]) == 2
        assert timing["decode_steps"] == 3
        assert timing["median_decode_tokens_per_second"] > 0
        assert timing["compiled"] == name.endswith("compiled")
        if timing["compiled"]:
            assert timing["trace_count"] == 1
    assert result["measured_trial_order"] == [
        ["stock_dynamic", "stock_fixed_eager", "stock_compiled"],
        ["stock_fixed_eager", "stock_compiled", "stock_dynamic"],
    ]
    for checked in result["numerical_checks"].values():
        assert checked["checked_decode_steps"] == 3
        assert checked["all_logits_finite"]
        assert checked["all_argmax_match"]
        assert checked["max_logit_rmse"] < 1e-5
    assert result["ratios"]["stock_compile_only"]["sample_count"] == 2


def test_compiled_greedy_generation_matches_native_and_honors_eos():
    from paretoquant.decode import FixedDecoder, greedy_token_ids
    from paretoquant.evaluation import reference_schedule

    with mx.stream(mx.cpu):
        model = tiny_model()
        expected = reference_schedule(model, [1, 2], steps=8)
        decoder = FixedDecoder(model, capacity=10)
        assert greedy_token_ids(decoder, [1, 2], max_tokens=8) == expected
        assert (
            greedy_token_ids(decoder, [1, 2], max_tokens=8, eos_tokens={expected[0]})
            == expected[:1]
        )
        assert decoder.trace_count == 1
        assert decoder.sample_trace_count == 1


def test_exclusive_decode_result_publication_preserves_competing_result(tmp_path):
    import runpy
    from pathlib import Path

    runner = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/benchmark_decode.py"),
        run_name="publication_unit",
    )
    output = tmp_path / "result.json"
    output.write_bytes(b"competing result")
    with pytest.raises(FileExistsError):
        runner["publish_result"](output, {"not": "published"})
    assert output.read_bytes() == b"competing result"


def test_generate_cli_exposes_opt_in_compilation_without_changing_default():
    import subprocess
    import sys

    from paretoquant import cli

    result = subprocess.run(
        [sys.executable, "-m", "paretoquant.cli", "generate", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--compiled" in result.stdout
    assert "opt-in" in result.stdout.lower()
    assert callable(cli.generate_local)


@pytest.mark.metal
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("PARETOQUANT_COMPILED_GPU") != "1", reason="opt-in compiled Metal check"
)
@pytest.mark.parametrize("bits", [3, 4, 6])
@pytest.mark.parametrize("fused", [False, True])
def test_quantized_compiled_qwen_native_logits_cache_and_greedy_tokens(bits, fused):
    import mlx.nn as nn

    from paretoquant.decode import FixedDecoder, greedy_token_ids
    from paretoquant.evaluation import reference_schedule
    from paretoquant.runtime import install_fusion

    if not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    model = tiny_model()
    model.set_dtype(mx.float16)
    nn.quantize(model, group_size=64, bits=bits)
    if fused:
        install_fusion(
            model,
            {
                f"model.layers.{layer}.mlp": {"backend": "fused", "bits": bits, "rows_per_group": 4}
                for layer in range(2)
            },
        )
    mx.eval(model.parameters())
    ids = [1, 2, 3]
    decoder = FixedDecoder(model, capacity=19)
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([ids]), cache=cache), decoder.prefill(ids), decoder.state)
    for token in range(4, 20):
        actual = decoder.step(token)
        expected = model(mx.array([[token]]), cache=cache)
        assert mx.allclose(actual, expected, atol=0.015, rtol=0.005).item()
        for fixed, dynamic in zip(decoder.state, cache):
            for index in (0, 1):
                assert mx.allclose(
                    fixed[index][:, :, : decoder.position],
                    dynamic.state[index][:, :, : decoder.position],
                    atol=0.003,
                    rtol=0.005,
                ).item()
    assert decoder.trace_count == 1
    expected = reference_schedule(model, ids, steps=16)
    assert greedy_token_ids(decoder, ids, max_tokens=16) == expected
    # Logit-output and sampled-token-output graphs each trace the model once.
    assert decoder.trace_count == 2
    assert decoder.sample_trace_count == 1
    assert greedy_token_ids(decoder, ids, max_tokens=16) == expected
    assert decoder.trace_count == 2
    assert decoder.sample_trace_count == 1


def test_native_prefill_policy_matches_split_prompt_logits_and_cache():
    from paretoquant.decode import FixedDecoder

    with mx.stream(mx.cpu):
        model = tiny_model()
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache), [c.state for c in cache])
        expected = model(mx.array([[4]]), cache=cache)
        decoder = FixedDecoder(model, capacity=12, native_prefill=True)
        actual = decoder.prefill([1, 2, 3, 4])
        assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-5).item()
        for fixed, dynamic in zip(decoder.state, cache):
            for index in (0, 1):
                assert mx.allclose(
                    fixed[index][:, :, :4],
                    dynamic.state[index][:, :, :4],
                    atol=1e-6,
                    rtol=1e-5,
                ).item()
        assert decoder.position == 4
