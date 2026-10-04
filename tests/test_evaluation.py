"""Uniform-logit toy fixtures validate metric/timing plumbing, not LLM quality."""

import math

import pytest

mx = pytest.importorskip("mlx.core")
from paretoquant.evaluation import (  # noqa: E402
    cached_decode_benchmark,
    reference_schedule,
    text_nll,
)


class ToyModel:
    layers = [object()]

    def __call__(self, tokens, cache=None):
        return mx.zeros((*tokens.shape, 16))


class ToyTokenizer:
    def encode(self, text):
        return [1, 2, 3, 4]


def test_nll_uses_all_next_token_targets():
    result = text_nll(ToyModel(), ToyTokenizer(), ["fixture", "fixture two"])
    assert result["token_count"] == 6
    assert result["text_count"] == 2
    assert result["mean_nll"] == pytest.approx(math.log(16), rel=1e-6)
    assert result["perplexity"] == pytest.approx(16, rel=1e-6)


def test_reference_schedule_and_cached_timing():
    model = ToyModel()
    schedule = reference_schedule(model, [1, 2], steps=3)
    assert schedule == [0, 0, 0]
    result = cached_decode_benchmark({"toy": model}, [1, 2], schedule, repeats=2, warmup=0)
    assert result["toy"]["decode_steps"] == 3
    assert len(result["toy"]["decode_samples_ms"]) == 2
    assert result["toy"]["median_decode_tokens_per_second"] > 0
    assert result["toy"]["measurement"] == "teacher_forced_cached_decode_wall_clock"


@pytest.mark.parametrize("texts", [[], [""]])
def test_invalid_evaluation_texts(texts):
    with pytest.raises(ValueError):
        text_nll(ToyModel(), ToyTokenizer(), texts)


def test_empty_decode_schedule_rejected():
    with pytest.raises(ValueError):
        cached_decode_benchmark({"toy": ToyModel()}, [1], [], repeats=1)
