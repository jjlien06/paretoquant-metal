"""Small held-out text NLL and paired, fixed-token cached decode measurements."""

import math
import statistics
from time import perf_counter_ns

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def text_nll(model, tokenizer, texts, *, max_tokens=128):
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("evaluation requires nonempty texts")
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two")
    total = 0.0
    count = 0
    for text in texts:
        ids = tokenizer.encode(text)[:max_tokens]
        if len(ids) < 2:
            raise ValueError("evaluation text must produce at least two tokens")
        logits = model(mx.array([ids[:-1]])).astype(mx.float32)
        targets = mx.array([ids[1:]])[..., None]
        selected = mx.take_along_axis(logits, targets, axis=-1).squeeze(-1)
        nll = mx.logsumexp(logits, axis=-1) - selected
        total += mx.sum(nll).item()
        count += len(ids) - 1
    mean = total / count
    return {
        "mean_nll": mean,
        "perplexity": math.exp(mean),
        "token_count": count,
        "text_count": len(texts),
        "max_tokens_per_text": max_tokens,
        "scope": "authored_smoke_texts_not_a_standard_task_benchmark",
    }


def reference_schedule(model, prompt_ids, *, steps=32):
    if not prompt_ids or steps < 1:
        raise ValueError("nonempty prompt and positive steps required")
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids]), cache=cache)
    result = []
    for index in range(steps):
        token = mx.argmax(logits[0, -1]).item()
        result.append(token)
        if index + 1 < steps:
            logits = model(mx.array([[token]]), cache=cache)
    mx.synchronize()
    return result


def cached_decode_benchmark(models, prompt_ids, schedule, *, repeats=3, warmup=1):
    if not models or not prompt_ids or not schedule or repeats < 1 or warmup < 0:
        raise ValueError(
            "models, prompt, schedule, positive repeats and nonnegative warmup required"
        )
    names = list(models)
    samples = {name: {"prefill": [], "decode": []} for name in names}
    for trial in range(-warmup, repeats):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        for name in order:
            model = models[name]
            cache = make_prompt_cache(model)
            mx.synchronize()
            start = perf_counter_ns()
            logits = model(mx.array([prompt_ids]), cache=cache)
            mx.eval(logits)
            mx.synchronize()
            prefill = (perf_counter_ns() - start) / 1e6
            start = perf_counter_ns()
            for token in schedule:
                logits = model(mx.array([[token]]), cache=cache)
                mx.eval(logits)
            mx.synchronize()
            decode = (perf_counter_ns() - start) / 1e6
            if trial >= 0:
                samples[name]["prefill"].append(prefill)
                samples[name]["decode"].append(decode)
    return {
        name: {
            "prefill_samples_ms": values["prefill"],
            "decode_samples_ms": values["decode"],
            "median_prefill_ms": statistics.median(values["prefill"]),
            "median_decode_ms": statistics.median(values["decode"]),
            "median_decode_tokens_per_second": len(schedule)
            * 1000
            / statistics.median(values["decode"]),
            "decode_steps": len(schedule),
            "prompt_token_count": len(prompt_ids),
            "warmup": warmup,
            "repeats": repeats,
            "measurement": "teacher_forced_cached_decode_wall_clock",
            "includes_token_sampling": False,
            "stdev_decode_ms": statistics.stdev(values["decode"])
            if len(values["decode"]) > 1
            else 0.0,
        }
        for name, values in samples.items()
    }
