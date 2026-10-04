"""Single-sequence Qwen2 decode with array-valued, fixed-shape KV state.

Prefill stays eager/native. Only one-token decoding is compiled. This is opt-in:
fixed-capacity attention can be slower, and compilation is not a speed guarantee.
"""

import statistics
from time import perf_counter_ns

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.qwen2 import Model

from .statistics import paired_latency_ratio


class _FixedCache:
    """Transient trace-local adapter; mutable Python objects never cross a trace."""

    def __init__(self, state):
        self.keys, self.values, self.offset = state

    def make_mask(self, length, *, return_array=False, window_size=None):
        if length != 1 or window_size is not None:
            raise ValueError("fixed-cache decoding supports one token and no sliding window")
        # Mask future storage including after the newly appended token. Attention
        # receives full-capacity arrays; unused zeros must never receive probability.
        return mx.arange(self.keys.shape[2]) <= self.offset

    def update_and_fetch(self, keys, values):
        indices = self.offset.reshape(1)
        self.keys = mx.slice_update(self.keys, keys, indices, axes=(2,))
        self.values = mx.slice_update(self.values, values, indices, axes=(2,))
        self.offset = self.offset + 1
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values, self.offset


class FixedDecoder:
    """Reusable compiled graph; reset native prefill state between requests/trials.

    Do not mutate the model after constructing this object: its weights/modules
    are captured in the compiled closure. The caller evaluates logits and state.
    ``trace_count`` counts Python tracing, NOT GPU kernel invocations.
    """

    def __init__(self, model, *, capacity, compiled=True):
        if type(model) is not Model or model.model.pipeline_size != 1:
            raise ValueError("fixed decoding requires exact unsharded Qwen2 Model")
        if type(capacity) is not int or not 1 <= capacity <= model.args.max_position_embeddings:
            raise ValueError("capacity must be a positive integer within model context")
        if type(compiled) is not bool:
            raise ValueError("compiled must be a boolean")
        self.model = model
        self.capacity = capacity
        self.compiled = compiled
        self.position = 0
        self.state = None
        self.trace_count = 0

        def forward(token, state):
            self.trace_count += 1
            caches = [_FixedCache(layer_state) for layer_state in state]
            logits = model(token, cache=caches)
            return logits, tuple(cache.state for cache in caches)

        self._forward = mx.compile(forward) if compiled else forward

    def prefill(self, prompt_ids):
        ids = list(prompt_ids)
        if not ids or len(ids) > self.capacity:
            raise ValueError("prompt must be nonempty and fit cache capacity")
        if any(
            type(token) is not int or not 0 <= token < self.model.args.vocab_size for token in ids
        ):
            raise ValueError("prompt token IDs must be integers within vocabulary")
        caches = make_prompt_cache(self.model)
        if any(type(cache) is not KVCache for cache in caches):
            raise ValueError("fixed decoding requires native nonquantized KVCache")
        logits = self.model(mx.array([ids]), cache=caches)
        state = []
        for cache in caches:
            keys, values, offset = cache.state
            padding = self.capacity - offset
            state.append(
                (
                    mx.pad(keys[:, :, :offset], ((0, 0), (0, 0), (0, padding), (0, 0))),
                    mx.pad(values[:, :, :offset], ((0, 0), (0, 0), (0, padding), (0, 0))),
                    mx.array(offset, dtype=mx.int32),
                )
            )
        self.state = tuple(state)
        self.position = len(ids)
        return logits

    def step(self, token):
        if self.state is None:
            raise ValueError("prefill required before decoding")
        if self.position >= self.capacity:
            raise ValueError("fixed cache capacity exhausted; re-prefill with a larger capacity")
        if type(token) is not int or not 0 <= token < self.model.args.vocab_size:
            raise ValueError("decode token ID must be an integer within vocabulary")
        logits, self.state = self._forward(mx.array([[token]]), self.state)
        self.position += 1
        return logits


class _DynamicDecoder:
    def __init__(self, model):
        self.model = model
        self.state = None

    def prefill(self, ids):
        self.state = make_prompt_cache(self.model)
        return self.model(mx.array([ids]), cache=self.state)

    def step(self, token):
        return self.model(mx.array([[token]]), cache=self.state)

    def eval_state(self):
        return tuple(cache.state[:2] for cache in self.state)


def _arrays(decoder):
    return decoder.eval_state() if isinstance(decoder, _DynamicDecoder) else decoder.state


def greedy_token_ids(decoder, prompt_ids, *, max_tokens, eos_tokens=()):
    """Actual autoregressive greedy decoding, separate from teacher-forced timing."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    eos = set(eos_tokens)
    if any(type(t) is not int or not 0 <= t < decoder.model.args.vocab_size for t in eos):
        raise ValueError("EOS IDs must be integers within vocabulary")
    logits = decoder.prefill(prompt_ids)
    mx.eval(logits, _arrays(decoder))
    tokens = []
    for index in range(max_tokens):
        token = mx.argmax(logits[0, -1]).item()
        tokens.append(token)
        if token in eos or index + 1 == max_tokens:
            break
        logits = decoder.step(token)
        mx.eval(logits, _arrays(decoder))
    return tokens


def _check_candidates(candidates, prompt_ids, schedule):
    """Compare each cache/compile variant against its own backend's native cache.

    No claim of bit-identical stock versus custom Metal arithmetic. Thresholds
    are fixed before timing, not selected to fit a particular run.
    """
    checks = {}
    for prefix in sorted(
        {name.rsplit("_", 1)[0] for name in candidates if name.endswith("_dynamic")}
    ):
        reference = candidates[f"{prefix}_dynamic"]
        others = {name: d for name, d in candidates.items() if name.startswith(prefix + "_")}
        for decoder in others.values():
            mx.eval(decoder.prefill(prompt_ids), _arrays(decoder))
        local = {
            name: {
                "reference": f"{prefix}_dynamic",
                "checked_decode_steps": 0,
                "all_logits_finite": True,
                "all_argmax_match": True,
                "max_absolute_logit_error": 0.0,
                "max_logit_rmse": 0.0,
                "max_absolute_tolerance": 0.0625,
                "rmse_tolerance": 0.005,
            }
            for name in others
            if name != f"{prefix}_dynamic"
        }
        for token in schedule:
            expected = reference.step(token).astype(mx.float32)
            mx.eval(expected, _arrays(reference))
            expected_finite = bool(mx.all(mx.isfinite(expected)).item())
            expected_argmax = mx.argmax(expected[0, -1]).item()
            for name, check in local.items():
                decoder = others[name]
                actual = decoder.step(token).astype(mx.float32)
                delta = actual - expected
                finite = expected_finite and bool(mx.all(mx.isfinite(actual)).item())
                absolute = mx.max(mx.abs(delta)).item()
                rmse = mx.sqrt(mx.mean(delta**2)).item()
                argmax = mx.argmax(actual[0, -1]).item() == expected_argmax
                mx.eval(_arrays(decoder))
                check["checked_decode_steps"] += 1
                check["all_logits_finite"] &= finite
                check["all_argmax_match"] &= argmax
                check["max_absolute_logit_error"] = max(check["max_absolute_logit_error"], absolute)
                check["max_logit_rmse"] = max(check["max_logit_rmse"], rmse)
                if not finite or not argmax or absolute > 0.0625 or rmse > 0.005:
                    raise ValueError(f"decode numerical admission failed for {name}: {check}")
        checks.update(local)
    return checks


def benchmark_decoders(models, prompt_ids, schedule, *, repeats=24, warmup=2):
    """Paired, rotated controls separate fusion, cache policy, and compilation.

    Evaluate all updated cache arrays as well as logits in every variant. Exclude
    prefill, padding, first compilation and correctness checks from decode time;
    retain setup timing separately. Sampling is not included (teacher forcing).
    """
    if not models or not prompt_ids or not schedule:
        raise ValueError("models, prompt and schedule must be nonempty")
    if type(repeats) is not int or repeats < 2 or type(warmup) is not int or warmup < 1:
        raise ValueError("at least two repeats and one warmup are required")
    for model in models.values():
        if any(type(t) is not int or not 0 <= t < model.args.vocab_size for t in schedule):
            raise ValueError("schedule token IDs must be integers within vocabulary")
    capacity = len(prompt_ids) + len(schedule)
    candidates = {}
    for name, model in models.items():
        if not isinstance(name, str) or not name or "_" in name:
            raise ValueError("model names must be nonempty strings without underscores")
        candidates[name + "_dynamic"] = _DynamicDecoder(model)
        candidates[name + "_fixed_eager"] = FixedDecoder(model, capacity=capacity, compiled=False)
        candidates[name + "_compiled"] = FixedDecoder(model, capacity=capacity, compiled=True)
    start = perf_counter_ns()
    numerical = _check_candidates(candidates, prompt_ids, schedule)
    mx.synchronize()
    admission_and_first_compile_ms = (perf_counter_ns() - start) / 1e6
    names = list(candidates)
    samples = {name: {"prefill": [], "decode": []} for name in names}
    measured_order = []
    for trial in range(-warmup, repeats):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        if trial >= 0:
            measured_order.append(order)
        for name in order:
            decoder = candidates[name]
            mx.synchronize()
            start = perf_counter_ns()
            mx.eval(decoder.prefill(prompt_ids), _arrays(decoder))
            mx.synchronize()
            prefill = (perf_counter_ns() - start) / 1e6
            start = perf_counter_ns()
            for token in schedule:
                mx.eval(decoder.step(token), _arrays(decoder))
            mx.synchronize()
            decode = (perf_counter_ns() - start) / 1e6
            if trial >= 0:
                samples[name]["prefill"].append(prefill)
                samples[name]["decode"].append(decode)
    timing = {
        name: {
            "prefill_and_cache_setup_samples_ms": sample["prefill"],
            "decode_samples_ms": sample["decode"],
            "median_decode_ms": statistics.median(sample["decode"]),
            "median_decode_tokens_per_second": len(schedule)
            * 1000
            / statistics.median(sample["decode"]),
            "decode_steps": len(schedule),
            "prompt_token_count": len(prompt_ids),
            "capacity": capacity if not name.endswith("_dynamic") else None,
            "compiled": name.endswith("_compiled"),
            "trace_count": candidates[name].trace_count if name.endswith("_compiled") else None,
            "repeats": repeats,
            "warmup": warmup,
            "measurement": "teacher_forced_cached_decode_wall_clock",
            "includes_token_sampling": False,
            "evaluates_updated_cache_arrays": True,
        }
        for name, sample in samples.items()
    }
    if any(item["compiled"] and item["trace_count"] != 1 for item in timing.values()):
        raise ValueError("compiled decoder retraced; fixed-state admission failed")
    pairs = {}
    for name in models:
        pairs[name + "_compile_only"] = (name + "_fixed_eager", name + "_compiled")
        pairs[name + "_compile_and_cache"] = (name + "_dynamic", name + "_compiled")
    if "stock" in models and "fused" in models:
        pairs.update(
            eager_fusion=("stock_dynamic", "fused_dynamic"),
            compiled_fusion=("stock_compiled", "fused_compiled"),
            combined_vs_stock_eager=("stock_dynamic", "fused_compiled"),
        )
    ratios = {
        name: {
            **paired_latency_ratio(
                timing[base]["decode_samples_ms"], timing[opt]["decode_samples_ms"]
            ),
            "baseline": base,
            "optimized": opt,
        }
        for name, (base, opt) in pairs.items()
    }
    return {
        "timing": timing,
        "ratios": ratios,
        "numerical_checks": numerical,
        "measured_trial_order": measured_order,
        "admission_and_first_compile_ms": admission_and_first_compile_ms,
    }
