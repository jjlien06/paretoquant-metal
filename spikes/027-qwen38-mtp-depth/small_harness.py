"""Single-host small-model backend, stand-in drafter and equivalence gates.

Model: local Qwen3.5-0.8B 4-bit control (same qwen3_5 hybrid GDN/full-attention
architecture as Qwen3.8-27B; NO MTP head). The drafter is a replay of the
model's own sequential greedy tokens with deterministic forced mismatches, so
it exercises verify/rollback mechanics only -- not real draft acceptance.
"""

import hashlib
from pathlib import Path

import mlx.core as mx
from prefix_rollback_k import CaptureK
from verifier_k import numerical_mode

ATOL, RTOL = 1e-6, 1e-5  # unchanged 012/021 strict tolerances
MODEL_DIR = str(Path(__file__).resolve().parents[2] / "models"
                / "control-qwen35-08b-da28692b5f139cb0ec58a356b437486b7dac7462")


def load_small(path=MODEL_DIR):
    from mlx_lm import load

    model, tok = load(path, tokenizer_config={"trust_remote_code": False})
    return model, tok


def prefill(model, cache, prompt, step=256):
    for start in range(0, len(prompt) - 1, step):
        ids = prompt[start:min(start + step, len(prompt) - 1)]
        mx.eval(model(mx.array([ids], mx.int32), cache=cache), [c.state for c in cache])
    out = model(mx.array([[prompt[-1]]], mx.int32), cache=cache)
    mx.eval(out, [c.state for c in cache])
    return out


def metric(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_ok": False, "strict_pass": False}
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    diff = mx.abs(a32 - b32)
    tol = ATOL + RTOL * mx.abs(a32)
    return {"shape_ok": True, "max_abs": float(diff.max().item()) if diff.size else 0.0,
            "violations": int((diff > tol).sum().item()) if diff.size else 0,
            "strict_pass": bool(mx.allclose(a, b, atol=ATOL, rtol=RTOL).item())}


def cache_metrics(left, right):
    from mlx_lm.models.cache import ArraysCache, KVCache

    worst = {"max_abs": 0.0, "violations": 0, "strict_pass": True, "offsets_equal": True}
    for x, y in zip(left, right, strict=True):
        if isinstance(x, KVCache):
            if x.offset != y.offset:
                worst["offsets_equal"] = worst["strict_pass"] = False
                continue
            pairs = [(x.keys[..., :x.offset, :], y.keys[..., :y.offset, :]),
                     (x.values[..., :x.offset, :], y.values[..., :y.offset, :])]
        else:
            assert isinstance(x, ArraysCache)
            pairs = list(zip(x.cache, y.cache, strict=True))
        for a, b in pairs:
            m = metric(a, b)
            worst["max_abs"] = max(worst["max_abs"], m.get("max_abs", float("inf")))
            worst["violations"] += m.get("violations", 1)
            worst["strict_pass"] = worst["strict_pass"] and m["strict_pass"]
    return worst


class SmallBackend:
    def __init__(self, model, mode="native"):
        self.model, self.mode = model, mode
        self.capture = None
        self.width = None
        self.counters = {"slot_qmv": 0, "singleton": 0, "rollback_commits": 0}

    def fresh(self, prompt):
        self.cache = self.model.make_cache()
        out = prefill(self.model, self.cache, prompt)
        return int(mx.argmax(out[0, -1]).item())

    def verify(self, tokens):
        W = len(tokens)
        x = mx.array([tokens], mx.int32)
        if W == 1:
            self.capture, self.width = None, 1
            out = self.model(x, cache=self.cache)
        else:
            self.capture, self.width = CaptureK(self.model, self.cache, W), W
            with numerical_mode(self.mode, width=W) as cnt, self.capture:
                out = self.model(x, cache=self.cache)
                mx.eval(out, [c.state for c in self.cache])
            self.counters["slot_qmv"] += cnt["slot_qmv"]
            self.counters["singleton"] += cnt["singleton"]
        mx.eval(out, [c.state for c in self.cache])
        self.last_logits = out
        return mx.argmax(out[0], -1).tolist()

    def commit_prefix(self, keep):
        if self.width == 1:
            if keep != 1:
                raise ValueError("width1 step keeps its single input")
        else:
            if keep < self.width:
                self.counters["rollback_commits"] += 1
            self.capture.commit(keep)
        self.capture = None


def sequential_greedy(model, prompt, max_tokens, eos_ids):
    cache = model.make_cache()
    out = prefill(model, cache, prompt)
    tokens = [int(mx.argmax(out[0, -1]).item())]
    while len(tokens) < max_tokens and tokens[-1] not in eos_ids:
        out = model(mx.array([[tokens[-1]]], mx.int32), cache=cache)
        mx.eval(out, [c.state for c in cache])
        tokens.append(int(mx.argmax(out[0, -1]).item()))
    return tokens, cache


class ReplayDrafter:
    """Proposes the reference greedy continuation, corrupting each proposal with
    deterministic pseudo-random probability `rate` (keyed by seed/round/slot)."""

    def __init__(self, reference, vocab, rate, seed=0):
        self.ref, self.vocab, self.rate, self.seed = list(reference), vocab, rate, seed
        self.round = 0

    def _u(self, pos):
        h = hashlib.sha256(f"{self.seed}:{self.round}:{pos}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def propose(self, tokens, k):
        p = len(tokens)
        out = []
        for i in range(k):
            true = self.ref[p + i] if p + i < len(self.ref) else 0
            out.append((true + 1) % self.vocab if self._u(p + i) < self.rate else true)
        self.round += 1
        return out


def capture_gate(model, prompt, forced, width, mode):
    """Width-W verify (instrumented) vs uninstrumented same width, then every
    prefix keep in 0..W vs native sequential width1 caches/logits + 1-token
    continuation. Strict tolerances; metrics retained, never relaxed."""
    from cache_snap import restore, snapshot

    pair = forced[:width]
    base = model.make_cache()
    prefill(model, base, prompt)
    saved = snapshot(base)
    # same-width: plain vs captured
    left, right = model.make_cache(), model.make_cache()
    restore(left, saved), restore(right, saved)
    x = mx.array([pair], mx.int32)
    with numerical_mode(mode, width=width):
        a = model(x, cache=left)
        mx.eval(a, [c.state for c in left])
        cap = CaptureK(model, right, width)
        with cap:
            b = model(x, cache=right)
            mx.eval(b, [c.state for c in right])
    same = {"logits": metric(a, b), "caches": cache_metrics(left, right)}
    rows = []
    for keep in range(width + 1):
        restore(left, saved), restore(right, saved)
        seq = []
        for t in pair[:keep]:
            o = model(mx.array([[t]], mx.int32), cache=left)
            mx.eval(o, [c.state for c in left])
            seq.append(o)
        cap = CaptureK(model, right, width)
        with numerical_mode(mode, width=width), cap:
            act = model(x, cache=right)
            mx.eval(act, [c.state for c in right])
        cap.commit(keep)
        row = {"keep": keep, "caches": cache_metrics(left, right)}
        if keep:
            row["logits_slot"] = metric(seq[-1][:, -1:], act[:, keep - 1:keep])
            row["argmax_equal"] = (int(mx.argmax(seq[-1][0, -1]).item())
                                   == int(mx.argmax(act[0, keep - 1]).item()))
        nxt = forced[width] if len(forced) > width else 1
        c1 = model(mx.array([[nxt]], mx.int32), cache=left)
        c2 = model(mx.array([[nxt]], mx.int32), cache=right)
        mx.eval(c1, c2, [c.state for c in left], [c.state for c in right])
        row["continuation_logits"] = metric(c1, c2)
        row["continuation_caches"] = cache_metrics(left, right)
        row["strict_pass"] = (row["caches"]["strict_pass"]
                              and row.get("logits_slot", {"strict_pass": True})["strict_pass"]
                              and row["continuation_logits"]["strict_pass"]
                              and row["continuation_caches"]["strict_pass"])
        row["argmax_pass"] = row.get("argmax_equal", True) and (
            int(mx.argmax(c1[0, -1]).item()) == int(mx.argmax(c2[0, -1]).item()))
        rows.append(row)
    same_pass = same["logits"]["strict_pass"] and same["caches"]["strict_pass"]
    return {"width": width, "mode": mode, "same_width": same, "same_width_strict_pass": same_pass,
            "prefixes": rows,
            "strict_pass": same_pass and all(r["strict_pass"] for r in rows),
            "argmax_pass": all(r["argmax_pass"] for r in rows)}


def e2e(model, prompt, *, depth, rate, mode, max_tokens, eos_ids, seed=0, reference=None):
    from speculative_k import generate

    if reference is None:
        reference = sequential_greedy(model, prompt, max_tokens, eos_ids)
    ref_tokens, ref_cache = reference
    vocab = model.language_model.args.vocab_size
    backend = SmallBackend(model, mode)
    drafter = ReplayDrafter(ref_tokens, vocab, rate, seed)
    res = generate(backend, drafter, prompt, depth=depth, max_tokens=max_tokens, eos_ids=eos_ids)
    cm = cache_metrics(ref_cache, backend.cache)
    first_div = next((i for i, (x, y) in enumerate(zip(ref_tokens, res["token_ids"])) if x != y),
                     None)
    return {
        "depth": depth, "mismatch_rate": rate, "mode": mode, "seed": seed,
        "max_tokens": max_tokens, "prompt_tokens": len(prompt),
        "tokens_equal": res["token_ids"] == ref_tokens, "first_divergence": first_div,
        "output_tokens": len(res["token_ids"]), "reference_tokens": len(ref_tokens),
        "finish_reason": res["finish_reason"],
        "verification_rounds": res["verification_rounds"],
        "proposed_drafts": res["proposed_drafts"], "accepted_drafts": res["accepted_drafts"],
        "keep_histogram": {str(k): res["keeps"].count(k) for k in sorted(set(res["keeps"]))},
        "width_histogram": {str(w): res["widths"].count(w) for w in sorted(set(res["widths"]))},
        "rollback_commits": backend.counters["rollback_commits"],
        "qmm_substitutions": {k: backend.counters[k] for k in ("slot_qmv", "singleton")},
        "final_cache_vs_sequential": cm,
    }
