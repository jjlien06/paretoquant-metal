"""CPU-only control-flow tests of depth-k speculation with a prefix-sensitive
symbolic target (no MLX). Integer doubles: control-flow evidence only."""

import hashlib
import random

import pytest
from speculative_k import generate

V = 50


def oracle(prefix):
    h = hashlib.sha256(",".join(map(str, prefix)).encode()).digest()
    return h[0] % V


class SymbolicBackend:
    """'Cache' = exact consumed token list. verify appends; commit truncates."""

    def __init__(self):
        self.cache = None
        self.pending = None

    def fresh(self, prompt):
        self.cache = list(prompt)
        return oracle(self.cache)

    def verify(self, tokens):
        assert self.pending is None, "commit required between verifies"
        preds = []
        for t in tokens:
            self.cache.append(t)
            preds.append(oracle(self.cache))
        self.pending = len(tokens)
        return preds

    def commit_prefix(self, keep):
        assert self.pending is not None and 0 <= keep <= self.pending
        drop = self.pending - keep
        if drop:
            del self.cache[-drop:]
        self.pending = None


def sequential(prompt, max_tokens, eos):
    cache = list(prompt)
    toks = [oracle(cache)]
    while len(toks) < max_tokens and toks[-1] not in eos:
        cache.append(toks[-1])
        toks.append(oracle(cache))
    return toks, cache


class Drafter:
    def __init__(self, ref, rate, seed):
        self.ref, self.rng, self.rate = ref, random.Random(seed), rate

    def propose(self, tokens, k):
        p = len(tokens)
        out = []
        for i in range(k):
            t = self.ref[p + i] if p + i < len(self.ref) else 0
            out.append((t + 1) % V if self.rng.random() < self.rate else t)
        return out


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("rate", [0.0, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("eos", [set(), {7}, {3, 11}])
@pytest.mark.parametrize("cap", [1, 2, 3, 5, 40])
def test_equivalence_and_cache_convention(depth, rate, eos, cap):
    prompt = [1, 2, 3, 4]
    ref, ref_cache = sequential(prompt, cap, eos)
    for seed in range(3):
        b = SymbolicBackend()
        r = generate(b, Drafter(ref, rate, seed), prompt, depth=depth, max_tokens=cap,
                     eos_ids=eos)
        assert r["token_ids"] == ref
        assert b.cache == ref_cache  # prompt + emitted[:-1]
        assert len(r["token_ids"]) <= cap
        assert all(1 <= k <= w <= depth + 1 for k, w in zip(r["keeps"], r["widths"]))
        if rate == 0.0 and depth:
            # perfect drafts: every non-final round emits full width
            assert all(k == w for k, w in zip(r["keeps"][:-1], r["widths"][:-1]))
        if rate == 1.0:
            assert r["accepted_drafts"] == 0


def test_rounds_shrink_with_depth_for_perfect_drafter():
    prompt = [9, 9]
    ref, _ = sequential(prompt, 64, set())
    rounds = []
    for depth in range(5):
        r = generate(SymbolicBackend(), Drafter(ref, 0.0, 0), prompt, depth=depth,
                     max_tokens=64, eos_ids=set())
        rounds.append(r["verification_rounds"])
    # 63 post-first tokens, ceil(63/(d+1)) rounds
    assert rounds == [63, 32, 21, 16, 13]


def test_rejects_bad_drafter_and_depth():
    class Bad:
        def propose(self, tokens, k):
            return [0] * (k + 1)

    with pytest.raises(ValueError):
        generate(SymbolicBackend(), Bad(), [1], depth=2, max_tokens=5, eos_ids=set())
    with pytest.raises(ValueError):
        generate(SymbolicBackend(), Bad(), [1], depth=5, max_tokens=5, eos_ids=set())
