"""030 adaptive depth control-flow tests with a symbolic prefix-sensitive backend.

Integer-prediction doubles: control-flow evidence only, never model/timing evidence.
"""

import random

import pytest
from rank_control import Command, Worker
from speculative import generate, generate_adaptive


def oracle(seq):
    return (sum(seq) * 7 + len(seq) * 3) % 50


class Backend:
    prefix_rollback = True

    def __init__(self, drafter_err, p2_values, rng):
        self.err, self.p2, self.rng = drafter_err, p2_values, rng
        self.ops = []

    def fresh(self, prompt):
        self.consumed = list(prompt)
        self.head_pairs = len(prompt)  # head KV = prompt positions + one per draft call
        return oracle(self.consumed), ("h", len(self.consumed))

    def _guess(self, seq):
        t = oracle(seq)
        return (t + 1) % 50 if self.rng.random() < self.err else t

    def draft(self, hidden, y):
        self.ops.append(("draft",))
        return self._guess(self.consumed + [y])

    def draft_adaptive(self, hidden, y, *, allow_second, min_p1=0.0):
        d1 = self._guess(self.consumed + [y])
        p1 = self.rng.choice(self.p2)
        if not allow_second or p1 < min_p1:
            return d1, p1, None, None
        d2 = self._guess(self.consumed + [y, d1])
        return d1, 0.9, d2, self.rng.choice(self.p2)

    def snapshot(self):
        self.ops.append(("snapshot",))
        return None

    def verify(self, tokens):
        self.ops.append(("verify", len(tokens)))
        self.pending = list(tokens)
        preds = [oracle(self.consumed + tokens[: i + 1]) for i in range(len(tokens))]
        self.consumed = self.consumed + list(tokens)
        self.width = len(tokens)
        self.preds = preds
        return preds, [("h", len(self.consumed) - len(tokens) + i + 1) for i in range(len(tokens))]

    def commit_prefix(self, keep):
        assert 0 <= keep <= self.width
        self.ops.append(("commit", keep))
        self.consumed = self.consumed[: len(self.consumed) - self.width + keep]
        return self.preds[:keep], [("h", len(self.consumed))] * keep

    def advance_mtp(self, hidden, token):
        self.ops.append(("advance",))

    def restore(self, saved):
        raise AssertionError("abort path")


def native(prompt, cap, eos):
    seq, out = list(prompt), []
    while True:
        t = oracle(seq)
        out.append(t)
        if t in eos or len(out) == cap:
            return out, seq
        seq.append(t)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0])
def test_adaptive_matches_native_and_cache(seed, tau):
    rng = random.Random(seed)
    prompt = [rng.randrange(50) for _ in range(rng.randrange(1, 6))]
    cap = rng.randrange(1, 40)
    eos = {rng.randrange(50)}
    b = Backend(rng.choice([0.0, 0.2, 0.5, 1.0]), [0.1, 0.6, 0.95], rng)
    r = generate_adaptive(b, prompt, max_tokens=cap, eos_ids=eos, tau=tau)
    expected, consumed = native(prompt, cap, eos)
    assert r["token_ids"] == expected
    # Terminal token emitted, not consumed: cache equals native convention.
    assert b.consumed == consumed
    assert r["verification_rounds"] == sum(op[0] == "verify" for op in b.ops)
    assert r["width3_rounds"] == sum(op == ("verify", 3) for op in b.ops)
    if tau == 1.0:
        assert r["width3_rounds"] == 0


def test_depth1_unchanged_path_still_matches():
    rng = random.Random(3)
    b = Backend(0.3, [0.5], rng)
    r = generate(b, [1, 2, 3], max_tokens=30, eos_ids={49})
    assert r["token_ids"] == native([1, 2, 3], 30, {49})[0]
    assert all(op != ("verify", 3) for op in b.ops)


def test_rank_worker_bounds_commit_by_verify_width():
    w = Worker(lambda op, tokens: None)
    w.handle(Command("r", 0, "reset", []))
    w.handle(Command("r", 1, "verify", [1, 2]))
    with pytest.raises(ValueError):
        w.handle(Command("r", 2, "commit_prefix", [3]))
    w = Worker(lambda op, tokens: None)
    w.handle(Command("r", 0, "reset", []))
    w.handle(Command("r", 1, "verify", [1, 2, 3]))
    w.handle(Command("r", 2, "commit_prefix", [2]))
    with pytest.raises(ValueError):
        Command("r", 3, "verify", [1, 2, 3, 4]).validate()


def test_adaptive_schedule_requires_triples():
    from workload_schedule import SCHEMA_ADAPTIVE, validate_schedule

    base = dict(case_id="c", prompt="p", max_tokens=8, warmup=False, repeat=0)
    rows = [dict(base, id=m, method=m) for m in ("native", "mtp", "adaptive")]
    validate_schedule({"schema": SCHEMA_ADAPTIVE, "requests": rows})
    with pytest.raises(ValueError):
        validate_schedule({"schema": SCHEMA_ADAPTIVE, "requests": rows[:2]})
