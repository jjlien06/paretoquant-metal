"""CPU-only reference fixtures; no GPU, SSH, checkpoints or downloads."""

import copy

import pytest


class Teacher:
    """Two genuine mutable rank states, prefix-dependent deterministic teacher."""

    def __init__(self, draft_matches=True, eos_at=None):
        self.matches = draft_matches
        self.eos_at = eos_at
        self.events = []
        self.rounds = 0

    def token(self, prefix):
        return 99 if len(prefix) == self.eos_at else (sum(prefix) * 3 + len(prefix)) % 17

    def fresh(self, prompt):
        self.ranks = [list(prompt), list(prompt)]
        self.mtp = list(prompt[:-1])
        self.events = []
        return self.token(prompt), tuple(prompt)

    def draft(self, hidden, y):
        assert hidden == tuple(self.ranks[0])
        self.mtp.append(y)
        actual = self.token(self.ranks[0] + [y])
        return actual if self.matches else (actual + 1) % 17

    def snapshot(self):
        return copy.deepcopy((self.ranks, self.mtp))

    def restore(self, saved):
        self.ranks, self.mtp = copy.deepcopy(saved)
        self.events.append(("restore", 0, 1))

    def verify(self, tokens):
        self.events.append(("verify", tuple(tokens)))
        self.rounds += 1
        preds = []
        hidden = []
        for t in tokens:
            for rank in self.ranks:
                rank.append(t)
            preds.append(self.token(self.ranks[0]))
            hidden.append(tuple(self.ranks[0]))
        return preds, hidden

    def replay(self, y):
        self.events.append(("replay", y, 0, 1))
        return self.verify([y])

    def advance_mtp(self, hidden, token):
        self.mtp.append(token)


def baseline(teacher, prompt, cap, eos):
    prefix = list(prompt)
    out = []
    for _ in range(cap):
        t = teacher.token(prefix)
        out.append(t)
        if t in eos:
            break
        prefix.append(t)
    return out


@pytest.mark.parametrize("matches", [True, False])
@pytest.mark.parametrize("cap", [1, 2, 3, 4, 7, 12])
@pytest.mark.parametrize("eos_at", [None, 3, 4, 5, 6])
def test_reference_exact_ids_eos_cap_both_rank_replay_and_freshness(matches, cap, eos_at):
    from speculative import generate

    teacher = Teacher(matches, eos_at)
    prompt = [2, 5, 8]
    eos = {99}
    expected = baseline(teacher, prompt, cap, eos)
    result = generate(teacher, prompt, max_tokens=cap, eos_ids=eos)
    assert result["token_ids"] == expected
    assert result["finish_reason"] == ("stop" if expected[-1] in eos else "length")
    assert teacher.ranks[0] == teacher.ranks[1] == prompt + expected[:-1]
    assert len(teacher.mtp) == len(teacher.ranks[0]) - 1
    if not matches and cap > 1 and expected[0] not in eos:
        assert ("restore", 0, 1) in teacher.events
        assert any(e[0] == "replay" and e[-2:] == (0, 1) for e in teacher.events)
    # Never reuse target or MTP cache on second request.
    again = generate(teacher, [3, 1], max_tokens=cap, eos_ids=eos)
    assert again["token_ids"] == baseline(teacher, [3, 1], cap, eos)


def test_invalid_request_never_initializes_backend():
    from speculative import generate

    teacher = Teacher()
    for prompt, cap in [([], 2), ([1], 0), ([True], 2), ([1], 513)]:
        with pytest.raises(ValueError):
            generate(teacher, prompt, max_tokens=cap, eos_ids=set())
    assert teacher.events == []
