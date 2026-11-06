"""Greedy one-token native draft/two-token teacher reference engine.

A width2 numerical admission is REQUIRED outside this reference algorithm before
claiming real-model greedy equivalence. No runtime or GPU admission is implied.
Backend operations must be collective across both target ranks; only draft and
advance_mtp are rank0-local. snapshot is taken AFTER drafting y: its MTP entry
is already a committed pair (previous teacher hidden, y), even if d is rejected.
"""


def generate(backend, prompt, *, max_tokens, eos_ids):
    if (
        type(max_tokens) is not int
        or not 1 <= max_tokens <= 512
        or not prompt
        or any(type(t) is not int or t < 0 for t in prompt)
    ):
        raise ValueError("Require token prompt and 1..512 token cap")
    eos = set(eos_ids)
    y, hidden = backend.fresh(list(prompt))
    tokens = [y]
    accepted = 0
    rounds = 0
    teacher_forwards = 0
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        d = backend.draft(hidden, y)
        saved = backend.snapshot()
        predictions, hiddens = backend.verify([y, d])
        rounds += 1
        teacher_forwards += 1
        if len(predictions) != 2 or len(hiddens) != 2:
            backend.restore(saved)
            raise ValueError("Teacher verification must return both slots")
        correction, bonus = predictions
        if correction != d:
            # Recurrent state is NOT trimmable. Restore every state on both ranks,
            # then replay only the accepted input y at native width1.
            if getattr(backend, "prefix_rollback", False):
                replay_predictions, replay_hidden = backend.commit_prefix(1)
            else:
                backend.restore(saved)
                replay_predictions, replay_hidden = backend.replay(y)
                teacher_forwards += 1
            # The replayed teacher is authoritative, not a numerically drifting
            # rejected width2 result. Gates separately require equality.
            y = replay_predictions[0]
            hidden = replay_hidden[0]
            tokens.append(y)
        else:
            accepted += 1
            tokens.append(d)
            if len(tokens) == max_tokens or d in eos:
                # Terminal d is emitted, not consumed: match native cache convention.
                if getattr(backend, "prefix_rollback", False):
                    _, replay_hidden = backend.commit_prefix(1)
                else:
                    backend.restore(saved)
                    _, replay_hidden = backend.replay(y)
                    teacher_forwards += 1
                hidden = replay_hidden[0]
                break
            backend.advance_mtp(hiddens[0], d)
            tokens.append(bonus)
            y, hidden = bonus, hiddens[1]
    return {
        "token_ids": tokens,
        "finish_reason": "stop" if tokens[-1] in eos else "length",
        "verification_rounds": rounds,
        "accepted_drafts": accepted,
        "teacher_forwards": teacher_forwards,
        "numerical_admission": "required_separately",
    }


def generate_adaptive(backend, prompt, *, max_tokens, eos_ids, tau, report=None):
    """030 per-round adaptive depth 1/2 (greedy, deterministic).

    Every round drafts d1 (same head op as depth1). If at least two tokens remain
    before the cap, it also computes the chained d2 and its head top-1 probability p2
    and verifies [y, d1, d2] at width3 iff p2 >= tau; otherwise [y, d1] at width2.
    Accept the longest matching prefix n, emit d1..dn + prediction[n] truncated at
    EOS/cap, and commit prefix keep = len(emitted) (terminal emitted, not consumed)
    on BOTH ranks via the bounded collective commit command. Head KV catch-up uses
    only verified teacher hidden states of consumed inputs.
    """
    import time

    if (
        type(max_tokens) is not int
        or not 1 <= max_tokens <= 512
        or not prompt
        or any(type(t) is not int or t < 0 for t in prompt)
        or type(tau) is not float
        or not 0.0 <= tau <= 1.0
    ):
        raise ValueError("Require token prompt, 1..512 cap and tau in [0,1]")
    if not getattr(backend, "prefix_rollback", False):
        raise ValueError("Adaptive depth requires admitted prefix rollback")
    emit = report or (lambda event: None)
    eos = set(eos_ids)
    y, hidden = backend.fresh(list(prompt))
    tokens = [y]
    accepted = rounds = teacher_forwards = width3_rounds = 0
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        remaining = max_tokens - len(tokens)
        d1, p1, d2, p2 = backend.draft_adaptive(hidden, y, allow_second=remaining >= 2)
        drafts = [d1] + ([d2] if d2 is not None and p2 >= tau else [])
        inputs = [y] + drafts
        saved = backend.snapshot()
        began = time.perf_counter()
        predictions, hiddens = backend.verify(inputs)
        verify_seconds = time.perf_counter() - began
        rounds += 1
        teacher_forwards += 1
        if len(predictions) != len(inputs) or len(hiddens) != len(inputs):
            backend.restore(saved)
            raise ValueError("Teacher verification must return every slot")
        n = 0
        while n < len(drafts) and predictions[n] == drafts[n]:
            n += 1
        out = []
        for t in drafts[:n] + [predictions[n]]:
            out.append(t)
            if t in eos or len(tokens) + len(out) == max_tokens:
                break
        keep = len(out)
        accepted += min(n, keep)
        width3_rounds += len(inputs) == 3
        began = time.perf_counter()
        if keep < len(inputs):
            backend.commit_prefix(keep)
        commit_seconds = time.perf_counter() - began
        # Head catch-up: pairs (verified hidden of input j, input j+1) for consumed j+1.
        for j in range(keep - 1):
            backend.advance_mtp(hiddens[j], inputs[j + 1])
        emit(
            {
                "phase": "adaptive_round",
                "round": rounds - 1,
                "width": len(inputs),
                "keep": keep,
                "accepted": n,
                "d1": d1,
                "p1": round(p1, 6),
                "d2": d2,
                "p2": None if p2 is None else round(p2, 6),
                "tau": tau,
                "verify_seconds": verify_seconds,
                "commit_seconds": commit_seconds,
            }
        )
        tokens.extend(out)
        y, hidden = tokens[-1], hiddens[keep - 1]
    return {
        "token_ids": tokens,
        "finish_reason": "stop" if tokens[-1] in eos else "length",
        "verification_rounds": rounds,
        "accepted_drafts": accepted,
        "teacher_forwards": teacher_forwards,
        "width3_rounds": width3_rounds,
        "adaptive_tau": tau,
        "numerical_admission": "required_separately",
    }
