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
