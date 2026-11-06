"""Greedy depth-k draft / width-(k+1) verify / prefix-keep commit control flow.

Backend contract (single host here; 021 would need collective versions):
  fresh(prompt) -> first greedy token (prefill done, last prompt token consumed)
  verify(tokens) -> greedy prediction for every input position (len(tokens))
  commit_prefix(keep) -> keep only the first `keep` verified inputs in all caches
Drafter contract: propose(tokens_so_far, k) -> list of k proposals for the next
k positions after tokens_so_far[-1]. (Real MTP: chained head; here a stand-in.)

Cache convention matches native greedy decode: after generation the target
cache holds prompt + every emitted token except the last one (terminal token is
emitted, not consumed).
"""


def generate(backend, drafter, prompt, *, depth, max_tokens, eos_ids):
    if type(depth) is not int or not 0 <= depth <= 4:
        raise ValueError("depth must be int in 0..4")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512 or not prompt:
        raise ValueError("Require prompt and 1..512 token cap")
    eos = set(eos_ids)
    y = backend.fresh(list(prompt))
    tokens = [y]
    rounds = proposed = accepted = 0
    keeps, widths = [], []
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        remaining = max_tokens - len(tokens)
        k = min(depth, remaining - 1)  # never draft past the cap
        drafts = list(drafter.propose(list(tokens), k)) if k else []
        if len(drafts) != k:
            raise ValueError("Drafter returned wrong count")
        inputs = [tokens[-1]] + drafts
        preds = backend.verify(inputs)
        if len(preds) != len(inputs):
            raise ValueError("Verifier must return every slot")
        rounds += 1
        proposed += k
        n = 0
        while n < k and preds[n] == drafts[n]:
            n += 1
        accepted += n
        emitted = drafts[:n] + [preds[n]]
        out = []
        for t in emitted:
            out.append(t)
            if t in eos or len(tokens) + len(out) == max_tokens:
                break
        # emitted token j was predicted from input j; all but the last emitted
        # token are consumed as inputs, plus the round's first input.
        keep = len(out)
        if keep < len(inputs):
            backend.commit_prefix(keep)
        else:
            backend.commit_prefix(len(inputs))  # explicit no-op commit
        keeps.append(keep)
        widths.append(len(inputs))
        tokens.extend(out)
    return {
        "token_ids": tokens,
        "finish_reason": "stop" if tokens[-1] in eos else "length",
        "verification_rounds": rounds,
        "proposed_drafts": proposed,
        "accepted_drafts": accepted,
        "keeps": keeps,
        "widths": widths,
    }
