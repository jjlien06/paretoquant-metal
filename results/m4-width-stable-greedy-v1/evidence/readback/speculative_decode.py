"""Opt-in greedy verification; no import-time MLX dependency.

Exact acceptance of the teacher's verification argmax, not a claim that batched
GPU arithmetic equals sequential arithmetic. No speedup or native TPS claim.
"""

import time


def accept_tokens(proposals, teacher, eos, remaining):
    """Accept a matching prefix, then the teacher correction/bonus."""
    if (
        type(remaining) is not int
        or remaining < 1
        or len(teacher) != len(proposals) + 1
        or any(type(t) is not int or t < 0 for t in (*proposals, *teacher, *eos))
    ):
        raise ValueError("Invalid verification tokens, width or remaining budget")
    emitted, accepted = [], 0
    for index, token in enumerate(teacher):
        matches = index < len(proposals) and proposals[index] == token
        emitted.append(token)
        accepted += int(matches)
        if not matches or token in eos or len(emitted) == remaining:
            break
    return emitted, accepted


def validate_draft_tokens(value):
    if type(value) is not int or value not in (1, 2, 4, 8):
        raise ValueError("draft_tokens must be an integer in {1, 2, 4, 8}")


def validate_packet(values, width, round_id, offset, vocab_size):
    """Validate the entire fixed CPU-int32 packet before using any payload."""
    validate_draft_tokens(width)
    if (
        any(type(x) is not int or not 0 <= x <= 2147483647 for x in (round_id, offset, vocab_size))
        or vocab_size == 0
    ):
        raise ValueError("Invalid packet metadata")
    if len(values) != width + 4 or any(type(x) is not int for x in values):
        raise ValueError("Invalid packet width or integer representation")
    r, n, count = values[:3]
    if (r, n) != (round_id, offset) or not 0 <= count <= width + 1:
        raise ValueError("Stale packet or invalid count")
    ids, padding = values[3 : 3 + count], values[3 + count :]
    if any(not 0 <= t < vocab_size for t in ids) or any(t != -1 for t in padding):
        raise ValueError("Invalid packet token or padding sentinel")
    return list(ids)


def packet(ids, width, round_id, offset, vocab_size):
    validate_draft_tokens(width)
    values = [round_id, offset, len(ids), *ids, *([-1] * (width + 1 - len(ids)))]
    validate_packet(values, width, round_id, offset, vocab_size)
    return values


def cache_offset(cache):
    if not isinstance(cache, (list, tuple)) or not cache:
        raise ValueError("Require nonempty retained-layer cache")
    for state in cache:
        if (
            not callable(getattr(state, "trim", None))
            or not callable(getattr(state, "is_trimmable", None))
            or state.is_trimmable() is not True
            or type(getattr(state, "offset", None)) is not int
            or state.offset < 0
        ):
            raise ValueError("Require trimmable KV caches with integer offsets")
    offsets = {state.offset for state in cache}
    if len(offsets) != 1:
        raise ValueError("Layer cache offsets differ")
    return offsets.pop()


def trim_to(cache, offset):
    old = cache_offset(cache)
    if type(offset) is not int or not 0 <= offset <= old:
        raise ValueError("Cannot extend a cache by trimming")
    if old != offset:
        for state in cache:
            state.trim(old - offset)
    if cache_offset(cache) != offset:
        raise ValueError("Cache trim failed to restore offset")


def validate_tokenizers(teacher, draft, target_vocab, draft_vocab):
    """Fail closed: a configured draft tokenizer is required on every rank.

    Teacher tokenizer IDs must be a semantic subset of draft IDs; embedding
    padding is not a token. No translation, retokenization or teacher masking.
    """
    if (
        type(target_vocab) is not int
        or type(draft_vocab) is not int
        or not 0 < target_vocab <= 2147483647
        or not 0 < draft_vocab <= 2147483647
    ):
        raise ValueError("Invalid model vocabulary sizes")
    maps = []
    for tokenizer in (teacher, draft):
        getter = getattr(tokenizer, "get_vocab", None)
        vocab = getter() if callable(getter) else None
        if (
            not isinstance(vocab, dict)
            or not vocab
            or any(not isinstance(t, str) or type(i) is not int or i < 0 for t, i in vocab.items())
            or len(set(vocab.values())) != len(vocab)
        ):
            raise ValueError("Require unambiguous tokenizer get_vocab dictionaries")
        maps.append({i: t for t, i in vocab.items()})
    shared = frozenset(maps[0])
    if any(i >= min(target_vocab, draft_vocab) or maps[0][i] != maps[1].get(i) for i in shared):
        raise ValueError("Teacher/draft shared token ID semantics differ")
    for name in ("bos_token_id", "eos_token_ids"):
        left, right = getattr(teacher, name, None), getattr(draft, name, None)
        if name == "eos_token_ids":
            if (
                not isinstance(left, (set, list, tuple))
                or not isinstance(right, (set, list, tuple))
                or any(type(i) is not int or not 0 <= i < target_vocab for i in (*left, *right))
            ):
                raise ValueError("Invalid EOS token IDs")
            left, right = set(left), set(right)
        elif any(
            i is not None and (type(i) is not int or not 0 <= i < target_vocab)
            for i in (left, right)
        ):
            raise ValueError("Invalid BOS token ID")
        if left != right:
            raise ValueError("Teacher/draft special token semantics differ")
    return shared


def _load_runtime():
    # Lazy imports only; CPU tests replace this boundary without importing MLX.
    import mlx.core as mx
    import phased_decode
    from mlx_lm.models.cache import make_prompt_cache

    return mx, make_prompt_cache, phased_decode


def speculative_generate(
    model,
    tokenizer,
    prompt,
    *,
    draft_model=None,
    draft_tokenizer=None,
    group=None,
    max_tokens=64,
    draft_tokens=4,
    eval_group_size=8,
    progress=None,
    prefill_step_size=256,
    context_limit=4096,
    verification_mode="batch",
):
    """Fresh-cache request; pending current token is never cached at round start.

    Receipt includes EOS. Forward counters count individual calls, including
    prefill and draft catch-up; target_verification_calls counts rounds, not
    individual teacher forwards. Sequential verification is a width-one
    arithmetic diagnostic, not a speculative speedup (no teacher amortization).
    A configured draft tokenizer is mandatory, including on rank 0.
    """
    if type(verification_mode) is not str or verification_mode not in ("batch", "sequential"):
        raise ValueError("verification_mode must be 'batch' or 'sequential'")
    validate_draft_tokens(draft_tokens)
    mx, make_cache, phased = _load_runtime()
    phased.validate_eval_group_size(eval_group_size)
    rank, size = phased.ownership(model, group)
    local_draft = size == 1 or rank == 1
    if local_draft != (draft_model is not None):
        raise ValueError("Draft required only on rank 1 (or single host)")
    vocab = model.args.vocab_size
    draft_vocab = draft_model.args.vocab_size if local_draft else vocab
    shared_ids = validate_tokenizers(tokenizer, draft_tokenizer, vocab, draft_vocab)
    if isinstance(prompt, str):
        bos = getattr(tokenizer, "bos_token", None)
        ids = list(
            tokenizer.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos))
        )
    else:
        ids = list(prompt)
    phased.validate_options(ids, max_tokens, prefill_step_size, context_limit)
    limit = min(
        context_limit,
        getattr(model.args, "max_position_embeddings", context_limit),
        getattr(draft_model.args, "max_position_embeddings", context_limit)
        if local_draft
        else context_limit,
    )
    phased.validate_options(ids, max_tokens, prefill_step_size, limit)
    if any(t not in shared_ids for t in ids):
        raise ValueError("Prompt token outside shared semantic vocabulary")
    emit = progress or (lambda event: None)
    target_cache = make_cache(model)
    draft_cache = make_cache(draft_model) if local_draft else None
    if cache_offset(target_cache) != 0 or (local_draft and cache_offset(draft_cache) != 0):
        raise ValueError("Require fresh caches")
    started = time.perf_counter()
    draft_calls, target_calls = 0, 0

    def target_call(tokens, logits):
        nonlocal target_calls
        target_calls += 1
        before = cache_offset(target_cache)
        result = phased.phased_forward(
            model,
            mx.array([tokens], dtype=mx.int32),
            target_cache,
            group=group,
            progress=progress,
            logits=logits,
            all_logits=logits,
            eval_group_size=eval_group_size,
        )
        if cache_offset(target_cache) != before + len(tokens):
            raise ValueError("Teacher forward cache growth differs from input width")
        return result

    def draft_call(tokens, logits):
        nonlocal draft_calls
        draft_calls += 1
        before = cache_offset(draft_cache)
        inputs = mx.array([tokens], dtype=mx.int32)
        core = getattr(draft_model, "model", None)
        result = (
            core(inputs, cache=draft_cache)
            if not logits and callable(core)
            else draft_model(inputs, cache=draft_cache)
        )
        mx.eval(result, [state.state for state in draft_cache])
        if cache_offset(draft_cache) != before + len(tokens):
            raise ValueError("Draft forward cache growth differs from input width")
        if not logits:
            return None
        prediction = mx.argmax(result[:, -1, :vocab], axis=-1).astype(mx.int32).reshape(-1)
        mx.eval(prediction)
        sampled = prediction.tolist()
        if len(sampled) != 1 or type(sampled[0]) is not int or sampled[0] not in shared_ids:
            raise ValueError("Draft argmax outside shared semantic vocabulary")
        return sampled[0]

    def send_ids(values, peer, round_id, offset):
        data = mx.array(packet(values, draft_tokens, round_id, offset, vocab), dtype=mx.int32)
        mx.eval(data)
        mx.eval(mx.distributed.send(data, peer, group=group, stream=mx.cpu))

    def receive_ids(peer, round_id, offset):
        data = mx.distributed.recv((draft_tokens + 4,), mx.int32, peer, group=group, stream=mx.cpu)
        mx.eval(data)
        return validate_packet(data.tolist(), draft_tokens, round_id, offset, vocab)

    emit({"phase": "prefill_before", "prompt_tokens": len(ids)})
    for start in range(0, len(ids) - 1, prefill_step_size):
        chunk = ids[start : min(start + prefill_step_size, len(ids) - 1)]
        target_call(chunk, False)
        if local_draft:
            draft_call(chunk, False)
    current, tokens, first = ids[-1], [], None
    eos, finish = set(tokenizer.eos_token_ids), "length"
    stats = {"rounds": 0, "proposed": 0, "accepted": 0}
    while len(tokens) < max_tokens:
        offset = len(ids) - 1 + len(tokens)
        if cache_offset(target_cache) != offset or (
            local_draft and cache_offset(draft_cache) != offset
        ):
            raise ValueError("Round caches do not share the pending-token offset")
        remaining = max_tokens - len(tokens)
        proposals, pending = [], current
        k = min(draft_tokens, remaining - 1)
        if local_draft:
            for _ in range(k):
                pending = draft_call([pending], True)
                proposals.append(pending)
                if pending in eos:
                    break
            if size == 2:
                send_ids(proposals, 0, stats["rounds"], offset)
        else:
            proposals = receive_ids(1, stats["rounds"], offset)
            if any(t not in shared_ids for t in proposals):
                raise ValueError("Proposal outside shared semantic vocabulary")
            if len(proposals) > k:
                raise ValueError("Proposal count exceeds remaining round budget")
            if any(t in eos for t in proposals[:-1]) or (
                len(proposals) != k and (not proposals or proposals[-1] not in eos)
            ):
                raise ValueError("Short proposals require a final draft EOS")
        width = len(proposals) + 1
        emit(
            {
                "phase": "speculative_round_before",
                "verification_mode": verification_mode,
                "round": stats["rounds"],
                "offset": offset,
                "actual_width": width,
            }
        )
        teacher_ids = []
        if verification_mode == "batch":
            scores = target_call([current, *proposals], True)
            if rank == 0:
                if scores.shape != (1, width, vocab):
                    raise ValueError("Teacher must return [1,actual_width,V] logits")
                argmax = mx.argmax(scores, axis=-1).astype(mx.int32).reshape(-1)
                mx.eval(argmax)
                teacher_ids = argmax.tolist()
        else:
            for verification_input in [current, *proposals]:
                scores = target_call([verification_input], True)
                if rank == 0:
                    if scores.shape != (1, 1, vocab):
                        raise ValueError("Teacher must return [1,1,V] logits per sequential step")
                    argmax = mx.argmax(scores, axis=-1).astype(mx.int32).reshape(-1)
                    mx.eval(argmax)
                    teacher_ids.extend(argmax.tolist())
        if rank == 0:
            teacher = validate_packet(
                packet(teacher_ids, draft_tokens, stats["rounds"], offset, vocab),
                draft_tokens,
                stats["rounds"],
                offset,
                vocab,
            )
            if size == 2:
                send_ids(teacher, 1, stats["rounds"], offset)
        else:
            teacher = receive_ids(0, stats["rounds"], offset)
        if len(teacher) != width:
            raise ValueError("Teacher packet count differs from verification width")
        emitted, accepted = accept_tokens(proposals, teacher, eos, remaining)
        if any(t not in shared_ids for t in emitted):
            raise ValueError("Teacher greedy ID outside shared semantic vocabulary")
        desired = offset + len(emitted)
        trim_to(target_cache, desired)
        if local_draft:
            if cache_offset(draft_cache) < desired:
                # Consume the last pending input, never the emitted bonus.
                if cache_offset(draft_cache) != desired - 1:
                    raise ValueError("Draft catch-up must be exactly one token")
                draft_call([proposals[-1] if proposals else current], False)
            trim_to(draft_cache, desired)
        if first is None:
            first = time.perf_counter() - started
        for token in emitted:
            emit({"phase": "token_done", "token_index": len(tokens), "token_id": token})
            tokens.append(token)
        stats["rounds"] += 1
        stats["proposed"] += len(proposals)
        stats["accepted"] += accepted
        emit(
            {
                "phase": "speculative_round_done",
                "verification_mode": verification_mode,
                "round": stats["rounds"] - 1,
                "offset": offset,
                "actual_width": width,
                "accepted": accepted,
                "proposed": len(proposals),
                "emitted": len(emitted),
                "cache_offset": desired,
            }
        )
        current = emitted[-1]
        if current in eos:
            finish = "stop"
            break
    elapsed = time.perf_counter() - started
    return {
        "token_ids": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "wall_seconds": elapsed,
        "first_token_seconds": first,
        "wall_tokens_per_second": len(tokens) / elapsed,
        "native_generation_tps": None,
        "prompt_tokens": len(ids),
        "finish_reason": finish,
        "acceptance_stats": stats,
        "draft_calls": draft_calls,
        "target_forward_calls": target_calls,
        "target_verification_calls": stats["rounds"],
        "verification_mode": verification_mode,
    }
