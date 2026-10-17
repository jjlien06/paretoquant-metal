"""Explicit Qwen3.5 layers, synchronized CPU handoffs, stock model arithmetic.

No custom fusion, compile, or native asynchronous benchmark scheduling.
"""

import time


def ownership(model, group):
    if model.model_type != "qwen3_5":
        raise ValueError("Phased execution admits only built-in Qwen3.5")
    core = model.model
    rank, size = (0, 1) if group is None else (group.rank(), group.size())
    if (rank, size) not in {(0, 1), (0, 2), (1, 2)}:
        raise ValueError("Require one host or exactly two pipeline ranks")
    if core.pipeline_rank != rank or core.pipeline_size != size:
        raise ValueError("Model ownership differs from supplied group")
    layers = core.pipeline_layers
    end = core.end_idx or model.language_model.args.num_hidden_layers
    if (
        not 0 <= core.start_idx < end <= model.language_model.args.num_hidden_layers
        or len(layers) != end - core.start_idx
    ):
        raise ValueError("Invalid retained layer interval")
    if not layers or any(layer is None for layer in layers):
        raise ValueError("Require nonempty owned original layers")
    if size == 1 and (core.start_idx != 0 or end != model.language_model.args.num_hidden_layers):
        raise ValueError("Single host must own every layer")
    if size == 2 and (
        (rank == 1 and (core.start_idx != 0 or end >= model.language_model.args.num_hidden_layers))
        or (
            rank == 0
            and (core.start_idx <= 0 or end != model.language_model.args.num_hidden_layers)
        )
    ):
        raise ValueError("Require reverse contiguous pipeline ownership")
    return rank, size


def validate_eval_group_size(eval_group_size):
    if type(eval_group_size) is not int or eval_group_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError("eval_group_size must be an integer in {1, 2, 4, 8, 16, 32}")


def validate_eval_schedule(eval_group_size, schedule, repeats, phased):
    """Resolve a bounded trusted request schedule, preserving the default size."""
    validate_eval_group_size(eval_group_size)
    if type(repeats) is not int or not 1 <= repeats <= 32:
        raise ValueError("Require 1..32 repeats")
    if schedule is not None and not isinstance(schedule, (list, tuple)):
        raise ValueError("Evaluation schedule must be a list")
    if not phased and (eval_group_size != 1 or schedule):
        raise ValueError("Grouped evaluation/schedules require phased execution")
    if schedule:
        if len(schedule) != repeats:
            raise ValueError("Evaluation schedule length must match repeats")
        for size in schedule:
            validate_eval_group_size(size)
    return list(schedule) if schedule else [eval_group_size] * repeats


def phased_forward(
    model,
    inputs,
    cache,
    *,
    group=None,
    progress=None,
    logits=True,
    eval_group_size=1,
    all_logits=False,
    return_hidden=False,
    receive_stage=None,
):
    """Evaluate local blocks; progress intervals are original [start_idx, end_idx).

    Grouped layer_done events attest block completion, not individual layer timing.
    Communication remains outside the local lazy layer graphs.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import ArraysCache, KVCache

    validate_eval_group_size(eval_group_size)
    if type(return_hidden) is not bool:
        raise ValueError("return_hidden must be a bool")
    if type(all_logits) is not bool:
        raise ValueError("all_logits must be a bool")
    rank, size = ownership(model, group)
    core = model.model
    layers = core.pipeline_layers
    if (
        not isinstance(cache, (list, tuple))
        or len(cache) != len(layers)
        or any(c is None or not hasattr(c, "state") for c in cache)
    ):
        raise ValueError("Cache must cover every retained layer")
    if any(
        not isinstance(state, ArraysCache if layer.is_linear else KVCache)
        for layer, state in zip(layers, cache)
    ):
        raise ValueError("Cache type differs from hybrid layer type")
    if len({c.offset for c in cache if isinstance(c, KVCache)}) > 1:
        raise ValueError("Layer cache offsets differ")
    if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] < 1:
        raise ValueError("Require inputs [1,T] with positive T")
    emit = progress or (lambda event: None)
    embedding = core.embed_tokens
    dtype = embedding.scales.dtype if hasattr(embedding, "scales") else embedding.weight.dtype
    if not mx.issubdtype(dtype, mx.floating):
        raise ValueError("Embedding compute dtype must be floating")
    if size == 2 and rank == 0:
        emit({"phase": "handoff_receive_before", "tokens": inputs.shape[1]})
        h = mx.distributed.recv(
            (1, inputs.shape[1], model.language_model.args.hidden_size),
            dtype,
            1,
            group=group,
            stream=mx.cpu,
        )
        mx.eval(h)  # finish network wait before any GPU layer graph exists
        emit(
            {
                "phase": "handoff_receive_done",
                "shape": list(h.shape),
                "dtype": str(h.dtype).removeprefix("mlx.core."),
            }
        )
        if receive_stage is not None:
            receive_stage()
    else:
        emit({"phase": "embedding_before"})
        h = embedding(inputs)
        mx.eval(h)
        emit({"phase": "embedding_done"})
    if h.dtype != dtype:
        raise ValueError("Hidden state dtype differs from embedding compute dtype")
    fa = next((c for c in cache if isinstance(c, KVCache)), None)
    ssm = next((c for c in cache if isinstance(c, ArraysCache)), None)
    fa_mask = create_attention_mask(h, fa) if fa is not None else None
    ssm_mask = create_ssm_mask(h, ssm) if ssm is not None else None
    if eval_group_size == 1:
        for index, (layer, state) in enumerate(zip(layers, cache), core.start_idx):
            emit({"phase": "layer_before", "layer_index": index})
            h = layer(h, ssm_mask if layer.is_linear else fa_mask, state)
            mx.eval(h, state.state)
            if h.dtype != dtype:
                raise ValueError("Layer changed hidden state dtype")
            emit({"phase": "layer_done", "layer_index": index})
    else:
        for start in range(0, len(layers), eval_group_size):
            end = min(start + eval_group_size, len(layers))
            interval = {"start_idx": core.start_idx + start, "end_idx": core.start_idx + end}
            emit({"phase": "block_before", **interval})
            for layer, state in zip(layers[start:end], cache[start:end]):
                h = layer(h, ssm_mask if layer.is_linear else fa_mask, state)
                if h.dtype != dtype:
                    raise ValueError("Layer changed hidden state dtype")
            mx.eval(h, [state.state for state in cache[start:end]])
            emit({"phase": "block_done", **interval})
            # Completion evidence, not per-layer timings; eval covers every live KV state.
            for index in range(interval["start_idx"], interval["end_idx"]):
                emit({"phase": "layer_done", "layer_index": index})
    if size == 2 and rank == 1:
        mx.eval(h)
        emit({"phase": "handoff_send_before"})
        sent = mx.distributed.send(h, 0, group=group, stream=mx.cpu)
        mx.eval(sent)
        emit(
            {
                "phase": "handoff_send_done",
                "shape": list(h.shape),
                "dtype": str(h.dtype).removeprefix("mlx.core."),
            }
        )
        return None
    if not logits:
        if not return_hidden:
            return None
        # Same native normalization as the reference return-hidden branch;
        # retain ALL positions for shifted MTP recurrence, never project vocab.
        emit({"phase": "hidden_norm_before"})
        h = core.norm(h)
        mx.eval(h)
        emit({"phase": "hidden_norm_done"})
        return None, h
    emit({"phase": "head_before"})
    h = core.norm(h if all_logits else h[:, -1:, :])
    out = (
        embedding.as_linear(h)
        if model.language_model.args.tie_word_embeddings
        else model.language_model.lm_head(h)
    )
    mx.eval(out)
    emit({"phase": "head_done"})
    return (out, h) if return_hidden else out


def validate_options(ids, max_tokens, prefill_step_size, context_limit):
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
        raise ValueError("Require 1..512 generated tokens")
    if type(prefill_step_size) is not int or not 1 <= prefill_step_size <= 256:
        raise ValueError("Require 1..256 prefill chunk tokens")
    if type(context_limit) is not int or not 1 <= context_limit <= 32768:
        raise ValueError("Invalid bounded context limit")
    if not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("Require nonempty token IDs")
    if len(ids) + max_tokens > context_limit:
        raise ValueError("Prompt plus generation exceeds bounded context")


def phased_generate(
    model,
    tokenizer,
    prompt,
    *,
    max_tokens=16,
    group=None,
    progress=None,
    prefill_step_size=256,
    context_limit=4096,
    eval_group_size=1,
):
    """One fresh-cache greedy request; EOS is included in the token receipt."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    validate_eval_group_size(eval_group_size)
    rank, size = ownership(model, group)
    if isinstance(prompt, str):
        bos = getattr(tokenizer, "bos_token", None)
        add_special = bos is None or not prompt.startswith(bos)
        ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    else:
        ids = list(prompt)
    limit = min(
        context_limit, getattr(model.language_model.args, "max_position_embeddings", context_limit)
    )
    validate_options(ids, max_tokens, prefill_step_size, limit)
    if any(t >= model.language_model.args.vocab_size for t in ids):
        raise ValueError("Prompt token outside vocabulary")
    emit = progress or (lambda event: None)
    cache = make_prompt_cache(model)
    started = time.perf_counter()
    emit({"phase": "prefill_before", "prompt_tokens": len(ids)})
    for start in range(0, len(ids) - 1, prefill_step_size):
        chunk = ids[start : min(start + prefill_step_size, len(ids) - 1)]
        phased_forward(
            model,
            mx.array([chunk], dtype=mx.int32),
            cache,
            group=group,
            progress=progress,
            logits=False,
            eval_group_size=eval_group_size,
        )
    current = mx.array([[ids[-1]]], dtype=mx.int32)
    tokens, first = [], None
    eos = set(tokenizer.eos_token_ids)
    finish = "length"
    for index in range(max_tokens):
        emit(
            {
                "phase": "prefill_last_before" if index == 0 else "decode_before",
                "token_index": index,
            }
        )
        scores = phased_forward(
            model, current, cache, group=group, progress=progress, eval_group_size=eval_group_size
        )
        if rank == 0:
            sampled = mx.argmax(scores[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(sampled)
            token = int(sampled.item())
            if size == 2:
                emit({"phase": "token_send_before", "token_index": index})
                mx.eval(mx.distributed.send(sampled, 1, group=group, stream=mx.cpu))
        else:
            emit({"phase": "token_receive_before", "token_index": index})
            sampled = mx.distributed.recv((1,), mx.int32, 0, group=group, stream=mx.cpu)
            mx.eval(sampled)
            token = int(sampled.item())
        if first is None:
            first = time.perf_counter() - started
        tokens.append(token)
        emit({"phase": "token_done", "token_index": index, "token_id": token})
        if token in eos:
            finish = "stop"
            break
        current = mx.array([[token]], dtype=mx.int32)
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
    }
