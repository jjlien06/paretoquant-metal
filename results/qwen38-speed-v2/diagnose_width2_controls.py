"""Diagnostic-only localization of M4 batched verification drift.

No default/runtime edits. Each optional substitution is scoped to an explicit
all-logits multirow diagnostic, and restored even on failure. Serial operators
are a localization control, NOT an acceleration claim.
"""
from contextlib import ExitStack
import os
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_next

import diagnose_width2 as diagnostic

original_forward = diagnostic.phased_forward
original_qmm = mx.quantized_matmul
original_sdpa = qwen3_next.scaled_dot_product_attention
original_conv = nn.Conv1d.__call__
mode = os.environ.get("PARETOQUANT_WIDTH_CONTROL", "qmm")
if mode == "slot":
    import sys
    sys.path.insert(0, os.environ["PARETOQUANT_SLOT_RUNTIME"])
    from slot_qmv import slot_qmv
if mode not in {"qmm", "qmm_sdpa", "qmm_sdpa_conv", "slot"}:
    raise ValueError("Unknown numerical localization control")


def singleton_qmm(x, *args, **kwargs):
    if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] > 1:
        if mode == "slot":
            w = args[0]
            scales = args[1] if len(args) > 1 else kwargs["scales"]
            biases = args[2] if len(args) > 2 else kwargs["biases"]
            assert kwargs.get("transpose") is True
            assert kwargs.get("mode", "affine") == "affine"
            assert kwargs.get("stream") is None
            return slot_qmv(x, w, scales, biases, bits=kwargs.get("bits", 4),
                            group_size=kwargs.get("group_size", 64))
        return mx.concatenate([
            original_qmm(mx.contiguous(x[:, i:i + 1, :]), *args, **kwargs)
            for i in range(x.shape[1])
        ], axis=1)
    return original_qmm(x, *args, **kwargs)


def singleton_sdpa(q, k, v, *, cache, scale, mask, **kwargs):
    if q.shape[-2] == 1:
        return original_sdpa(q, k, v, cache=cache, scale=scale, mask=mask, **kwargs)
    assert q.shape[0] == 1 and mask == "causal"
    assert cache.offset == k.shape[-2] and not hasattr(cache, "bits")
    width = q.shape[-2]
    return mx.concatenate([
        mx.fast.scaled_dot_product_attention(
            mx.contiguous(q[..., i:i + 1, :]),
            k[..., :k.shape[-2] - width + i + 1, :],
            v[..., :v.shape[-2] - width + i + 1, :],
            scale=scale, mask=None,
        )
        for i in range(width)
    ], axis=-2)


def singleton_conv(module, x):
    kernel = module.weight.shape[1]
    if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == kernel + 1:
        return mx.concatenate([
            original_conv(module, mx.contiguous(x[:, i:i + kernel, :]))
            for i in range(2)
        ], axis=1)
    return original_conv(module, x)


def controlled_forward(model, inputs, cache, **kwargs):
    active = kwargs.get("all_logits") is True and inputs.shape[1] > 1
    if not active:
        return original_forward(model, inputs, cache, **kwargs)
    with ExitStack() as contexts:
        contexts.enter_context(patch.object(mx, "quantized_matmul", singleton_qmm))
        if mode in {"qmm_sdpa", "qmm_sdpa_conv"}:
            contexts.enter_context(patch.object(qwen3_next, "scaled_dot_product_attention", singleton_sdpa))
        if mode == "qmm_sdpa_conv":
            contexts.enter_context(patch.object(nn.Conv1d, "__call__", singleton_conv))
        return original_forward(model, inputs, cache, **kwargs)


if __name__ == "__main__":
    diagnostic.phased_forward = controlled_forward
    print("NUMERICAL_CONTROL", mode)
    diagnostic.main()
