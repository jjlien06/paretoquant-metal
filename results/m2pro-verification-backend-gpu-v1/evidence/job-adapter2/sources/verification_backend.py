"""CPU-testable verification boundary; no numerical or admission guarantees.

Scoped module patches are process-global and NOT concurrency-safe. Use only
inside a single-threaded isolated actor (one worker per GPU); never install as
a general global plugin or allow concurrent model calls in the same process.
"""

from contextlib import ExitStack
from unittest.mock import patch


def make_verification_forward(original_forward, *, mx, qwen2, qmm, attention):
    """Wrap an injected phase forward without importing MLX or model modules.

    Only explicit ``all_logits=True`` and metadata shape (1, L), 2 <= L <= 9,
    activate scoped component substitutions. Width one and inactive paths call
    the original forward unchanged. Missing/False flags do not inspect shape.
    Non-bool flags raise TypeError; invalid active metadata raises ValueError
    before any patch or forward call. Shape containers must be exact built-in
    tuple/list instances with exact built-in integer dimensions (not bools).

    The injected attention callable owns native SDPA delegation. The qmm
    adapter only enforces the affine call contract; it does not validate
    tensor contents, quantization layouts, or component numerical correctness.
    Arguments, cache, return objects, and exceptions are forwarded unchanged.
    Exception cleanup restores functions, not original-forward cache mutations.
    """

    def quantized_matmul(
        x,
        weights,
        scales,
        biases=None,
        *,
        transpose=False,
        group_size=64,
        bits=4,
        mode="affine",
        stream=None,
    ):
        if transpose is not True:
            raise ValueError("quantized_matmul requires transpose=True")
        if type(mode) is not str or mode != "affine":
            raise ValueError('quantized_matmul requires mode="affine"')
        if stream is not None:
            raise ValueError("quantized_matmul requires stream=None")
        if biases is None:
            raise ValueError("quantized_matmul requires affine biases")
        return qmm(x, weights, scales, biases, bits=bits, group_size=group_size)

    def forward(model, inputs, cache, **kwargs):
        if "all_logits" in kwargs and type(kwargs["all_logits"]) is not bool:
            raise TypeError("all_logits must be a bool when present")
        if kwargs.get("all_logits") is True:
            shape = getattr(inputs, "shape", ())
            if (
                type(shape) not in (tuple, list)
                or len(shape) != 2
                or any(type(dimension) is not int for dimension in shape)
                or shape[0] != 1
                or not 1 <= shape[1] <= 9
            ):
                raise ValueError("inputs shape must be (1, L) with integer 1 <= L <= 9")
            if shape[1] > 1:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(mx, "quantized_matmul", quantized_matmul))
                    stack.enter_context(
                        patch.object(qwen2, "scaled_dot_product_attention", attention)
                    )
                    return original_forward(model, inputs, cache, **kwargs)
        return original_forward(model, inputs, cache, **kwargs)

    return forward
