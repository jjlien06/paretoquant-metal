"""Explicit per-rank numerical mode. Only active on width2/width3 teacher verification.

030: generalized from fixed width2 to the explicit verify width W in VERIFY_WIDTHS.
"""

VERIFY_WIDTHS = (2, 3)

from contextlib import contextmanager

MODES = ("native", "singleton-qmm", "slot-qmv")


@contextmanager
def numerical_mode(mode, *, active, width=2):
    if width not in VERIFY_WIDTHS:
        raise ValueError("Unadmitted verify width")
    if mode not in MODES:
        raise ValueError("Unknown verifier mode")
    import mlx.core as mx

    original = mx.quantized_matmul

    def singleton(
        x,
        w,
        scales,
        biases=None,
        transpose=True,
        group_size=None,
        bits=None,
        mode="affine",
        *,
        stream=None,
    ):
        kwargs = dict(
            transpose=transpose, group_size=group_size, bits=bits, mode=mode, stream=stream
        )
        if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == width:
            if (
                selected == "slot-qmv"
                and transpose is True
                and mode == "affine"
                and stream is None
                and biases is not None
            ):
                from slot_qmv import _array_meta, shape_plan, slot_qmv

                try:
                    shape_plan(
                        *(_array_meta(a) for a in (x, w, scales, biases)),
                        bits=bits,
                        group_size=group_size,
                    )
                except ValueError:
                    pass  # unsupported geometry uses admitted singleton arithmetic
                else:
                    return slot_qmv(x, w, scales, biases, bits=bits, group_size=group_size)
            return mx.concatenate(
                [
                    original(mx.contiguous(x[:, i : i + 1, :]), w, scales, biases, **kwargs)
                    for i in range(width)
                ],
                axis=1,
            )
        return original(x, w, scales, biases, **kwargs)

    selected = mode
    if active and mode in ("singleton-qmm", "slot-qmv"):
        mx.quantized_matmul = singleton
    try:
        yield
    finally:
        mx.quantized_matmul = original


def forward(model, inputs, cache, *, verifier_mode="native", **kwargs):
    from phased_decode import phased_forward

    width = inputs.shape[1]
    with numerical_mode(
        verifier_mode,
        active=width in VERIFY_WIDTHS and kwargs.get("all_logits") is True,
        width=width if width in VERIFY_WIDTHS else 2,
    ):
        return phased_forward(model, inputs, cache, **kwargs)
