"""Width-W verifier numerical modes (generalizes 021 verifier.py, width2 only).

native        : stock batched quantized_matmul at width W.
singleton-qmm : every [1, W, K] quantized_matmul is computed as W independent
                width-1 calls (batch-invariant with sequential decode).
slot-qmv      : copied 021 slot-shared Metal QMV kernel (its shape_plan already
                admits 1..9 slots); unsupported geometry falls back to singleton.
Only active inside an explicit verify of exactly `width` tokens.
"""

from contextlib import contextmanager

MODES = ("native", "singleton-qmm", "slot-qmv")


@contextmanager
def numerical_mode(mode, *, width, active=True):
    if mode not in MODES:
        raise ValueError("Unknown verifier mode")
    import mlx.core as mx

    original = mx.quantized_matmul
    counters = {"slot_qmv": 0, "singleton": 0}

    def substitute(x, w, scales, biases=None, transpose=True, group_size=None, bits=None,
                   mode="affine", *, stream=None):
        kwargs = dict(transpose=transpose, group_size=group_size, bits=bits, mode=mode,
                      stream=stream)
        if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == width:
            if (selected == "slot-qmv" and transpose is True and mode == "affine"
                    and stream is None and biases is not None):
                from slot_qmv import _array_meta, shape_plan, slot_qmv

                try:
                    shape_plan(*(_array_meta(a) for a in (x, w, scales, biases)),
                               bits=bits, group_size=group_size)
                except ValueError:
                    pass
                else:
                    counters["slot_qmv"] += 1
                    return slot_qmv(x, w, scales, biases, bits=bits, group_size=group_size)
            counters["singleton"] += 1
            return mx.concatenate(
                [original(mx.contiguous(x[:, i:i + 1, :]), w, scales, biases, **kwargs)
                 for i in range(width)], axis=1)
        return original(x, w, scales, biases, **kwargs)

    selected = mode
    if active and width >= 2 and mode in ("singleton-qmm", "slot-qmv"):
        mx.quantized_matmul = substitute
    try:
        yield counters
    finally:
        mx.quantized_matmul = original
