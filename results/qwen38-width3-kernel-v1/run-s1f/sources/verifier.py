"""Explicit per-rank numerical mode. Only active on width2/width3 teacher verification.

030: generalized from fixed width2 to the explicit verify width W in VERIFY_WIDTHS.
"""

VERIFY_WIDTHS = (2, 3)

from contextlib import contextmanager

MODES = ("native", "singleton-qmm", "slot-qmv", "slot-qmv-tuned-m2", "slot-qmv-tuned-m4")
TUNED = {"slot-qmv-tuned-m2": "m2", "slot-qmv-tuned-m4": "m4"}


from functools import lru_cache


@lru_cache(maxsize=1)
def _tables():
    import json
    from pathlib import Path

    return json.loads((Path(__file__).resolve().parent / "geometry_table.json").read_text())


def tuned_launch(mode, n, k, width, bits, group_size):
    """031: explicit per-host width-3 launch table (geometry_table.json, source-registered).
    Returns (rows, simds, kmode) or 'native' or 'slot-qmv' (030 default geometry).
    Width 2 and unlisted shapes use the host's 030-admitted mode (m2 native, m4 slot-qmv)."""
    table = _tables()
    host = TUNED[mode]
    default = table["defaults"][host]
    if width != 3 or bits != 4 or group_size != 64:
        return default
    entry = table["tables"][host].get(f"{n},{k},{width}", default)
    return tuple(entry) if isinstance(entry, list) else entry


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
                selected in TUNED
                and transpose is True
                and mode == "affine"
                and stream is None
                and biases is not None
            ):
                launch = tuned_launch(selected, w.shape[0], x.shape[-1], width, bits, group_size)
                if launch == "native":
                    return original(x, w, scales, biases, **kwargs)
                if launch != "slot-qmv":
                    from slot_qmv_t import slot_qmv_t

                    rows, simds, kmode = launch
                    return slot_qmv_t(
                        x, w, scales, biases, bits=bits, group_size=group_size,
                        rows=rows, simds=simds, mode=kmode,
                    )
            if (
                selected in ("slot-qmv",) + tuple(TUNED)
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
    if active and mode in ("singleton-qmm", "slot-qmv") + tuple(TUNED):
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
