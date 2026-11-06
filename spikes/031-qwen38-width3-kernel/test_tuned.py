import json
from pathlib import Path
import pytest
import verifier


def test_modes_registered():
    assert {"slot-qmv-tuned-m2", "slot-qmv-tuned-m4"} <= set(verifier.MODES)


def test_width2_and_unlisted_keep_030_mode():
    assert verifier.tuned_launch("slot-qmv-tuned-m4", 17408, 5120, 2, 4, 64) == "slot-qmv"
    assert verifier.tuned_launch("slot-qmv-tuned-m2", 17408, 5120, 2, 4, 64) == "native"
    assert verifier.tuned_launch("slot-qmv-tuned-m4", 8, 64, 3, 4, 64) == "slot-qmv"
    assert verifier.tuned_launch("slot-qmv-tuned-m4", 17408, 5120, 3, 3, 64) == "slot-qmv"


def test_width3_tables_match_file():
    t = json.loads((Path(verifier.__file__).parent / "geometry_table.json").read_text())
    for k, v in t["tables"]["m4"].items():
        n, kk, w = map(int, k.split(","))
        assert verifier.tuned_launch("slot-qmv-tuned-m4", n, kk, w, 4, 64) == tuple(v)
    assert verifier.tuned_launch("slot-qmv-tuned-m2", 248320, 5120, 3, 4, 64) == "native"


def test_unknown_geometry_raises():
    from slot_qmv_t import slot_qmv_t
    from slot_qmv import ArrayMeta  # noqa: F401
    with pytest.raises(ValueError):
        class A:  # minimal meta-only stand-in
            def __init__(s, shape, dtype): s.shape, s.dtype = shape, dtype
        slot_qmv_t(A((1, 3, 512), "bfloat16"), A((64, 64), "uint32"), A((64, 8), "bfloat16"),
                   A((64, 8), "bfloat16"), bits=4, rows=3, simds=3)
