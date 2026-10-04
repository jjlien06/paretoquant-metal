"""Real GPU numerical tests; random tensors here are explicit test fixtures."""

import platform

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from paretoquant.metal import fused_gate_up, stock_gate_up  # noqa: E402

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        platform.machine() != "arm64" or not mx.metal.is_available(),
        reason="Apple Silicon Metal required",
    ),
]


@pytest.mark.parametrize("bits", [3, 4, 6])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("rows,k,group", [(7, 64, 32), (32, 128, 64), (17, 256, 128)])
def test_fused_matches_stock(bits, dtype, rows, k, group):
    mx.random.seed(17)
    x = mx.random.normal((k,)).astype(dtype)
    gate = mx.quantize(mx.random.normal((rows, k)).astype(dtype) * 0.1, group, bits)
    up = mx.quantize(mx.random.normal((rows, k)).astype(dtype) * 0.1, group, bits)
    actual = fused_gate_up(x, gate, up, bits=bits, group_size=group)
    expected = stock_gate_up(x, gate, up, bits=bits, group_size=group)
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual),
        np.array(expected),
        rtol=0.008 if dtype == mx.float16 else 2e-4,
        atol=0.01 if dtype == mx.float16 else 2e-5,
    )


@pytest.mark.parametrize("rows_per_group", [1, 2, 4, 8])
def test_launch_configuration_preserves_output(rows_per_group):
    mx.random.seed(3)
    x = mx.random.normal((1, 1, 128)).astype(mx.float16)
    gate = mx.quantize(mx.random.normal((19, 128)).astype(mx.float16), 64, 3)
    up = mx.quantize(mx.random.normal((19, 128)).astype(mx.float16), 64, 3)
    actual = fused_gate_up(x, gate, up, bits=3, group_size=64, rows_per_group=rows_per_group)
    expected = stock_gate_up(x, gate, up, bits=3, group_size=64)
    assert actual.shape == (1, 1, 19)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0.01, atol=0.2)


@pytest.mark.parametrize("bits", [3, 4, 6])
def test_packed_kernel_matches_scalar_for_realistic_width(bits):
    mx.random.seed(29)
    x = mx.random.normal((1, 1, 896)).astype(mx.float16)
    gate = mx.quantize(mx.random.normal((257, 896)).astype(mx.float16) * 0.02, 64, bits)
    up = mx.quantize(mx.random.normal((257, 896)).astype(mx.float16) * 0.02, 64, bits)
    actual = fused_gate_up(x, gate, up, bits=bits, variant="packed")
    expected = fused_gate_up(x, gate, up, bits=bits, variant="scalar")
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0.01, atol=0.001)


def test_invalid_kernel_variant_rejected():
    x, q = fixtures()
    with pytest.raises(ValueError, match="variant"):
        fused_gate_up(x, q, q, bits=4, variant="unknown")


@pytest.mark.parametrize("bits", [3, 4, 6])
def test_compiled_kernel_uses_dynamic_weight_inputs(bits):
    from paretoquant.metal import compiled_fused_gate_up

    mx.random.seed(51)
    x = mx.random.normal((1, 1, 128)).astype(mx.float16)
    for _ in range(2):
        gate = mx.quantize(mx.random.normal((33, 128)).astype(mx.float16) * 0.05, 64, bits)
        up = mx.quantize(mx.random.normal((33, 128)).astype(mx.float16) * 0.05, 64, bits)
        actual = compiled_fused_gate_up(x, gate, up, bits=bits)
        expected = stock_gate_up(x, gate, up, bits=bits)
        mx.eval(actual, expected)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0.01, atol=0.001)


def fixtures():
    x = mx.ones((64,), dtype=mx.float32)
    q = mx.quantize(mx.ones((8, 64)), 64, 4)
    return x, q


def test_rejects_multiple_token_inputs():
    x, q = fixtures()
    with pytest.raises(ValueError, match="single"):
        fused_gate_up(mx.stack([x, x]), q, q, bits=4, group_size=64)


@pytest.mark.parametrize("bits", [0, 2, 5, 8])
def test_rejects_unsupported_precision(bits):
    x, q = fixtures()
    with pytest.raises(ValueError, match="bits"):
        fused_gate_up(x, q, q, bits=bits, group_size=64)


def test_rejects_bad_launch_configuration():
    x, q = fixtures()
    with pytest.raises(ValueError, match="rows_per_group"):
        fused_gate_up(x, q, q, bits=4, group_size=64, rows_per_group=3)


def test_rejects_inconsistent_shapes():
    x, q = fixtures()
    up = mx.quantize(mx.ones((7, 64)), 64, 4)
    with pytest.raises(ValueError, match="shape"):
        fused_gate_up(x, q, up, bits=4, group_size=64)
