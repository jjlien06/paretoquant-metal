"""Numerical admission: unchanged 012 atol=1e-6, rtol=1e-5, plus exact argmax.

No CLI switches exist to relax these. Same-width phased/native equality is NOT
width2/width1 admission. The latter must be exercised on the actual checkpoint.
"""
import mlx.core as mx
ATOL=1e-6
RTOL=1e-5


def assert_logits_equal(native, candidate):
    assert native.shape==candidate.shape and native.dtype==candidate.dtype, 'logit metadata'
    assert mx.allclose(native,candidate,atol=ATOL,rtol=RTOL).item(), 'logit numerical gate'
    assert mx.array_equal(mx.argmax(native,-1),mx.argmax(candidate,-1)).item(), 'greedy argmax gate'
