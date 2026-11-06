"""Op-level bitwise check of a fixed geometry on the 0.8B control shapes (W=3) vs stock singleton QMV."""
import sys, json; sys.path.insert(0, '.')
import mlx.core as mx
from slot_qmv_t import slot_qmv_t
SMALL = [(6144,1024),(2048,1024),(16,1024),(1024,2048),(3584,1024),(1024,3584),(4096,1024),(512,1024)]
g = tuple(map(int, sys.argv[1].split(',')))
res = {}
mx.random.seed(3)
for N, K in SMALL:
    ok = True
    for s in (0.02, 0.5):
        q = mx.quantize((mx.random.normal((N, K)) * s).astype(mx.bfloat16), group_size=64, bits=4)
        for xs in (1.0, 0.05, 9.0):
            x = (mx.random.normal((1, 3, K)) * xs).astype(mx.bfloat16)
            ref = mx.concatenate([mx.quantized_matmul(mx.contiguous(x[:, i:i+1]), *q, transpose=True, group_size=64, bits=4) for i in range(3)], 1)
            ok &= bool(mx.array_equal(slot_qmv_t(x, *q, bits=4, group_size=64, rows=g[0], simds=g[1], mode=g[2]), ref).item())
    res[f'{N},{K},3'] = ok
print(json.dumps(res))
