"""Per-shape geometry tuner. Admits a geometry for (N,K,W) only if bitwise equal to stock singleton QMV on
several random inputs AND faster (median, interleaved, multi-copy to defeat cache) than both stock width-W QMM
and the 030 slot-QMV default (4,2)."""
import argparse, json, statistics, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx
from slot_qmv_t import slot_qmv_t, GEOMETRIES
from op_screen import SHAPES

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--widths', default='2,3'); ap.add_argument('--shapes', default=''); a = ap.parse_args()
    mx.random.seed(2); res = []
    shapes = {k: v for k, v in SHAPES.items() if not a.shapes or k in a.shapes.split(',')}
    for name, (N, K) in shapes.items():
        copies = max(2, int(2 * 64e6 // (N * K // 2)) + 1)
        ws = []
        for _ in range(copies):
            q = mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4); mx.eval(q); ws.append(q)
        for W in map(int, a.widths.split(',')):
            xs = [mx.random.normal((1, W, K)).astype(mx.bfloat16) * s for s in (1.0, 0.1, 7.0)]; mx.eval(xs)
            x = xs[0]
            fns = {'stock_width': lambda q: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=4)}
            exact = {}
            for g in GEOMETRIES:
                if N % (g[0] * g[1]): continue
                ok = True
                for xi in xs:
                    for q in ws[:2]:
                        ref = mx.concatenate([mx.quantized_matmul(mx.contiguous(xi[:, i:i+1]), *q, transpose=True, group_size=64, bits=4) for i in range(W)], 1)
                        y = slot_qmv_t(xi, *q, bits=4, group_size=64, rows=g[0], simds=g[1], mode=g[2])
                        ok &= bool(mx.array_equal(y, ref).item())
                label = '%dx%dm%d' % g
                exact[label] = ok
                if ok:
                    fns[label] = (lambda g: lambda q: slot_qmv_t(x, *q, bits=4, group_size=64, rows=g[0], simds=g[1], mode=g[2]))(g)
            for f in fns.values():
                for _ in range(3): mx.eval([f(q) for q in ws])
            t = {k: [] for k in fns}
            keys = list(fns)
            for r in range(a.reps):
                for k in (keys if r % 2 == 0 else keys[::-1]):
                    mx.synchronize(); t0 = time.perf_counter(); mx.eval([fns[k](q) for q in ws]); mx.synchronize()
                    t[k].append((time.perf_counter() - t0) * 1e3 / copies)
            med = {k: statistics.median(v) for k, v in t.items()}
            cands = {k: v for k, v in med.items() if k != 'stock_width'}
            best = min(cands, key=cands.get) if cands else None
            admitted = best if best and med[best] < med['stock_width'] and med[best] < med.get('4x2m0', 1e9) * 1.0001 else ('4x2m0' if '4x2m0' in med and med['4x2m0'] < med['stock_width'] else None)
            row = dict(shape=name, N=N, K=K, width=W, exact=exact, median_ms=med, best=best, admitted=admitted, samples_ms=t)
            res.append(row)
            print(name, W, 'stock=%.4f 4x2=%.4f best=%s %.4f admitted=%s inexact=%s' % (med['stock_width'], med.get('4x2m0', -1), best, med.get(best, -1), admitted, [k for k, v in exact.items() if not v]), flush=True)
    Path(a.out).write_text(json.dumps(dict(host=platform.node(), mlx=mx.__version__, results=res), indent=1))

if __name__ == '__main__':
    main()
