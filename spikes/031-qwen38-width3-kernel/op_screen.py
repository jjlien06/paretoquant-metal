"""Per-shape operator screen: stock singleton QMV (reference) vs stock width-W QMM vs slot-QMV (v1) vs slot-QMV word-load (w).

Exactness: bitwise equality to concatenated stock singleton quantized_matmul (the native width-1 arithmetic).
Timing: synchronized, warm-up, interleaved; each timed call evaluates a chain of `chain` matmuls on distinct
weight copies so weights are not cache-resident (>= 2x SLC).
"""
import argparse, json, statistics, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx
import slot_qmv as v1
import slot_qmv_w as vw

SHAPES = {  # (N, K) of 27B projections (Q4 g64)
    'gdn.in_proj_qkv': (10240, 5120), 'gdn.in_proj_z': (6144, 5120), 'gdn.in_proj_b': (48, 5120),
    'gdn.in_proj_a': (48, 5120), 'gdn.out_proj': (5120, 6144), 'mlp.gate': (17408, 5120),
    'mlp.up': (17408, 5120), 'mlp.down': (5120, 17408), 'attn.q_proj': (12288, 5120),
    'attn.k_proj': (1024, 5120), 'attn.v_proj': (1024, 5120), 'attn.o_proj': (5120, 6144),
    'lm_head': (248320, 5120),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--reps', type=int, default=25)
    ap.add_argument('--widths', default='2,3'); a = ap.parse_args()
    mx.random.seed(1); res = []
    for name, (N, K) in SHAPES.items():
        copies = max(2, int(2 * 64e6 // (N * K // 2)) + 1)
        ws = []
        for _ in range(copies):
            w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
            q = mx.quantize(w, group_size=64, bits=4); mx.eval(q); ws.append(q)
        for W in map(int, a.widths.split(',')):
            x = (mx.random.normal((1, W, K))).astype(mx.bfloat16); mx.eval(x)
            def stock_single(q):
                return mx.concatenate([mx.quantized_matmul(mx.contiguous(x[:, i:i+1]), *q, transpose=True, group_size=64, bits=4) for i in range(W)], axis=1)
            fns = {
                'stock_singleton': stock_single,
                'stock_width': lambda q: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=4),
                'slot_v1': lambda q: v1.slot_qmv(x, *q, bits=4, group_size=64),
                'slot_w': lambda q: vw.slot_qmv(x, *q, bits=4, group_size=64),
            }
            ref = stock_single(ws[0]); mx.eval(ref)
            exact = {}
            for k, f in fns.items():
                y = f(ws[0]); mx.eval(y)
                exact[k] = bool(mx.array_equal(y, ref).item())
            t = {k: [] for k in fns}
            for k, f in fns.items():
                for _ in range(3): mx.eval([f(q) for q in ws])
            for r in range(a.reps):
                for k in (fns if r % 2 == 0 else list(reversed(list(fns)))):
                    mx.synchronize(); t0 = time.perf_counter(); mx.eval([fns[k](q) for q in ws]); mx.synchronize()
                    t[k].append((time.perf_counter() - t0) * 1e3 / copies)
            row = dict(shape=name, N=N, K=K, width=W, exact_vs_singleton=exact,
                       median_ms={k: statistics.median(v) for k, v in t.items()}, samples_ms=t)
            res.append(row)
            print(name, W, exact, ' '.join(f'{k}={v:.4f}' for k, v in row['median_ms'].items()), flush=True)
    Path(a.out).write_text(json.dumps(dict(host=platform.node(), mlx=mx.__version__, results=res), indent=1))


if __name__ == '__main__':
    main()
