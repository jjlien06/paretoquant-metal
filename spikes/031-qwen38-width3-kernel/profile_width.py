"""Stage-1 profiler: width-1/2/3 verify cost on real Qwen3.8-27B layer shapes (synthetic Q4 weights).

Synchronized (mx.eval of outputs + every live cache), warm-up, paired interleaved order.
Modes: native mx.quantized_matmul vs slot-qmv substitution (verifier.numerical_mode).
Also isolates: quantized matmuls only, GDN recurrence only, full-attention SDPA only.
"""
import argparse, json, statistics, sys, time, platform, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.gated_delta import gated_delta_update

CFG = {'attention_bias': False, 'attn_output_gate': True, 'full_attention_interval': 4, 'head_dim': 256,
       'hidden_act': 'silu', 'hidden_size': 5120, 'intermediate_size': 17408, 'linear_conv_kernel_dim': 4,
       'linear_key_head_dim': 128, 'linear_num_key_heads': 16, 'linear_num_value_heads': 48,
       'linear_value_head_dim': 128, 'model_type': 'qwen3_5_text', 'num_attention_heads': 24,
       'num_key_value_heads': 4, 'rms_norm_eps': 1e-06, 'vocab_size': 1024, 'tie_word_embeddings': False,
       'rope_parameters': {'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'partial_rotary_factor': 0.25,
                           'rope_theta': 10000000, 'rope_type': 'default'}, 'partial_rotary_factor': 0.25,
       'max_position_embeddings': 262144}


def build(nlayers):
    cfg = dict(CFG, num_hidden_layers=nlayers)
    args = qwen3_5.TextModelArgs.from_dict(cfg)
    m = qwen3_5.TextModel(args)
    m.set_dtype(mx.bfloat16)
    nn.quantize(m, group_size=64, bits=4, class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and 'layers' in p)
    mx.eval(m.parameters())
    return m


def timeit(fn, reps):
    out = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); fn(); mx.synchronize(); out.append((time.perf_counter() - t) * 1e3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--out', required=True)
    ap.add_argument('--modes', default='native,slot-qmv')
    ap.add_argument('--skip-qmm', action='store_true')
    a = ap.parse_args()
    mx.random.seed(0)
    m = build(a.layers)
    inner = m.model
    cache = m.make_cache()
    h0 = (mx.random.normal((1, a.context, 5120)) * 0.5).astype(mx.bfloat16)
    # prefill through layers directly
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    def run_layers(h, cache):
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        for layer, c in zip(inner.layers, cache):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        return h
    h = run_layers(h0, cache); mx.eval(h, [c.state for c in cache])
    import verifier

    def step(W, mode):
        x = (mx.random.normal((1, W, 5120)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        def f():
            with verifier.numerical_mode(mode, active=(W in (2, 3)), width=W if W in (2, 3) else 2):
                y = run_layers(x, cache)
            mx.eval(y, [c.state for c in cache])
            for c in cache:
                if not hasattr(c, 'lengths') or c.__class__.__name__ == 'KVCache':
                    pass
            for i in inner.fa_idx_list if hasattr(inner, 'fa_idx_list') else []:
                pass
            for c in cache:
                if c.__class__.__name__ == 'KVCache':
                    c.trim(W)
        return f

    # quantized-matmul-only microbench across all projections of one linear + one full layer
    lin = inner.layers[0]; full = inner.layers[3]
    projs = {}
    for name, mod in list(lin.named_modules()) + [('full.' + n, mm) for n, mm in full.self_attn.named_modules()]:
        if isinstance(mod, nn.QuantizedLinear) and (name.startswith('full.') or name.startswith('linear_attn') or name.startswith('mlp')):
            projs[name] = mod
    from slot_qmv import slot_qmv
    def qmm_fn(W, mode):
        xs = {}
        for n, mod in projs.items():
            K = mod.weight.shape[1] * 8
            xs[n] = (mx.random.normal((1, W, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(list(xs.values()))
        def f():
            ys = []
            for n, mod in projs.items():
                if mode == 'slot-qmv' and W > 1:
                    ys.append(slot_qmv(xs[n], mod.weight, mod.scales, mod.biases, bits=4, group_size=64))
                else:
                    ys.append(mx.quantized_matmul(xs[n], mod.weight, mod.scales, mod.biases, transpose=True, group_size=64, bits=4))
            mx.eval(ys)
        return f

    # GDN recurrence only
    def gdn_fn(W):
        q = mx.random.normal((1, W, 16, 128)).astype(mx.bfloat16); k = mx.random.normal((1, W, 16, 128)).astype(mx.bfloat16)
        v = mx.random.normal((1, W, 48, 128)).astype(mx.bfloat16)
        aa = mx.random.normal((1, W, 48)).astype(mx.bfloat16); bb = mx.random.normal((1, W, 48)).astype(mx.bfloat16)
        st = mx.zeros((1, 48, 128, 128), dtype=mx.float32)
        mx.eval(q, k, v, aa, bb, st)
        def f():
            o, s = gated_delta_update(q, k, v, aa, bb, lin.linear_attn.A_log, lin.linear_attn.dt_bias, st, None)
            mx.eval(o, s)
        return f

    cases = []
    for W in (1, 2, 3):
        for mode in a.modes.split(','):
            if W == 1 and mode != 'native':
                continue
            cases.append(('layers', W, mode, step(W, mode)))
            if not a.skip_qmm and mode in ('native', 'slot-qmv'):
                cases.append(('qmm', W, mode, qmm_fn(W, mode)))
        cases.append(('gdn', W, 'native', gdn_fn(W)))
    for _, _, _, f in cases:
        for _ in range(5):
            f()
    samples = {(k, W, md): [] for k, W, md, _ in cases}
    for r in range(a.reps):
        order = cases if r % 2 == 0 else list(reversed(cases))
        for k, W, md, f in order:
            samples[(k, W, md)] += timeit(f, 1)
    res = []
    for (k, W, md), v in samples.items():
        res.append(dict(kind=k, width=W, mode=md, median_ms=statistics.median(v), min_ms=min(v), n=len(v), samples=v))
        print(f'{k:7s} W={W} {md:9s} median {statistics.median(v):8.3f} ms  min {min(v):8.3f}')
    meta = dict(host=platform.node(), chip=subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip(),
                layers=a.layers, layer_kinds=['linear' if l.is_linear else 'full' for l in inner.layers], context=a.context,
                projections={n: list(m_.weight.shape) for n, m_ in projs.items()}, mlx=mx.__version__,
                note='synthetic random Q4 g64 BF16 weights at real 27B shapes; layers timing includes eval of every live cache')
    Path(a.out).write_text(json.dumps(dict(meta=meta, results=res), indent=1))


if __name__ == '__main__':
    main()
