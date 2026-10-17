"""Actual packed checkpoint matrix screening, not a full-model speed claim.

F16/3-bit paths are explicitly different arithmetic/quantization. Random fixed
activations are operator diagnostics, not a model-quality or calibration set.
"""
import json
import os
from pathlib import Path
import statistics
import time

import mlx.core as mx


def timed(call, trials=15):
    for _ in range(4):
        mx.eval(call())
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        mx.eval(call())
        samples.append(time.perf_counter() - start)
    return {'median_ms': statistics.median(samples) * 1000, 'seconds': samples}


def main():
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(128 * 1024**2)
    mx.random.seed(197)
    fixture = Path(os.environ.get('PARETOQUANT_OPERATOR_FIXTURE', str(Path(__file__).resolve().parents[2] / 'models/qwen38-operator-fixture-v2/operator.safetensors')))
    weights = mx.load(str(fixture))
    rows = []
    for name in ('gate_proj', 'down_proj'):
        prefix = 'language_model.model.layers.0.mlp.' + name
        w, s, b = (weights[prefix + '.' + part] for part in ('weight', 'scales', 'biases'))
        k = w.shape[1] * 8
        raw = mx.random.normal((1, 2, k)).astype(mx.bfloat16)
        mx.eval(raw)
        modes = {}
        for dtype_name, dtype in [('bfloat16', mx.bfloat16), ('float16', mx.float16)]:
            x, scales, biases = raw.astype(dtype), s.astype(dtype), b.astype(dtype)
            mx.eval(x, scales, biases)
            single = lambda: mx.quantized_matmul(x[:, :1, :], w, scales, biases, transpose=True, group_size=64, bits=4)
            pair = lambda: mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=4)
            serial = lambda: mx.concatenate([mx.quantized_matmul(mx.contiguous(x[:, i:i+1, :]), w, scales, biases, transpose=True, group_size=64, bits=4) for i in range(2)], axis=1)
            expected, actual = serial(), pair()
            mx.eval(expected, actual)
            modes[dtype_name] = {'single': timed(single), 'native_pair': timed(pair),
                                 'serial_pair': timed(serial),
                                 'pair_max_abs_vs_singletons': float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))).item()),
                                 'pair_exact_vs_singletons': bool(mx.array_equal(expected, actual).item())}
        # Requantization is a screening candidate only, never a base weight identity claim.
        dense = mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.bfloat16)
        q3, s3, b3 = mx.quantize(dense, group_size=64, bits=3)
        mx.eval(q3, s3, b3)
        reconstructed = mx.dequantize(q3, s3, b3, group_size=64, bits=3)
        relative_mse = mx.mean((dense.astype(mx.float32)-reconstructed.astype(mx.float32))**2)/mx.mean(dense.astype(mx.float32)**2)
        rel = float(relative_mse.item())
        del dense, reconstructed
        one3 = lambda: mx.quantized_matmul(raw[:, :1, :], q3, s3, b3, transpose=True, group_size=64, bits=3)
        modes['requantized_3bit_bf16'] = {'single': timed(one3), 'relative_weight_mse': rel,
                                         'quality_scope': 'weight-only diagnostic; no task/perplexity evidence'}
        rows.append({'projection': name, 'logical_shape': [w.shape[0], k], 'modes': modes})
        del q3, s3, b3, raw
        mx.clear_cache()
    result={'fixture': str(fixture), 'device': mx.device_info(), 'rows': rows,
            'peak_mlx_bytes': mx.get_peak_memory(),
            'not_generation_or_quality_benchmark': True}
    destination = Path(os.environ.get('PARETOQUANT_OPERATOR_OUTPUT', str(Path(__file__).with_name('operator-screen-m2-v1.json'))))
    with destination.open('x') as out:json.dump(result, out, indent=2)
    for row in rows:
        print(row['projection'],{name:{key:round(value['median_ms'],4) for key,value in mode.items() if isinstance(value,dict) and 'median_ms' in value} for name,mode in row['modes'].items()})
    print('peak_mlx_bytes',result['peak_mlx_bytes'])

if __name__=='__main__':main()
